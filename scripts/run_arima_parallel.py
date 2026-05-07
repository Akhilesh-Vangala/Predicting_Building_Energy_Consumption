"""Parallel ARIMA fitting across meters using ProcessPoolExecutor.

Each meter's ARIMA model is independent, so we can fit N_WORKERS meters
simultaneously. macOS requires the 'spawn' multiprocessing context and
thread-count env vars to be set before any numpy/pmdarima import.

Usage:
    python -m scripts.run_arima_parallel --config configs/default.yaml

Outputs:
    results/metrics/models_engineered.json  (updated with arima row)
    results/tables/models_engineered.csv    (updated)
"""
from __future__ import annotations

# Thread-count env vars must be set before ANY numpy/scipy import.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.pipeline import evaluate_predictions, prepare_data
from src.utils import load_json, save_json, set_seed, setup_logging

logger = logging.getLogger(__name__)

# --- tunables ---
N_WORKERS = 4       # CPU cores to use; leave at least 4 for the LSTM run
N_METERS  = 30      # number of meters to fit ARIMA on
MAX_P     = 2
MAX_Q     = 2
MAX_ORDER = 4
STEPWISE  = True
# ----------------


def _fit_one_meter(args: tuple) -> dict | None:
    """Worker function: fit ARIMA for a single (building_id, meter) pair.

    Runs in a spawned subprocess — all imports happen here so the worker
    process is clean and macOS-safe.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    (bid, mtr, train_ts, train_y, val_ts, val_y,
     max_p, max_q, max_order, stepwise) = args

    try:
        from pmdarima import auto_arima
        import warnings
        warnings.filterwarnings("ignore")

        # Match src/models/arima.py exactly:
        # fit on log1p-space, use SARIMA m=24, expm1 predictions back to real space.
        y_train_log = np.log1p(np.clip(train_y, 0, None))

        series = pd.Series(y_train_log, index=pd.to_datetime(train_ts))
        series = series.asfreq("h").interpolate(method="time").fillna(method="bfill").fillna(0)

        if len(series) < 2 * 24 or np.all(series.values == series.values[0]):
            raise ValueError("series too short or constant")

        model = auto_arima(
            series,
            seasonal=True,
            m=24,
            max_p=max_p, max_q=max_q,
            max_order=max_order,
            stepwise=stepwise,
            suppress_warnings=True,
            error_action="ignore",
            information_criterion="aic",
        )

        n_val = len(val_ts)
        forecast_log = model.predict(n_periods=n_val)
        preds_real = np.clip(np.expm1(forecast_log), 0, None)

        ts_strs = pd.to_datetime(val_ts).strftime("%Y-%m-%d %H:%M:%S").tolist()

        return {
            "building_id": int(bid),
            "meter": int(mtr),
            "pred": preds_real.tolist(),
            "actual": val_y.tolist(),
            "val_ts": ts_strs,
        }
    except Exception as e:
        return {"building_id": int(bid), "meter": int(mtr), "error": str(e)}


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-meters", type=int, default=N_METERS)
    parser.add_argument("--n-workers", type=int, default=N_WORKERS)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.random_state)

    logger.info("Loading data...")
    prep = prepare_data(cfg, feature_set="engineered", target_log=False)

    keys = ["building_id", "meter"]
    ts_col = "timestamp"

    # Stratified sample: all 4 meter types represented proportionally.
    all_keys = (
        prep.train_full[keys]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    n = args.n_meters
    parts = [
        g.sample(n=max(1, round(n * len(g) / len(all_keys))), random_state=42)
        for _, g in all_keys.groupby("meter", group_keys=False)
    ]
    selected = pd.concat(parts).drop_duplicates().head(n)
    logger.info("Selected %d meters (stratified by meter type).", len(selected))

    # Build per-meter train/val numpy arrays to pass to workers.
    train_full = prep.train_full.sort_values(keys + [ts_col])
    val_full   = prep.val_full.sort_values(keys + [ts_col])

    tasks = []
    for _, row in selected.iterrows():
        bid, mtr = int(row["building_id"]), int(row["meter"])
        tr = train_full[(train_full.building_id == bid) & (train_full.meter == mtr)]
        vl = val_full[(val_full.building_id == bid) & (val_full.meter == mtr)]
        if len(tr) < 50 or len(vl) == 0:
            continue
        tasks.append((
            bid, mtr,
            tr[ts_col].to_numpy(),
            tr["meter_reading"].to_numpy(dtype=np.float64),
            vl[ts_col].to_numpy(),
            vl["meter_reading"].to_numpy(dtype=np.float64),
            MAX_P, MAX_Q, MAX_ORDER, STEPWISE,
        ))

    logger.info("Submitting %d tasks to %d workers...", len(tasks), args.n_workers)
    t0 = time.perf_counter()

    ctx = mp.get_context("spawn")   # required for macOS safety with pmdarima
    results = []
    done = 0

    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_fit_one_meter, task): task for task in tasks}
        for future in as_completed(futures):
            done += 1
            res = future.result()
            if res and "error" not in res:
                results.append(res)
                logger.info("[%d/%d] building=%d meter=%d  OK",
                            done, len(tasks), res["building_id"], res["meter"])
            else:
                bid = res.get("building_id", "?") if res else "?"
                err = res.get("error", "unknown") if res else "exception"
                logger.warning("[%d/%d] building=%d FAILED: %s", done, len(tasks), bid, err)

    elapsed = time.perf_counter() - t0
    logger.info("Done. %.1f min, %d/%d meters succeeded.",
                elapsed / 60, len(results), len(tasks))

    if not results:
        logger.error("No ARIMA results collected.")
        return

    # Assemble into a DataFrame for scoring.
    all_rows = []
    for r in results:
        bid, mtr = r["building_id"], r["meter"]
        for ts_str, pred_val, actual_val in zip(r["val_ts"], r["pred"], r["actual"]):
            all_rows.append({"building_id": bid, "meter": mtr,
                             "ts_str": ts_str, "pred": float(pred_val),
                             "actual": float(actual_val)})

    pred_df = pd.DataFrame(all_rows)

    val_meta = val_full.copy()
    val_meta["ts_str"] = pd.to_datetime(val_meta[ts_col]).dt.strftime("%Y-%m-%d %H:%M:%S")
    available = [c for c in ["building_id", "meter", ts_col, "meter_reading",
                              "primary_use", "site_id", "ts_str"]
                 if c in val_meta.columns]
    merged = pred_df.merge(val_meta[available], on=["building_id", "meter", "ts_str"],
                           how="inner")

    if merged.empty:
        logger.error("Merge empty — timestamp alignment failed.")
        return

    y_pred_real = np.clip(merged["pred"].to_numpy(dtype=np.float64), 0, None)
    y_true_real = np.clip(merged["actual"].to_numpy(dtype=np.float64), 0, None)
    preds_log   = np.log1p(y_pred_real).astype(np.float32)
    y_true_log  = np.log1p(y_true_real).astype(np.float32)

    eval_payload = evaluate_predictions(merged, preds_log, y_true_log, target_log=True)
    overall = eval_payload["overall"]

    logger.info("[arima] RMSE=%.2f | MAE=%.2f | CV-RMSE=%.3f | n_meters=%d",
                overall["rmse"], overall["mae"], overall["cv_rmse"], len(results))

    # Persist predictions + update results tables.
    pred_dir = cfg.paths.metrics / "predictions_engineered"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "arima_log_preds.npy", preds_log)

    new_row = {"model": "arima", "family": "time_series",
               **overall, "train_seconds": float(elapsed)}

    metrics_path = cfg.paths.metrics / "models_engineered.json"
    table_path   = cfg.paths.tables  / "models_engineered.csv"

    existing = load_json(metrics_path) if metrics_path.exists() else {"models": {}}
    existing.setdefault("models", {})["arima"] = {
        "family": "time_series", "metrics": overall,
        "by_primary_use": eval_payload["by_primary_use"],
        "by_meter": eval_payload["by_meter"],
        "by_site": eval_payload["by_site"],
        "feature_importance": None,
        "train_seconds": float(elapsed),
        "n_meters": len(results),
        "n_predicted_rows": len(merged),
    }
    save_json(existing, metrics_path)

    df = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()
    df = df[df["model"] != "arima"] if not df.empty else df
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True).sort_values("rmse")
    df.to_csv(table_path, index=False)
    logger.info("Updated %s and %s", table_path, metrics_path)


if __name__ == "__main__":
    main()
