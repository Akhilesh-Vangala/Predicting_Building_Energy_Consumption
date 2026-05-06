"""
Parallel ARIMA fitting — runs each meter in a separate process.
Replaces the serial loop in train_all.py for ARIMA.
Results are merged into models_engineered.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))


def _fit_one_meter(args: tuple) -> dict | None:
    """Fit ARIMA for a single (building_id, meter) pair. Runs in a worker process."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    (building_id, meter, train_series, val_series, arima_params, idx, total) = args

    import warnings
    warnings.filterwarnings("ignore")

    from pmdarima import auto_arima

    try:
        model = auto_arima(
            train_series,
            seasonal=arima_params.get("seasonal", False),
            m=arima_params.get("m", 24),
            max_p=arima_params.get("max_p", 2),
            max_q=arima_params.get("max_q", 2),
            max_order=arima_params.get("max_order", 4),
            stepwise=arima_params.get("stepwise", True),
            suppress_warnings=True,
            error_action="ignore",
        )
        preds = model.predict(n_periods=len(val_series))
        preds = np.clip(preds, 0, None)
        print(f"[{idx}/{total}] bld={building_id} meter={meter} order={model.order} OK", flush=True)
        return {
            "building_id": int(building_id),
            "meter": int(meter),
            "preds": preds.tolist(),
            "actuals": val_series.tolist(),
            "train_len": len(train_series),
        }
    except Exception as e:
        print(f"[{idx}/{total}] bld={building_id} meter={meter} FAILED: {e}", flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--feature-set", default="engineered")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    log = logging.getLogger(__name__)

    from src.config import load_config
    from src.pipeline import prepare_data, evaluate_predictions
    from src.utils import load_json, save_json

    cfg = load_config(args.config)
    arima_params = cfg.models.get("arima", {})
    max_meters = arima_params.get("max_meters", 100)

    log.info("Loading data...")
    t0 = time.perf_counter()
    prep = prepare_data(cfg, feature_set=args.feature_set)

    # Sample meters
    keys = ["building_id", "meter"]
    all_keys = prep.train_full[keys].drop_duplicates()
    if max_meters and len(all_keys) > max_meters:
        all_keys = all_keys.sample(n=int(max_meters), random_state=42)
    log.info("Fitting ARIMA on %d meters with %d workers", len(all_keys), args.n_jobs)

    # Build per-meter series for workers
    target_col = "meter_reading"
    timestamp_col = "timestamp"
    work_items = []
    for idx, (_, row) in enumerate(all_keys.iterrows(), 1):
        bid, meter = int(row["building_id"]), int(row["meter"])
        train_m = prep.train_full[(prep.train_full["building_id"] == bid) & (prep.train_full["meter"] == meter)]
        val_m   = prep.val_full[(prep.val_full["building_id"] == bid) & (prep.val_full["meter"] == meter)]
        if len(train_m) < 50 or len(val_m) == 0:
            continue
        train_series = train_m.sort_values(timestamp_col)[target_col].to_numpy(dtype=np.float64)
        val_series   = val_m.sort_values(timestamp_col)[target_col].to_numpy(dtype=np.float64)
        work_items.append((bid, meter, train_series, val_series, dict(arima_params), idx, len(all_keys)))

    log.info("Data prep done in %.1fs, submitting %d tasks", time.perf_counter() - t0, len(work_items))

    # Run in parallel
    t1 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=__import__("multiprocessing").get_context("spawn")) as pool:
        futures = {pool.submit(_fit_one_meter, item): item for item in work_items}
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                results.append(r)

    elapsed = time.perf_counter() - t1
    log.info("Parallel fitting done: %d/%d meters in %.1fs", len(results), len(work_items), elapsed)

    if not results:
        log.error("No ARIMA results — exiting")
        return

    # Aggregate predictions across all meters
    all_preds   = np.concatenate([np.array(r["preds"])   for r in results])
    all_actuals = np.concatenate([np.array(r["actuals"]) for r in results])

    # Evaluate: clip at 0, no upper clip (real-space baseline approach)
    all_preds   = np.clip(all_preds, 0, None)
    all_actuals = np.clip(all_actuals, 0, None)

    rmse_val  = float(np.sqrt(np.mean((all_preds - all_actuals) ** 2)))
    mae_val   = float(np.mean(np.abs(all_preds - all_actuals)))
    mean_y    = float(np.mean(all_actuals)) if np.mean(all_actuals) > 0 else 1.0
    cv_rmse   = rmse_val / mean_y
    rmsle_val = float(np.sqrt(np.mean((np.log1p(all_preds) - np.log1p(all_actuals)) ** 2)))

    log.info("[arima] RMSE=%.2f | MAE=%.2f | CV-RMSE=%.3f | meters=%d | rows=%d",
             rmse_val, mae_val, cv_rmse, len(results), len(all_preds))

    # Merge into existing metrics JSON
    metrics_path = cfg.paths.metrics / f"models_{args.feature_set}.json"
    table_path   = cfg.paths.tables  / f"models_{args.feature_set}.csv"

    existing = load_json(metrics_path) if metrics_path.exists() else {"models": {}, "baselines": {}}
    existing.setdefault("models", {})

    existing["models"]["arima"] = {
        "family": "time_series",
        "metrics": {
            "rmse": rmse_val,
            "mae": mae_val,
            "cv_rmse": cv_rmse,
            "rmsle": rmsle_val,
        },
        "by_meter": [],
        "by_primary_use": [],
        "by_site": [],
        "feature_importance": None,
        "train_seconds": float(elapsed),
        "n_predicted_rows": len(all_preds),
        "n_meters": len(results),
    }

    save_json(existing, metrics_path)
    log.info("Saved to %s", metrics_path)

    # Rebuild CSV table
    rows = []
    for name, m in existing["models"].items():
        met = m["metrics"]
        rows.append({
            "model": name, "family": m["family"],
            "rmse": met["rmse"], "mae": met["mae"],
            "cv_rmse": met["cv_rmse"], "rmsle": met["rmsle"],
            "train_seconds": m.get("train_seconds", 0),
        })
    for name, b in existing.get("baselines", {}).items():
        met = b["metrics"]
        rows.append({
            "model": f"baseline_{name}", "family": "baseline",
            "rmse": met["rmse"], "mae": met["mae"],
            "cv_rmse": met["cv_rmse"], "rmsle": met["rmsle"],
            "train_seconds": 0,
        })
    pd.DataFrame(rows).to_csv(table_path, index=False)
    log.info("Updated %s", table_path)


if __name__ == "__main__":
    main()
