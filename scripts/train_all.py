from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd

from src.config import load_config
from src.models import (
    ARIMAModel, DecisionTreeModel, ElasticNetModel, LassoModel,
    LightGBMModel, LSTMModel, MLPModel, OLSModel, RandomForestModel, RidgeModel,
)
from src.pipeline import baseline_lag_24h, baseline_meter_mean, evaluate_predictions, prepare_data
from src.utils import load_json, save_json, set_seed, setup_logging, timer

logger = logging.getLogger(__name__)


def _build_models(cfg) -> list:
    p = cfg.models
    return [
        ("ols", "linear", OLSModel()),
        ("ridge", "linear", RidgeModel(**p.get("linear", {}))),
        ("lasso", "linear", LassoModel(**p.get("linear", {}))),
        ("elasticnet", "linear", ElasticNetModel(**p.get("linear", {}))),
        ("decision_tree", "trees", DecisionTreeModel(max_depth=p.get("decision_tree", {}).get("max_depth", [16])[-1])),
        ("random_forest", "trees", RandomForestModel(**p.get("random_forest", {}))),
        ("lightgbm", "trees", LightGBMModel(**p.get("lightgbm", {}))),
        ("mlp", "neural", MLPModel(**p.get("mlp", {}))),
    ]


def _run_arima(cfg, prep, rows: list, detailed: dict, pred_dir: Path) -> None:
    import time

    logger.info("=== fitting arima (time_series) ===")
    arima = ARIMAModel(**cfg.models.get("arima", {}))
    t0 = time.perf_counter()
    val_with_pred = arima.fit_predict_per_meter(
        prep.train_full, prep.val_full,
        target_col="meter_reading",
        max_meters=cfg.models.get("arima", {}).get("max_meters", None),
        seasonal_period=int(cfg.models.get("arima", {}).get("m", 24)),
        max_p=int(cfg.models.get("arima", {}).get("max_p", 3)),
        max_q=int(cfg.models.get("arima", {}).get("max_q", 3)),
        stepwise=bool(cfg.models.get("arima", {}).get("stepwise", True)),
    )
    elapsed = time.perf_counter() - t0
    val_with_pred = val_with_pred.dropna(subset=["pred"])
    if len(val_with_pred) == 0:
        logger.warning("ARIMA produced no predictions; skipping")
        return

    y_pred_real = val_with_pred["pred"].astype(np.float64).to_numpy()
    y_true_real = val_with_pred["meter_reading"].astype(np.float64).to_numpy()
    eval_payload = evaluate_predictions(val_with_pred, np.log1p(np.clip(y_pred_real, 0, None)),
                                        np.log1p(np.clip(y_true_real, 0, None)), target_log=True)
    row = {
        "model": "arima", "family": "time_series",
        **eval_payload["overall"], "train_seconds": float(elapsed),
    }
    rows.append(row)
    detailed["models"]["arima"] = {
        "family": "time_series",
        "metrics": eval_payload["overall"],
        "by_primary_use": eval_payload["by_primary_use"],
        "by_meter": eval_payload["by_meter"],
        "by_site": eval_payload["by_site"],
        "feature_importance": None,
        "train_seconds": float(elapsed),
        "n_predicted_rows": int(len(val_with_pred)),
    }
    np.save(pred_dir / "arima_real_preds.npy", y_pred_real)
    logger.info("[arima] RMSE=%.2f | MAE=%.2f | CV-RMSE=%.3f",
                row["rmse"], row["mae"], row["cv_rmse"])


def _run_lstm(cfg, prep, rows: list, detailed: dict, pred_dir: Path) -> None:
    import time

    logger.info("=== fitting lstm (time_series) ===")
    lstm = LSTMModel(**cfg.models.get("lstm", {}))
    t0 = time.perf_counter()
    preds_log, y_true_log, val_idx = lstm.fit_predict_sequences(
        prep.train_full, prep.val_full,
        feature_cols=prep.feature_cols,
        target_col="meter_reading",
    )
    elapsed = time.perf_counter() - t0
    if len(val_idx) == 0:
        logger.warning("LSTM produced no predictions; skipping")
        return

    val_subset = prep.val_full.loc[val_idx]
    eval_payload = evaluate_predictions(val_subset, preds_log, y_true_log, target_log=True)
    row = {
        "model": "lstm", "family": "time_series",
        **eval_payload["overall"], "train_seconds": float(elapsed),
    }
    rows.append(row)
    detailed["models"]["lstm"] = {
        "family": "time_series",
        "metrics": eval_payload["overall"],
        "by_primary_use": eval_payload["by_primary_use"],
        "by_meter": eval_payload["by_meter"],
        "by_site": eval_payload["by_site"],
        "feature_importance": None,
        "train_seconds": float(elapsed),
        "n_predicted_rows": int(len(val_idx)),
    }
    np.save(pred_dir / "lstm_real_preds.npy", preds_log)
    logger.info("[lstm] RMSE=%.2f | MAE=%.2f | CV-RMSE=%.3f",
                row["rmse"], row["mae"], row["cv_rmse"])


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-set", choices=["engineered", "raw"], default="engineered")
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--refit", action="store_true",
                        help="ignore saved predictions and refit every model")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.random_state)

    logger.info("Preparing data with feature_set=%s", args.feature_set)
    with timer("prepare_data"):
        prep = prepare_data(cfg, feature_set=args.feature_set, target_log=True)

    X_train = prep.train_full[prep.feature_cols]
    X_val = prep.val_full[prep.feature_cols]
    y_train = prep.target_log_train
    y_val = prep.target_log_val

    metrics_path = cfg.paths.metrics / f"models_{args.feature_set}.json"
    table_path = cfg.paths.tables / f"models_{args.feature_set}.csv"
    pred_dir = cfg.paths.metrics / f"predictions_{args.feature_set}"
    pred_dir.mkdir(parents=True, exist_ok=True)

    bl_mean = baseline_meter_mean(prep)
    bl_lag = baseline_lag_24h(prep)

    rows: list[dict] = []
    rows.append({"model": bl_mean["name"], "family": "baseline", **bl_mean["metrics"], "train_seconds": 0.0})
    rows.append({"model": bl_lag["name"], "family": "baseline", **bl_lag["metrics"], "train_seconds": 0.0})

    detailed: dict = {
        "config": {
            "feature_set": args.feature_set,
            "site_subset": cfg.data.site_subset if not cfg.data.use_full else "all",
            "n_train": prep.summary["n_train"],
            "n_val": prep.summary["n_val"],
            "feature_cols": prep.feature_cols,
        },
        "baselines": {
            "meter_mean": {"metrics": bl_mean["metrics"]},
            "lag_24h": {"metrics": bl_lag["metrics"]},
        },
        "models": {},
    }

    prior_loaded: set[str] = set()
    if not args.refit and metrics_path.exists():
        prior = load_json(metrics_path)
        prior_models = prior.get("models", {}) if isinstance(prior, dict) else {}
        for prior_name, prior_payload in prior_models.items():
            if args.only is not None and prior_name in args.only:
                continue
            if prior_name in args.skip:
                continue
            detailed["models"][prior_name] = prior_payload
            metrics = prior_payload.get("metrics", {})
            if not metrics:
                continue
            rows.append({
                "model": prior_name,
                "family": prior_payload.get("family", "unknown"),
                **metrics,
                "train_seconds": float(prior_payload.get("train_seconds", 0.0)),
            })
            prior_loaded.add(prior_name)

    for name, family, model in _build_models(cfg):
        if args.only is not None and name not in args.only:
            continue
        if name in args.skip:
            continue

        pred_file = pred_dir / f"{name}_log_preds.npy"
        if not args.refit and pred_file.exists():
            logger.info("=== %s (%s): reusing saved predictions ===", name, family)
            preds_log = np.load(pred_file)
            train_seconds = 0.0
            importance = None
        else:
            logger.info("=== fitting %s (%s) ===", name, family)
            with timer(f"fit_{name}"):
                if isinstance(model, (LightGBMModel, MLPModel)):
                    model.fit(X_train, y_train, eval_set=(X_val, y_val))
                    model.train_seconds_ = getattr(model, "train_seconds_", 0.0)
                    preds_log = model.predict(X_val)
                else:
                    preds_log, _ = model.fit_predict(X_train, y_train, X_val)
            np.save(pred_file, preds_log)
            train_seconds = float(getattr(model, "train_seconds_", 0.0))
            importance = model.feature_importance()

        eval_payload = evaluate_predictions(prep.val_full, preds_log, y_val, target_log=True)
        row = {
            "model": name,
            "family": family,
            **eval_payload["overall"],
            "train_seconds": train_seconds,
        }
        if name not in prior_loaded:
            rows.append(row)
        detailed["models"][name] = {
            "family": family,
            "metrics": eval_payload["overall"],
            "by_primary_use": eval_payload["by_primary_use"],
            "by_meter": eval_payload["by_meter"],
            "by_site": eval_payload["by_site"],
            "feature_importance": importance,
            "train_seconds": train_seconds,
        }

        df_partial = pd.DataFrame(rows).sort_values("rmse")
        df_partial.to_csv(table_path, index=False)
        save_json(detailed, metrics_path)

        logger.info("[%s] RMSE=%.2f | MAE=%.2f | CV-RMSE=%.3f",
                    name, row["rmse"], row["mae"], row["cv_rmse"])

    if args.feature_set == "engineered":
        if (args.only is None or "arima" in args.only) and "arima" not in args.skip:
            _run_arima(cfg, prep, rows, detailed, pred_dir)
    if (args.only is None or "lstm" in args.only) and "lstm" not in args.skip:
        _run_lstm(cfg, prep, rows, detailed, pred_dir)

    df = pd.DataFrame(rows).sort_values("rmse")
    df.to_csv(table_path, index=False)
    save_json(detailed, metrics_path)

    np.save(pred_dir / "baseline_meter_mean.npy", bl_mean["predictions"])
    if bl_lag["predictions"] is not None:
        np.save(pred_dir / "baseline_lag_24h.npy", bl_lag["predictions"])
    np.save(pred_dir / "y_true_raw.npy", prep.target_raw_val)

    logger.info("\n%s", df.to_string(index=False))
    logger.info("Wrote %s and %s", table_path, metrics_path)


if __name__ == "__main__":
    main()
