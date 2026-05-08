"""
Additional analyses for paper:
  1. Worst-predicted buildings audit (per-(building_id, meter) RMSE/bias/variance)
  2. Per-primary-use LightGBM vs global LightGBM
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline import prepare_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PRED_DIR = ROOT / "results/metrics/predictions_engineered"
TABLES_DIR = ROOT / "results/tables"


# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def bias(a, b):
    return float(np.mean(a - b))

def variance_component(r, b):
    return float(np.sqrt(max(r**2 - b**2, 0)))


# ---------------------------------------------------------------
# 1. Worst-predicted buildings audit
# ---------------------------------------------------------------

def worst_buildings_audit(val_df: pd.DataFrame, y_true: np.ndarray, y_pred_rf: np.ndarray):
    logger.info("Running worst-buildings audit …")

    df = val_df[["building_id", "meter", "primary_use", "site_id"]].copy().reset_index(drop=True)
    df["y_true"] = y_true
    df["y_pred"] = y_pred_rf
    df["resid"] = df["y_pred"] - df["y_true"]

    rows = []
    for (bid, meter), grp in df.groupby(["building_id", "meter"]):
        r = rmse(grp["y_pred"].values, grp["y_true"].values)
        b = bias(grp["y_pred"].values, grp["y_true"].values)
        v = variance_component(r, b)
        rows.append({
            "building_id": bid,
            "meter": meter,
            "primary_use": grp["primary_use"].iloc[0],
            "site_id": grp["site_id"].iloc[0],
            "n_rows": len(grp),
            "mean_actual_kwh": float(grp["y_true"].mean()),
            "rmse": r,
            "bias": b,
            "variance_component": v,
        })

    audit = pd.DataFrame(rows).sort_values("rmse", ascending=False)
    out = TABLES_DIR / "worst_buildings_audit.csv"
    audit.to_csv(out, index=False)
    logger.info("Saved %s (top-5 by RMSE):", out)
    logger.info("\n%s", audit.head(20).to_string(index=False))
    return audit


# ---------------------------------------------------------------
# 2. Per-primary-use LightGBM vs global
# ---------------------------------------------------------------

def per_use_lightgbm(cfg, val_df: pd.DataFrame, train_df: pd.DataFrame,
                     feature_cols: list[str], y_true: np.ndarray,
                     global_lgbm_preds: np.ndarray):
    import lightgbm as lgb

    logger.info("Training per-primary-use LightGBM …")

    lgb_params = {
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 1500,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
        "early_stopping_rounds": 50,
        "verbose": -1,
        "n_jobs": 4,
        "random_state": 42,
    }

    val_idx = val_df.index if hasattr(val_df, "index") else np.arange(len(val_df))
    val_reset = val_df.reset_index(drop=True)
    train_reset = train_df.reset_index(drop=True)

    y_train_log = np.log1p(train_reset["meter_reading"].values)
    y_val_log = np.log1p(val_reset["meter_reading"].values)

    uses = val_reset["primary_use"].unique()
    rows = []

    y_pred_per_use = np.full(len(val_reset), np.nan)

    for use in sorted(uses):
        tr_mask = train_reset["primary_use"] == use
        va_mask = val_reset["primary_use"] == use

        n_tr = tr_mask.sum()
        n_va = va_mask.sum()
        if n_tr < 100 or n_va < 10:
            logger.warning("Skipping %s: too few rows (train=%d, val=%d)", use, n_tr, n_va)
            continue

        X_tr = train_reset.loc[tr_mask, feature_cols].values
        X_va = val_reset.loc[va_mask, feature_cols].values
        y_tr = y_train_log[tr_mask.values]
        y_va = y_val_log[va_mask.values]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        preds_log = model.predict(X_va)
        preds_raw = np.expm1(np.clip(preds_log, 0, 14))
        actual_raw = val_reset.loc[va_mask, "meter_reading"].values

        r = rmse(preds_raw, actual_raw)

        # global model on same rows
        global_r = rmse(global_lgbm_preds[va_mask.values], actual_raw)

        y_pred_per_use[va_mask.values] = preds_raw

        rows.append({
            "primary_use": use,
            "n_val": int(n_va),
            "rmse_per_use_lgbm": r,
            "rmse_global_lgbm": global_r,
            "delta": global_r - r,
            "pct_improvement": 100 * (global_r - r) / global_r,
        })
        logger.info("  %-40s n=%6d  per-use=%8.0f  global=%8.0f  Δ=%+.0f",
                    use, n_va, r, global_r, global_r - r)

    result = pd.DataFrame(rows).sort_values("rmse_per_use_lgbm", ascending=False)
    out = TABLES_DIR / "per_primary_use_lgbm.csv"
    result.to_csv(out, index=False)
    logger.info("Saved %s", out)
    logger.info("\n%s", result.to_string(index=False))

    # aggregate RMSE across all rows where we made a prediction
    covered = ~np.isnan(y_pred_per_use)
    agg_r = rmse(y_pred_per_use[covered], y_true[covered])
    global_agg_r = rmse(global_lgbm_preds[covered], y_true[covered])
    logger.info("Aggregate (covered rows) — per-use: %.0f  global: %.0f", agg_r, global_agg_r)

    return result


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------

def main():
    cfg = load_config(str(ROOT / "configs/default.yaml"))
    prep = prepare_data(cfg)

    val_df = prep.val_full.reset_index(drop=True)
    train_df = prep.train_full.reset_index(drop=True)
    feature_cols = prep.feature_cols

    y_true = np.load(PRED_DIR / "y_true_raw.npy")
    rf_preds_log = np.load(PRED_DIR / "random_forest_log_preds.npy")
    lgbm_preds_log = np.load(PRED_DIR / "lightgbm_log_preds.npy")

    rf_preds = np.expm1(np.clip(rf_preds_log, 0, 14))
    lgbm_preds = np.expm1(np.clip(lgbm_preds_log, 0, 14))

    audit = worst_buildings_audit(val_df, y_true, rf_preds)
    per_use = per_use_lightgbm(cfg, val_df, train_df, feature_cols, y_true, lgbm_preds)


if __name__ == "__main__":
    main()
