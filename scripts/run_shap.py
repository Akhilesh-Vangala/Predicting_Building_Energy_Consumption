"""Generate SHAP analysis plots for the final paper.

Produces three publication-quality figures:
  1. shap_beeswarm.pdf  — global summary (feature direction + magnitude)
  2. shap_waterfall_electricity.pdf — single easy electricity reading
  3. shap_waterfall_steam.pdf       — single hard steam reading
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.models.boosting import LightGBMModel
from src.pipeline import prepare_data
from src.viz.style import set_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

LATEX_FIGURES = Path("/Users/akhileshvangala/Desktop/ml latex/figures")
PLOT_DIR = ROOT / "results" / "plots" / "analysis"
LATEX_FIGURES.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_LABELS = {
    "lag_24h": "Lag 24h",
    "rolling_mean_24h": "Rolling mean 24h",
    "lag_168h": "Lag 168h",
    "hour_sin": "Hour (sin)",
    "hour_cos": "Hour (cos)",
    "te_primary_use_x_hour": "Target encoding",
    "air_temperature": "Air temperature",
    "rolling_std_168h": "Rolling std 168h",
    "rolling_mean_168h": "Rolling mean 168h",
    "rolling_std_24h": "Rolling std 24h",
    "log_square_feet": "Log sq.ft.",
    "dayofweek_sin": "Day-of-week (sin)",
    "dayofweek_cos": "Day-of-week (cos)",
    "dew_temperature": "Dew temperature",
    "temp_squared": "Temp²",
    "lag_diff_24h": "Lag diff 24h",
    "month_sin": "Month (sin)",
    "month_cos": "Month (cos)",
    "sqft_x_temp": "Sqft × temp",
    "hour_x_weekend": "Hour × weekend",
    "is_weekend": "Weekend",
    "is_holiday": "Holiday",
    "wind_speed": "Wind speed",
    "cloud_coverage": "Cloud coverage",
    "building_age": "Building age",
    "primary_use_code": "Primary-use code",
    "precip_depth_1_hr": "Precipitation",
}


def save(fig: plt.Figure, stem: str) -> None:
    for dest in [PLOT_DIR, LATEX_FIGURES]:
        fig.savefig(dest / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(dest / f"{stem}.png", bbox_inches="tight", dpi=200)
    log.info("Saved %s", stem)


def main() -> None:
    set_style()
    cfg = load_config(ROOT / "configs" / "default.yaml")

    log.info("Preparing data ...")
    prep = prepare_data(cfg, feature_set="engineered")

    X_train = prep.train_full[prep.feature_cols].astype(np.float32)
    y_train = prep.target_log_train
    X_val   = prep.val_full[prep.feature_cols].astype(np.float32)
    y_val   = prep.target_log_val

    log.info("Training LightGBM ...")
    model_params = dict(
        num_leaves=63, learning_rate=0.05, n_estimators=1500,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        min_data_in_leaf=50, early_stopping_rounds=50, random_state=42,
    )
    lgbm = LightGBMModel(params=model_params)
    lgbm.fit(X_train, y_train, eval_set=(X_val, y_val))
    booster = lgbm.model_

    # --- SHAP on a representative 5,000-row sample from val ---
    rng = np.random.default_rng(42)
    val_full = prep.val_full.copy()
    val_full["_y_true_raw"] = prep.target_raw_val
    val_full["_y_pred_log"] = np.expm1(
        np.clip(booster.predict(X_val, num_iteration=booster.best_iteration), 0, 14)
    )
    val_full["_abs_err"] = np.abs(val_full["_y_true_raw"] - val_full["_y_pred_log"])

    sample_idx = rng.choice(len(val_full), size=5000, replace=False)
    X_sample = X_val.iloc[sample_idx]

    log.info("Computing SHAP values on 5,000-row sample ...")
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)

    # Rename columns for display
    display_names = [FEATURE_LABELS.get(c, c) for c in prep.feature_cols]

    # ------------------------------------------------------------------ #
    # Figure 1: Global beeswarm
    # ------------------------------------------------------------------ #
    log.info("Plotting global beeswarm ...")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=display_names,
        max_display=15,
        show=False,
        plot_size=None,
        color_bar_label="Feature value",
    )
    ax = plt.gca()
    ax.set_xlabel("SHAP value (impact on log-scale prediction)", fontsize=9)
    ax.set_title("SHAP Feature Importance (LightGBM)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(plt.gcf(), "shap_beeswarm")
    plt.close("all")

    # ------------------------------------------------------------------ #
    # Figure 2: Waterfall — easy electricity meter (low error)
    # ------------------------------------------------------------------ #
    elec_mask = (val_full["meter"] == 0) & (val_full["_y_true_raw"] > 20) & \
                (val_full["_y_true_raw"] < 300) & (val_full["_abs_err"] < 5)
    elec_cands = val_full[elec_mask]
    if len(elec_cands) == 0:
        elec_cands = val_full[val_full["meter"] == 0].nsmallest(50, "_abs_err")

    elec_row_global = elec_cands.index[0]
    elec_row_local  = list(val_full.index).index(elec_row_global)

    # Re-compute SHAP for this single row (need from original X_val)
    elec_X = X_val.iloc[[elec_row_local]]
    elec_sv = explainer(elec_X)
    elec_sv.feature_names = display_names

    actual_kwh  = float(val_full["_y_true_raw"].iloc[elec_row_local])
    pred_kwh    = float(val_full["_y_pred_log"].iloc[elec_row_local])
    prim_use    = val_full["primary_use"].iloc[elec_row_local]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    shap.plots.waterfall(elec_sv[0], max_display=12, show=False)
    plt.title(
        f"Electricity meter — {prim_use}\n"
        f"Actual {actual_kwh:.1f} kWh  |  Predicted {pred_kwh:.1f} kWh  "
        f"|  Error {abs(actual_kwh - pred_kwh):.1f} kWh",
        fontsize=9, fontweight="bold",
    )
    plt.tight_layout()
    save(plt.gcf(), "shap_waterfall_electricity")
    plt.close("all")

    # ------------------------------------------------------------------ #
    # Figure 3: Waterfall — hard steam meter (large error)
    # ------------------------------------------------------------------ #
    steam_mask = (val_full["meter"] == 2) & (val_full["_y_true_raw"] > 500) & \
                 (val_full["_abs_err"] > 1000)
    steam_cands = val_full[steam_mask]
    if len(steam_cands) == 0:
        steam_cands = val_full[val_full["meter"] == 2].nlargest(50, "_abs_err")

    steam_row_global = steam_cands.index[0]
    steam_row_local  = list(val_full.index).index(steam_row_global)

    steam_X = X_val.iloc[[steam_row_local]]
    steam_sv = explainer(steam_X)
    steam_sv.feature_names = display_names

    actual_kwh_s = float(val_full["_y_true_raw"].iloc[steam_row_local])
    pred_kwh_s   = float(val_full["_y_pred_log"].iloc[steam_row_local])
    prim_use_s   = val_full["primary_use"].iloc[steam_row_local]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    shap.plots.waterfall(steam_sv[0], max_display=12, show=False)
    plt.title(
        f"Steam meter — {prim_use_s}\n"
        f"Actual {actual_kwh_s:,.0f} kWh  |  Predicted {pred_kwh_s:,.0f} kWh  "
        f"|  Error {abs(actual_kwh_s - pred_kwh_s):,.0f} kWh",
        fontsize=9, fontweight="bold",
    )
    plt.tight_layout()
    save(plt.gcf(), "shap_waterfall_steam")
    plt.close("all")

    log.info("All SHAP figures saved.")


if __name__ == "__main__":
    main()
