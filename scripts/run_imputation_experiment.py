"""
Imputation experiment: compare 4 strategies for zero-streak meter readings.

The pipeline is identical for all strategies except how consecutive-zero
readings (48 h+ streaks) are handled. LightGBM is used as the fixed model
so the effect of imputation is isolated.

Strategies
----------
drop        drop zero-streak rows entirely  [current default]
ffill       forward-fill within each (building, meter) group
meter_mean  replace zeros with the per-meter mean of non-streak readings
keep_zeros  keep zeros as-is (no streak removal)

Outputs:
  results/metrics/imputation_experiment.csv
  results/figures/imputation_experiment.{pdf,png}
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.utils import setup_logging

logger = logging.getLogger(__name__)

STRATEGIES = ["drop", "ffill", "meter_mean", "keep_zeros"]

STRATEGY_LABELS = {
    "drop":       "Drop streaks (current)",
    "ffill":      "Forward fill",
    "meter_mean": "Meter mean fill",
    "keep_zeros": "Keep zeros",
}


def _get_streak_mask(df: pd.DataFrame, min_hours: int,
                     target_col: str = "meter_reading") -> pd.Series:
    """Return boolean mask of rows that are zero AND part of a streak >= min_hours."""
    is_zero = (df[target_col] == 0).astype(np.int8)
    streak_len = is_zero.groupby([df["building_id"], df["meter"]]).transform(
        lambda s: s.groupby((s != s.shift()).cumsum()).transform("sum")
    )
    return (is_zero == 1) & (streak_len >= min_hours)


def _apply_strategy(df: pd.DataFrame, strategy: str, min_hours: int,
                    target_col: str = "meter_reading") -> pd.DataFrame:
    """
    Apply common cleaning steps (negative drop, unit conversion, outlier cap)
    then handle zero-streaks according to `strategy`.
    """
    from src.data.clean import cap_outliers, convert_site0_to_kwh, drop_negative, drop_zero_streaks

    df = drop_negative(df)
    df = convert_site0_to_kwh(df)

    df = df.sort_values(["building_id", "meter", "timestamp"]).reset_index(drop=True)

    if strategy == "drop":
        df = drop_zero_streaks(df, min_hours=min_hours, target_col=target_col)

    elif strategy == "ffill":
        bad = _get_streak_mask(df, min_hours, target_col)
        df.loc[bad, target_col] = np.nan
        df[target_col] = (
            df.groupby(["building_id", "meter"])[target_col]
            .transform(lambda s: s.ffill().bfill())
        )

    elif strategy == "meter_mean":
        bad = _get_streak_mask(df, min_hours, target_col)
        means = (
            df.loc[~bad]
            .groupby(["building_id", "meter"])[target_col]
            .mean()
            .rename("_fill_mean")
        )
        df = df.join(means, on=["building_id", "meter"])
        df.loc[bad, target_col] = df.loc[bad, "_fill_mean"].fillna(0.0)
        df = df.drop(columns=["_fill_mean"])

    elif strategy == "keep_zeros":
        pass  # no streak removal

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    df = cap_outliers(df, quantile=0.999, target_col=target_col)
    return df


def _run_strategy(strategy: str, merged_df: pd.DataFrame, cfg) -> dict:
    from src.data.split import time_split
    from src.features.pipeline import FEATURE_COLUMNS, build_features
    from src.features.target_encoding import TargetEncoder
    from src.models import LightGBMModel
    from src.pipeline import evaluate_predictions

    import warnings
    warnings.filterwarnings("ignore")

    t0 = time.perf_counter()
    df = _apply_strategy(
        merged_df.copy(),
        strategy,
        min_hours=cfg.data.zero_streak_min_hours,
    )
    logger.info("[%s] rows after cleaning: %d", strategy, len(df))

    df, _ = build_features(df, cfg.features, encoder=None, is_train=True)
    train_df, val_df = time_split(df, cfg.split.train_months, cfg.split.val_months)

    encoder = TargetEncoder(
        keys=["primary_use", "hour"],
        target_col="meter_reading",
        smoothing=cfg.features.target_encoding_smoothing,
    ).fit(train_df.dropna(subset=["meter_reading", "primary_use"]))

    for frame in (train_df, val_df):
        te_cols = [c for c in frame.columns if c.startswith("te_")]
        frame.drop(columns=te_cols, inplace=True, errors="ignore")

    train_df = encoder.transform(train_df)
    val_df = encoder.transform(val_df)

    feature_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    train_df = train_df.dropna(subset=feature_cols + ["meter_reading"]).reset_index(drop=True)
    val_df = val_df.dropna(subset=feature_cols + ["meter_reading"]).reset_index(drop=True)

    X_train = train_df[feature_cols]
    y_train = np.log1p(np.clip(train_df["meter_reading"].to_numpy(), 0, None))
    X_val = val_df[feature_cols]
    y_val = np.log1p(np.clip(val_df["meter_reading"].to_numpy(), 0, None))

    lgb = LightGBMModel(**cfg.models.get("lightgbm", {}))
    lgb.fit(X_train, y_train, eval_set=(X_val, y_val))
    preds = lgb.predict(X_val)

    ev = evaluate_predictions(val_df, preds, y_val, target_log=True)
    elapsed = time.perf_counter() - t0

    logger.info("[%s] RMSE=%.2f  MAE=%.2f  CV-RMSE=%.3f  in %.1fs",
                strategy, ev["overall"]["rmse"], ev["overall"]["mae"],
                ev["overall"]["cv_rmse"], elapsed)
    return {
        "strategy": strategy,
        "label": STRATEGY_LABELS[strategy],
        "n_train": len(train_df),
        "n_val": len(val_df),
        "train_seconds": round(elapsed, 1),
        **ev["overall"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES,
                        choices=STRATEGIES)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    # Load raw data once — all strategies share the same merge
    from src.data.clean import impute_weather
    from src.data.load import load_subset, merge_with_context

    logger.info("Loading raw data...")
    readings, meta, weather = load_subset(cfg)
    weather = impute_weather(weather)
    merged_df = merge_with_context(readings, meta, weather)
    logger.info("Merged df: %d rows", len(merged_df))

    results = []
    for strategy in args.strategies:
        logger.info("=== strategy: %s ===", strategy)
        row = _run_strategy(strategy, merged_df, cfg)
        results.append(row)

    df = pd.DataFrame(results)
    out_csv = cfg.paths.metrics / "imputation_experiment.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Saved %s", out_csv)

    _plot(df, cfg.paths.plots)


def _plot(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from src.viz.style import ACCENT, PRIMARY, SUPPORT, save_figure, set_style

    set_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    labels = df["label"].tolist()
    rmses = df["rmse"].tolist()
    baseline_rmse = df.loc[df["strategy"] == "drop", "rmse"].values
    baseline = float(baseline_rmse[0]) if len(baseline_rmse) else min(rmses)

    colors = [ACCENT if r < baseline - 1 else (SUPPORT[0] if r == baseline else PRIMARY)
              for r in rmses]
    bars = ax.barh(labels, rmses, color=colors)

    for bar, val in zip(bars, rmses):
        ax.text(val * 1.005, bar.get_y() + bar.get_height() / 2,
                f"{val:,.1f}", va="center", fontsize=9)

    ax.axvline(baseline, color="#555", linewidth=0.9, linestyle="--", label="current (drop)")
    ax.set_xlabel("Validation RMSE (kWh)")
    ax.set_title("Effect of zero-streak imputation strategy on LightGBM")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_dir / "imputation_experiment", formats=("pdf", "png"))
    plt.close(fig)
    logger.info("imputation_experiment figure saved to %s", out_dir)


if __name__ == "__main__":
    main()
