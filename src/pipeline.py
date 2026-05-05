from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.data.clean import basic_clean, impute_weather
from src.data.load import load_subset, merge_with_context
from src.data.split import time_split, summarize_split
from src.features.pipeline import (
    FEATURE_COLUMNS, RAW_FEATURE_COLUMNS, build_features, feature_groups,
)
from src.features.target_encoding import TargetEncoder

logger = logging.getLogger(__name__)


@dataclass
class PreparedData:
    train_full: pd.DataFrame
    val_full: pd.DataFrame
    feature_cols: list[str]
    target_log_train: np.ndarray
    target_log_val: np.ndarray
    target_raw_train: np.ndarray
    target_raw_val: np.ndarray
    encoder: TargetEncoder | None
    summary: dict


def prepare_data(cfg: AppConfig, target_log: bool = True,
                 feature_set: str = "engineered") -> PreparedData:
    """Load → clean → merge → engineer features → time split.

    `feature_set` is `"engineered"` for the full 28-feature catalogue or
    `"raw"` for the 5 minimal columns we use in the ablation study.
    """
    readings, meta, weather = load_subset(cfg)
    weather = impute_weather(weather)

    logger.info("Loaded %s readings, %s meta rows, %s weather rows",
                f"{len(readings):,}", f"{len(meta):,}", f"{len(weather):,}")

    df = merge_with_context(readings, meta, weather)
    df = basic_clean(
        df,
        outlier_quantile=cfg.data.outlier_quantile,
        zero_streak_hours=cfg.data.zero_streak_min_hours,
    )
    logger.info("After clean: %s rows", f"{len(df):,}")

    df, encoder = build_features(df, cfg.features, encoder=None, is_train=True)
    train_df, val_df = time_split(df, cfg.split.train_months, cfg.split.val_months)

    encoder = TargetEncoder(
        keys=["primary_use", "hour"],
        target_col="meter_reading",
        smoothing=cfg.features.target_encoding_smoothing,
    ).fit(train_df.dropna(subset=["meter_reading", "primary_use"]))

    train_df = train_df.drop(columns=[c for c in train_df.columns if c.startswith("te_")], errors="ignore")
    val_df = val_df.drop(columns=[c for c in val_df.columns if c.startswith("te_")], errors="ignore")
    train_df = encoder.transform(train_df)
    val_df = encoder.transform(val_df)

    if feature_set == "engineered":
        feature_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    elif feature_set == "raw":
        feature_cols = [c for c in RAW_FEATURE_COLUMNS if c in train_df.columns]
    else:
        raise ValueError(f"unknown feature_set: {feature_set}")

    needed = feature_cols + ["meter_reading", "timestamp", "building_id", "meter",
                             "site_id", "primary_use"]
    needed = [c for c in needed if c in train_df.columns]
    n_before = len(train_df)
    train_df = train_df.dropna(subset=feature_cols + ["meter_reading"]).reset_index(drop=True)
    val_df = val_df.dropna(subset=feature_cols + ["meter_reading"]).reset_index(drop=True)
    logger.info("After dropna on features: train %s -> %s",
                f"{n_before:,}", f"{len(train_df):,}")

    y_train_raw = train_df["meter_reading"].to_numpy(dtype=np.float64)
    y_val_raw = val_df["meter_reading"].to_numpy(dtype=np.float64)
    y_train = np.log1p(np.clip(y_train_raw, 0, None)) if target_log else y_train_raw
    y_val = np.log1p(np.clip(y_val_raw, 0, None)) if target_log else y_val_raw

    summary = {
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "feature_cols": feature_cols,
        "feature_groups": feature_groups(),
        "split": summarize_split(train_df, val_df),
        "site_subset": cfg.data.site_subset if not cfg.data.use_full else "all",
        "feature_set": feature_set,
    }
    return PreparedData(
        train_full=train_df,
        val_full=val_df,
        feature_cols=feature_cols,
        target_log_train=y_train,
        target_log_val=y_val,
        target_raw_train=y_train_raw,
        target_raw_val=y_val_raw,
        encoder=encoder,
        summary=summary,
    )


def evaluate_predictions(
    val_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    target_log: bool = True,
) -> dict[str, Any]:
    from src.evaluation.metrics import score_predictions
    from src.evaluation.per_group import (
        per_building_type, per_meter_type, per_site,
    )

    if target_log:
        y_pred_real = np.expm1(np.clip(y_pred, a_min=0.0, a_max=14.0))
        y_true_real = np.expm1(np.clip(y_true, a_min=0.0, a_max=14.0))
    else:
        y_pred_real = np.clip(y_pred, a_min=0.0, a_max=None)
        y_true_real = np.clip(y_true, a_min=0.0, a_max=None)

    base = score_predictions(y_true_real, y_pred_real)
    eval_df = val_df.copy()
    eval_df["y_true"] = y_true_real
    eval_df["y_pred"] = y_pred_real

    by_use = per_building_type(eval_df).to_dict(orient="records")
    by_meter = per_meter_type(eval_df).to_dict(orient="records")
    by_site_ = per_site(eval_df).to_dict(orient="records")

    return {
        "overall": base,
        "by_primary_use": by_use,
        "by_meter": by_meter,
        "by_site": by_site_,
    }


def baseline_meter_mean(prep: PreparedData) -> dict[str, Any]:
    from src.evaluation.metrics import score_predictions

    train = prep.train_full.copy()
    val = prep.val_full.copy()
    means = train.groupby(["building_id", "meter"], observed=True)["meter_reading"].mean()
    val_means = val.set_index(["building_id", "meter"]).index.map(means).to_numpy()
    val_means = np.where(pd.isna(val_means), float(train["meter_reading"].mean()), val_means)
    return {
        "name": "baseline_meter_mean",
        "metrics": score_predictions(prep.target_raw_val, val_means.astype(np.float64)),
        "predictions": val_means,
    }


def baseline_lag_24h(prep: PreparedData) -> dict[str, Any]:
    from src.evaluation.metrics import score_predictions

    val = prep.val_full.copy()
    if "lag_24h" not in val.columns:
        return {"name": "baseline_lag_24h", "metrics": {}, "predictions": None}
    pred = val["lag_24h"].fillna(val["meter_reading"].median()).to_numpy(dtype=np.float64)
    return {
        "name": "baseline_lag_24h",
        "metrics": score_predictions(prep.target_raw_val, pred),
        "predictions": pred,
    }
