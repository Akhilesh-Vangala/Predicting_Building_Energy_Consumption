from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from src.config import FeaturesConfig
from src.features.building_features import add_building_features
from src.features.interactions import add_interaction_features
from src.features.lag_features import add_lag_features, add_rolling_features
from src.features.target_encoding import TargetEncoder
from src.features.time_features import add_time_features
from src.features.weather_features import add_weather_features

logger = logging.getLogger(__name__)


_TIME = ["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
         "month_sin", "month_cos", "is_weekend", "is_holiday"]
_LAG = ["lag_24h", "lag_168h", "lag_diff_24h"]
_ROLL = ["rolling_mean_24h", "rolling_std_24h",
         "rolling_mean_168h", "rolling_std_168h"]
_WEATHER = ["air_temperature", "dew_temperature", "temp_squared",
            "wind_speed", "cloud_coverage", "precip_depth_1_hr"]
_BUILDING = ["log_square_feet", "building_age", "primary_use_code"]
_INTER = ["hour_x_weekend", "sqft_x_temp"]
_TE = ["te_primary_use_x_hour"]

FEATURE_COLUMNS: list[str] = (
    _TIME + _LAG + _ROLL + _WEATHER + _BUILDING + _INTER + _TE
)

RAW_FEATURE_COLUMNS: list[str] = [
    "hour",
    "air_temperature",
    "log_square_feet",
    "primary_use_code",
    "meter",
]


def feature_groups() -> dict[str, list[str]]:
    return {
        "time": list(_TIME),
        "lag": list(_LAG),
        "rolling": list(_ROLL),
        "weather": list(_WEATHER),
        "building": list(_BUILDING),
        "interaction": list(_INTER),
        "target_encoding": list(_TE),
    }


def build_features(
    df: pd.DataFrame,
    cfg: FeaturesConfig,
    encoder: TargetEncoder | None = None,
    is_train: bool = True,
) -> tuple[pd.DataFrame, TargetEncoder]:
    """Apply the full feature pipeline.

    On training data, pass `is_train=True` and `encoder=None`; we fit a target
    encoder. On validation we pass the trained encoder so no validation target
    leaks into features.
    """
    logger.info("Building features for %s rows (is_train=%s)", f"{len(df):,}", is_train)

    df = add_time_features(df)
    df = add_lag_features(df, lag_hours=cfg.lag_hours)
    df = add_rolling_features(df, windows=cfg.rolling_windows)
    df = add_building_features(df)
    df = add_weather_features(df)
    df = add_interaction_features(df)

    if cfg.use_target_encoding:
        if encoder is None:
            if not is_train:
                raise ValueError("encoder must be provided for non-training data")
            encoder = TargetEncoder(
                keys=["primary_use", "hour"],
                target_col="meter_reading",
                smoothing=cfg.target_encoding_smoothing,
            )
            encoder.fit(df.dropna(subset=["meter_reading", "primary_use"]))
        df = encoder.transform(df)

    return df, encoder


def assert_no_future_leakage(train: pd.DataFrame, val: pd.DataFrame,
                             timestamp_col: str = "timestamp") -> None:
    if train[timestamp_col].max() >= val[timestamp_col].min():
        raise AssertionError(
            "Temporal leakage: train.max >= val.min — features must not see future."
        )
