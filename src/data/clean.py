from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KBTU_TO_KWH = 0.293071


def convert_site0_to_kwh(df: pd.DataFrame) -> pd.DataFrame:
    if "site_id" not in df.columns:
        return df
    mask = (df["site_id"] == 0) & (df["meter"] == 0)
    n = int(mask.sum())
    if n > 0:
        df.loc[mask, "meter_reading"] = (
            df.loc[mask, "meter_reading"] * KBTU_TO_KWH
        ).astype(np.float32)
        logger.info("Converted %d site-0 electricity rows from kBTU to kWh", n)
    return df


def drop_zero_streaks(
    df: pd.DataFrame,
    min_hours: int = 48,
    target_col: str = "meter_reading",
) -> pd.DataFrame:
    df = df.sort_values(["building_id", "meter", "timestamp"]).reset_index(drop=True)
    is_zero = (df[target_col] == 0).astype(np.int8)

    streaks = is_zero.groupby([df["building_id"], df["meter"]]).transform(
        lambda s: s.groupby((s != s.shift()).cumsum()).transform("sum")
    )

    bad_mask = (is_zero == 1) & (streaks >= min_hours)
    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        df = df.loc[~bad_mask].reset_index(drop=True)
        logger.info("Dropped %s rows in zero-streaks of %d+ hours", f"{n_bad:,}", min_hours)
    return df


def cap_outliers(
    df: pd.DataFrame,
    target_col: str = "meter_reading",
    quantile: float = 0.999,
) -> pd.DataFrame:
    caps = df.groupby(["building_id", "meter"])[target_col].transform("quantile", quantile)
    over = df[target_col] > caps
    n = int(over.sum())
    if n > 0:
        df.loc[over, target_col] = caps[over].astype(np.float32)
        logger.info("Capped %s readings at the %.3f per-meter quantile", f"{n:,}", quantile)
    return df


def drop_negative(df: pd.DataFrame, target_col: str = "meter_reading") -> pd.DataFrame:
    neg = df[target_col] < 0
    n = int(neg.sum())
    if n > 0:
        df = df.loc[~neg].reset_index(drop=True)
        logger.info("Dropped %s negative readings", f"{n:,}")
    return df


def impute_weather(weather: pd.DataFrame) -> pd.DataFrame:
    cont_cols = [
        "air_temperature", "dew_temperature", "wind_speed",
        "sea_level_pressure", "wind_direction",
    ]
    weather = weather.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    for col in cont_cols:
        if col in weather.columns:
            weather[col] = weather.groupby("site_id")[col].transform(
                lambda s: s.interpolate(method="linear", limit_direction="both")
            )

    if "cloud_coverage" in weather.columns:
        weather["cloud_coverage"] = weather["cloud_coverage"].fillna(
            weather["cloud_coverage"].median()
        )
    if "precip_depth_1_hr" in weather.columns:
        weather["precip_depth_1_hr"] = weather["precip_depth_1_hr"].fillna(0.0)
    return weather


def basic_clean(df: pd.DataFrame, outlier_quantile: float = 0.999,
                zero_streak_hours: int = 48) -> pd.DataFrame:
    df = drop_negative(df)
    df = convert_site0_to_kwh(df)
    df = drop_zero_streaks(df, min_hours=zero_streak_hours)
    df = cap_outliers(df, quantile=outlier_quantile)
    return df
