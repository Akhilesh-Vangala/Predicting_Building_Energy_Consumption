from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    lag_hours: Sequence[int] = (24, 168),
    target_col: str = "meter_reading",
    group_keys: Sequence[str] = ("building_id", "meter"),
) -> pd.DataFrame:
    df = df.sort_values(list(group_keys) + ["timestamp"]).reset_index(drop=True)

    grouped = df.groupby(list(group_keys), observed=True)[target_col]
    for lag in lag_hours:
        df[f"lag_{lag}h"] = grouped.shift(lag).astype(np.float32)

    if 24 in lag_hours and 168 in lag_hours:
        df["lag_diff_24h"] = (df["lag_24h"] - df["lag_168h"]).astype(np.float32)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: Sequence[int] = (24, 168),
    target_col: str = "meter_reading",
    group_keys: Sequence[str] = ("building_id", "meter"),
) -> pd.DataFrame:
    df = df.sort_values(list(group_keys) + ["timestamp"]).reset_index(drop=True)
    grouped = df.groupby(list(group_keys), observed=True)[target_col]

    for w in windows:
        min_periods = max(1, w // 4)
        df[f"rolling_mean_{w}h"] = grouped.transform(
            lambda s: s.shift(1).rolling(window=w, min_periods=min_periods).mean()
        ).astype(np.float32)
        df[f"rolling_std_{w}h"] = grouped.transform(
            lambda s: s.shift(1).rolling(window=w, min_periods=min_periods).std()
        ).astype(np.float32)

    return df
