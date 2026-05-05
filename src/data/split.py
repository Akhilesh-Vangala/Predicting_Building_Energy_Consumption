from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def time_split(
    df: pd.DataFrame,
    train_months: Iterable[int],
    val_months: Iterable[int],
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_set = set(int(m) for m in train_months)
    val_set = set(int(m) for m in val_months)
    if train_set & val_set:
        raise ValueError("train_months and val_months must not overlap")

    months = df[timestamp_col].dt.month
    train = df.loc[months.isin(train_set)].reset_index(drop=True)
    val = df.loc[months.isin(val_set)].reset_index(drop=True)

    if len(train) == 0 or len(val) == 0:
        raise ValueError("time_split produced an empty fold; check month config")

    if train[timestamp_col].max() >= val[timestamp_col].min():
        raise AssertionError(
            "Temporal leakage: train.max() >= val.min(). "
            f"train.max={train[timestamp_col].max()} val.min={val[timestamp_col].min()}"
        )

    return train, val


def summarize_split(train: pd.DataFrame, val: pd.DataFrame, timestamp_col: str = "timestamp") -> dict:
    return {
        "train_rows": len(train),
        "val_rows": len(val),
        "train_min": str(train[timestamp_col].min()),
        "train_max": str(train[timestamp_col].max()),
        "val_min": str(val[timestamp_col].min()),
        "val_max": str(val[timestamp_col].max()),
        "train_buildings": int(train["building_id"].nunique()) if "building_id" in train.columns else None,
        "val_buildings": int(val["building_id"].nunique()) if "building_id" in val.columns else None,
    }
