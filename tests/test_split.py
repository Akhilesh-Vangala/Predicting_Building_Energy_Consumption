from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.split import time_split, summarize_split


def _toy(n_per_month: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for m in range(1, 13):
        ts = pd.date_range(f"2016-{m:02d}-01", periods=n_per_month, freq="h")
        for b in range(3):
            rows.extend(
                {"building_id": b, "meter": 0, "timestamp": t,
                 "meter_reading": float(rng.uniform(0, 100))}
                for t in ts
            )
    return pd.DataFrame(rows)


def test_time_split_no_leakage_default_months() -> None:
    df = _toy()
    train, val = time_split(df, train_months=range(1, 10), val_months=[10, 11, 12])
    assert train["timestamp"].max() < val["timestamp"].min()


def test_time_split_rejects_overlap() -> None:
    df = _toy()
    with pytest.raises(ValueError):
        time_split(df, train_months=[1, 2, 3], val_months=[3, 4])


def test_time_split_rejects_empty_val() -> None:
    df = _toy()
    with pytest.raises(ValueError):
        time_split(df, train_months=list(range(1, 13)), val_months=[])


def test_summary_keys_present() -> None:
    df = _toy()
    train, val = time_split(df, train_months=range(1, 10), val_months=[10, 11, 12])
    summary = summarize_split(train, val)
    assert {"train_rows", "val_rows", "train_min", "train_max",
            "val_min", "val_max"} <= summary.keys()
