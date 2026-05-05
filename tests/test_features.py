from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.lag_features import add_lag_features, add_rolling_features
from src.features.time_features import add_time_features
from src.features.target_encoding import TargetEncoder


def _toy(n_hours: int = 240) -> pd.DataFrame:
    ts = pd.date_range("2016-01-01", periods=n_hours, freq="h")
    rng = np.random.default_rng(0)
    rows = []
    for b in range(2):
        for m in [0, 1]:
            for t in ts:
                rows.append({
                    "building_id": b,
                    "meter": m,
                    "site_id": 1,
                    "primary_use": "Education",
                    "timestamp": t,
                    "meter_reading": float(rng.uniform(0, 50)),
                })
    return pd.DataFrame(rows)


def test_time_features_cyclical_bounds() -> None:
    df = add_time_features(_toy())
    for col in ["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
                "month_sin", "month_cos"]:
        assert df[col].min() >= -1.0 - 1e-6
        assert df[col].max() <= 1.0 + 1e-6
    assert df["is_weekend"].isin([0, 1]).all()
    assert df["is_holiday"].isin([0, 1]).all()


def test_lag_uses_only_past() -> None:
    df = _toy()
    df = add_lag_features(df, lag_hours=(24, 168))
    grp = df.sort_values(["building_id", "meter", "timestamp"]).groupby(
        ["building_id", "meter"], observed=True
    )
    for _, g in grp:
        for offset, col in [(24, "lag_24h"), (168, "lag_168h")]:
            actual = g[col].to_numpy()
            expected = g["meter_reading"].shift(offset).to_numpy()
            np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-6)


def test_rolling_excludes_current_row() -> None:
    df = _toy(n_hours=240)
    df = add_rolling_features(df, windows=(24,))
    g = df.sort_values(["building_id", "meter", "timestamp"]).groupby(
        ["building_id", "meter"], observed=True
    )
    for _, sub in g:
        s = sub["meter_reading"]
        manual = s.shift(1).rolling(window=24, min_periods=6).mean().to_numpy()
        np.testing.assert_allclose(sub["rolling_mean_24h"].to_numpy(),
                                   manual, equal_nan=True, rtol=1e-5)


def test_target_encoder_no_leakage() -> None:
    df = _toy()
    df = add_time_features(df)
    train = df[df["timestamp"] < "2016-01-08"]
    val = df[df["timestamp"] >= "2016-01-08"]
    enc = TargetEncoder(keys=["primary_use", "hour"], smoothing=10)
    enc.fit(train.dropna(subset=["meter_reading"]))
    encoded_val = enc.transform(val)
    assert "te_primary_use_x_hour" in encoded_val.columns
    assert encoded_val["te_primary_use_x_hour"].isna().sum() == 0


def test_target_encoder_global_mean_for_missing_keys() -> None:
    train = pd.DataFrame({
        "primary_use": ["Education"] * 24,
        "hour": list(range(24)),
        "meter_reading": [10.0] * 24,
    })
    val = pd.DataFrame({
        "primary_use": ["Healthcare"],
        "hour": [10],
        "meter_reading": [50.0],
    })
    enc = TargetEncoder(keys=["primary_use", "hour"], smoothing=5)
    enc.fit(train)
    out = enc.transform(val)
    np.testing.assert_allclose(out["te_primary_use_x_hour"].iloc[0],
                               np.log1p(10.0), rtol=1e-5)
