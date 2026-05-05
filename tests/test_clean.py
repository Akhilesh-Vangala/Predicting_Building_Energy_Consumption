from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.clean import (
    cap_outliers, convert_site0_to_kwh, drop_negative, drop_zero_streaks,
)


def _hourly(n: int, vals: list[float], building: int = 0, meter: int = 0,
            site: int = 0, start: str = "2016-01-01") -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame({
        "building_id": building, "meter": meter, "site_id": site,
        "timestamp": ts, "meter_reading": vals,
    })


def test_drop_negative_removes_negatives() -> None:
    df = _hourly(4, [1.0, -2.0, 3.0, -1.0])
    out = drop_negative(df)
    assert len(out) == 2
    assert (out["meter_reading"] >= 0).all()


def test_convert_site0_kbtu_to_kwh() -> None:
    df = _hourly(2, [100.0, 200.0], site=0, meter=0)
    out = convert_site0_to_kwh(df.copy())
    np.testing.assert_allclose(out["meter_reading"].to_numpy(),
                               np.array([100.0, 200.0]) * 0.293071, rtol=1e-5)


def test_convert_site0_skips_other_sites_and_meters() -> None:
    df = pd.concat([
        _hourly(2, [100.0, 200.0], site=0, meter=0),
        _hourly(2, [100.0, 200.0], site=1, meter=0),
        _hourly(2, [100.0, 200.0], site=0, meter=1),
    ], ignore_index=True)
    out = convert_site0_to_kwh(df.copy())
    converted = (df["site_id"] == 0) & (df["meter"] == 0)
    np.testing.assert_allclose(
        out.loc[converted, "meter_reading"].to_numpy(),
        df.loc[converted, "meter_reading"].to_numpy() * 0.293071, rtol=1e-5,
    )
    np.testing.assert_allclose(
        out.loc[~converted, "meter_reading"].to_numpy(),
        df.loc[~converted, "meter_reading"].to_numpy(),
    )


def test_drop_zero_streaks_drops_long_runs() -> None:
    vals = [10.0] * 5 + [0.0] * 50 + [10.0] * 5
    df = _hourly(60, vals)
    out = drop_zero_streaks(df, min_hours=48)
    assert (out["meter_reading"] > 0).all() or len(out) == 10


def test_drop_zero_streaks_keeps_short_runs() -> None:
    vals = [10.0] * 5 + [0.0] * 10 + [10.0] * 5
    df = _hourly(20, vals)
    out = drop_zero_streaks(df, min_hours=48)
    assert len(out) == 20


def test_cap_outliers_caps_values() -> None:
    vals = list(np.linspace(1, 100, 999)) + [10_000.0]
    df = _hourly(1000, vals)
    out = cap_outliers(df.copy(), quantile=0.999)
    assert out["meter_reading"].max() < 10_000.0
