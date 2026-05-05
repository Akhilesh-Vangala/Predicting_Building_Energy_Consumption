from __future__ import annotations

from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

SITE_COUNTRY = {
    0: "US", 1: "UK", 2: "US", 3: "US", 4: "US", 5: "UK",
    6: "US", 7: "CA", 8: "US", 9: "US", 10: "US", 11: "CA",
    12: "IE", 13: "US", 14: "US", 15: "US",
}


@lru_cache(maxsize=64)
def _country_holiday_set(country: str, year: int) -> frozenset:
    cal = holidays.country_holidays(country, years=[year])
    return frozenset(d.isoformat() for d in cal.keys())


def _per_site_holiday_flag(df: pd.DataFrame, ts: pd.DatetimeIndex) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.int8)
    if "site_id" not in df.columns:
        years = sorted({int(y) for y in ts.year.unique()})
        us_set: set = set()
        for y in years:
            us_set |= _country_holiday_set("US", y)
        date_strs = ts.strftime("%Y-%m-%d").to_numpy()
        out[:] = np.fromiter((d in us_set for d in date_strs), dtype=np.int8, count=len(date_strs))
        return out

    date_strs = ts.strftime("%Y-%m-%d").to_numpy()
    site_ids = df["site_id"].to_numpy()
    countries = np.array([SITE_COUNTRY.get(int(s), "US") for s in site_ids])
    years = sorted({int(y) for y in ts.year.unique()})
    for country in np.unique(countries):
        country_set: set = set()
        for y in years:
            country_set |= _country_holiday_set(country, y)
        mask = countries == country
        sub = date_strs[mask]
        out[mask] = np.fromiter((d in country_set for d in sub), dtype=np.int8, count=len(sub))
    return out


def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    ts = pd.DatetimeIndex(df[ts_col])

    df["hour"] = ts.hour.astype(np.int8)
    df["dayofweek"] = ts.dayofweek.astype(np.int8)
    df["month"] = ts.month.astype(np.int8)
    df["is_weekend"] = (ts.dayofweek >= 5).astype(np.int8)
    df["is_holiday"] = _per_site_holiday_flag(df, ts)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)

    return df
