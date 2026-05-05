from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def build_consumption_profiles(
    df: pd.DataFrame,
    target_col: str = "meter_reading",
    timestamp_col: str = "timestamp",
    meter_keys: tuple[str, ...] = ("building_id", "meter"),
) -> pd.DataFrame:
    """Per-meter normalised hour-of-day consumption profile.

    Returns a DataFrame indexed by `meter_keys` with 24 columns h00..h23
    representing each meter's average consumption at each hour, normalised
    so the row sums to 1.
    """
    work = df[[*meter_keys, timestamp_col, target_col]].copy()
    work["hour"] = pd.DatetimeIndex(work[timestamp_col]).hour
    profile = (
        work.groupby([*meter_keys, "hour"], observed=True)[target_col]
            .mean()
            .unstack("hour")
            .reindex(columns=range(24))
            .fillna(0.0)
    )
    sums = profile.sum(axis=1)
    nonzero = sums > 0
    profile.loc[nonzero] = profile.loc[nonzero].div(sums[nonzero], axis=0)
    profile.columns = [f"h{h:02d}" for h in profile.columns]
    return profile


def fit_kmeans(profiles: pd.DataFrame, k: int, random_state: int = 42) -> KMeans:
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    km.fit(profiles.to_numpy(dtype=np.float32))
    return km


def elbow_curve(profiles: pd.DataFrame, k_range: Iterable[int],
                random_state: int = 42) -> pd.DataFrame:
    rows: list[dict] = []
    for k in k_range:
        km = fit_kmeans(profiles, k, random_state=random_state)
        rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "n_iter": int(km.n_iter_),
        })
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def assign_clusters(km: KMeans, profiles: pd.DataFrame) -> pd.Series:
    labels = km.predict(profiles.to_numpy(dtype=np.float32))
    return pd.Series(labels, index=profiles.index, name="cluster")
