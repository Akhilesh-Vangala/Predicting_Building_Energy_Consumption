from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TargetEncoder:
    """Smoothed target mean encoder fit on training data only.

    For each combination of `keys`, replaces the category with
        (n * mean + smoothing * global_mean) / (n + smoothing)
    so rare combinations regress toward the global mean and there is no leakage
    from the validation set.
    """

    def __init__(self, keys: Sequence[str], target_col: str = "meter_reading",
                 smoothing: int = 30):
        self.keys = list(keys)
        self.target_col = target_col
        self.smoothing = smoothing
        self.global_mean_: float | None = None
        self.encoding_: pd.DataFrame | None = None
        self.col_name_: str = "te_" + "_x_".join(self.keys)

    def fit(self, df: pd.DataFrame) -> "TargetEncoder":
        if self.target_col not in df.columns:
            raise ValueError(f"target column {self.target_col} missing")

        target = np.log1p(df[self.target_col].clip(lower=0))
        self.global_mean_ = float(target.mean())

        agg = (
            df.assign(_target=target)
              .groupby(self.keys, observed=True)["_target"]
              .agg(["mean", "count"])
              .reset_index()
        )
        smoothed = (
            (agg["count"] * agg["mean"] + self.smoothing * self.global_mean_)
            / (agg["count"] + self.smoothing)
        )
        self.encoding_ = agg[self.keys].copy()
        self.encoding_[self.col_name_] = smoothed.astype(np.float32)
        logger.info("Fit target encoder on %s -> %d categories",
                    self.keys, len(self.encoding_))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.encoding_ is None or self.global_mean_ is None:
            raise RuntimeError("TargetEncoder must be fit before transform()")
        out = df.merge(self.encoding_, on=self.keys, how="left")
        out[self.col_name_] = out[self.col_name_].fillna(self.global_mean_).astype(np.float32)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
