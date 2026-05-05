from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class ARIMAModel(BaseModel):
    """Per-meter ARIMA. We fit one model per (building_id, meter) on the
    training segment and forecast the validation segment in chronological order.

    `fit_per_meter` expects the *raw* per-meter time series, not the engineered
    feature matrix used by the other models. The `fit` / `predict` interface is
    intentionally restricted; orchestration lives in `fit_predict_per_meter`.
    """

    name = "arima"
    family = "time_series"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._models: dict[tuple[int, int], Any] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ARIMAModel":
        raise NotImplementedError(
            "Use ARIMAModel.fit_predict_per_meter for per-meter ARIMA fitting."
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(
            "Use ARIMAModel.fit_predict_per_meter for per-meter ARIMA fitting."
        )

    def fit_predict_per_meter(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str = "meter_reading",
        meter_keys: tuple[str, str] = ("building_id", "meter"),
        timestamp_col: str = "timestamp",
        max_meters: int | None = None,
        log_target: bool = True,
        seasonal_period: int = 24,
        max_p: int = 3,
        max_q: int = 3,
        stepwise: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Fit auto_arima per meter, return val_df with a `pred` column.

        Falls back to a seasonal-naive forecast (lag-168) on meters where ARIMA
        fails to fit (singular matrix, too few points, etc).
        """
        try:
            from pmdarima.arima import auto_arima
        except ImportError as e:
            raise ImportError(
                "ARIMA requires pmdarima — install it with `pip install pmdarima`"
            ) from e

        keys = list(meter_keys)
        train_df = train_df.sort_values(keys + [timestamp_col])
        val_df = val_df.sort_values(keys + [timestamp_col]).copy()
        val_df["pred"] = np.nan

        groups = train_df.groupby(keys, observed=True)
        n_groups = groups.ngroups
        if max_meters is not None and n_groups > max_meters:
            sampled = list(groups.groups.keys())
            rng = np.random.default_rng(42)
            sampled = rng.choice(np.array(sampled, dtype=object), size=max_meters, replace=False)
            sampled_set = {tuple(s) for s in sampled}
        else:
            sampled_set = None

        n_fit = n_fallback = 0
        meter_count = 0
        n_to_fit = min(n_groups, max_meters) if max_meters is not None else n_groups
        for key, train_grp in groups:
            if sampled_set is not None and key not in sampled_set:
                continue
            meter_count += 1
            logger.info("ARIMA fitting meter %d/%d key=%s (train_len=%d)",
                        meter_count, n_to_fit, key, len(train_grp))
            val_grp = val_df.loc[
                (val_df["building_id"] == key[0]) & (val_df["meter"] == key[1])
            ]
            if len(val_grp) == 0:
                continue
            y_train = train_grp[target_col].astype(float).to_numpy()
            if log_target:
                y_train = np.log1p(np.clip(y_train, 0, None))

            forecast = None
            try:
                if len(y_train) < 2 * seasonal_period or np.all(y_train == y_train[0]):
                    raise ValueError("series too short or constant")
                model = auto_arima(
                    y_train,
                    seasonal=True,
                    m=seasonal_period,
                    max_p=max_p,
                    max_q=max_q,
                    stepwise=stepwise,
                    suppress_warnings=True,
                    error_action="ignore",
                )
                forecast = model.predict(n_periods=len(val_grp))
                self._models[key] = model
                n_fit += 1
            except Exception as e:
                if verbose:
                    logger.debug("ARIMA fit failed for %s: %s", key, e)
                last_week = train_grp.tail(seasonal_period * 7)[target_col].astype(float).to_numpy()
                if len(last_week) >= seasonal_period:
                    weekly = last_week[-seasonal_period:]
                    forecast = np.tile(weekly, int(np.ceil(len(val_grp) / seasonal_period)))[:len(val_grp)]
                    if log_target:
                        forecast = np.log1p(np.clip(forecast, 0, None))
                else:
                    fallback = float(np.mean(np.clip(y_train, 0, None))) if len(y_train) else 0.0
                    forecast = np.full(len(val_grp), fallback)
                n_fallback += 1

            if log_target:
                forecast = np.expm1(forecast)
            val_df.loc[val_grp.index, "pred"] = forecast

        logger.info("ARIMA: fit %d meters, fell back on %d", n_fit, n_fallback)
        return val_df
