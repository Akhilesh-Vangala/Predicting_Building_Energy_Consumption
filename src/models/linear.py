from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    ElasticNetCV, LassoCV, LinearRegression, RidgeCV,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel


class _LinearScaledMixin(BaseModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.scaler_: StandardScaler | None = None
        self.kept_idx_: np.ndarray | None = None

    def _fit_scaler(self, X: pd.DataFrame) -> np.ndarray:
        self.feature_names_ = X.columns.tolist()
        Xn = X.to_numpy(dtype=np.float64)
        stds = Xn.std(axis=0)
        self.kept_idx_ = np.where(stds > 1e-8)[0]
        Xk = Xn[:, self.kept_idx_]
        self.scaler_ = StandardScaler()
        return self.scaler_.fit_transform(Xk)

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.scaler_ is None or self.kept_idx_ is None:
            raise RuntimeError("model must be fit before predict")
        Xn = X.to_numpy(dtype=np.float64)
        return self.scaler_.transform(Xn[:, self.kept_idx_])

    def feature_importance(self) -> dict[str, float] | None:
        if self.model_ is None or self.feature_names_ is None or self.kept_idx_ is None:
            return None
        coefs = getattr(self.model_, "coef_", None)
        if coefs is None:
            return None
        kept_names = [self.feature_names_[i] for i in self.kept_idx_]
        return {n: float(c) for n, c in zip(kept_names, np.ravel(coefs))}


class OLSModel(_LinearScaledMixin):
    name = "ols"
    family = "linear"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "OLSModel":
        Xs = self._fit_scaler(X)
        self.model_ = LinearRegression()
        self.model_.fit(Xs, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._transform(X))


class RidgeModel(_LinearScaledMixin):
    name = "ridge"
    family = "linear"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RidgeModel":
        Xs = self._fit_scaler(X)
        alphas = self.params.get("alphas", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        n_splits = self.params.get("cv", 3)
        self.model_ = RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=n_splits))
        self.model_.fit(Xs, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._transform(X))


class LassoModel(_LinearScaledMixin):
    name = "lasso"
    family = "linear"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LassoModel":
        Xs = self._fit_scaler(X)
        self.model_ = LassoCV(
            alphas=self.params.get("alphas"),
            cv=TimeSeriesSplit(n_splits=self.params.get("cv", 3)),
            n_jobs=self.params.get("n_jobs", -1),
            random_state=self.params.get("random_state", 42),
            max_iter=self.params.get("max_iter", 5000),
        )
        self.model_.fit(Xs, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._transform(X))


class ElasticNetModel(_LinearScaledMixin):
    name = "elasticnet"
    family = "linear"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ElasticNetModel":
        Xs = self._fit_scaler(X)
        self.model_ = ElasticNetCV(
            alphas=self.params.get("alphas"),
            l1_ratio=self.params.get("l1_ratios", [0.1, 0.3, 0.5, 0.7, 0.9]),
            cv=TimeSeriesSplit(n_splits=self.params.get("cv", 3)),
            n_jobs=self.params.get("n_jobs", -1),
            random_state=self.params.get("random_state", 42),
            max_iter=self.params.get("max_iter", 5000),
        )
        self.model_.fit(Xs, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._transform(X))
