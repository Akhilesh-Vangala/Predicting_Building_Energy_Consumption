from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from src.models.base import BaseModel


class DecisionTreeModel(BaseModel):
    name = "decision_tree"
    family = "trees"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "DecisionTreeModel":
        self.feature_names_ = X.columns.tolist()
        self.model_ = DecisionTreeRegressor(
            max_depth=self.params.get("max_depth", 16),
            min_samples_leaf=self.params.get("min_samples_leaf", 50),
            random_state=self.params.get("random_state", 42),
        )
        self.model_.fit(X.to_numpy(dtype=np.float32), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X.to_numpy(dtype=np.float32))

    def feature_importance(self) -> dict[str, float] | None:
        if self.model_ is None or self.feature_names_ is None:
            return None
        imp = self.model_.feature_importances_
        return {n: float(v) for n, v in zip(self.feature_names_, imp)}


class RandomForestModel(BaseModel):
    name = "random_forest"
    family = "trees"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RandomForestModel":
        self.feature_names_ = X.columns.tolist()
        self.model_ = RandomForestRegressor(
            n_estimators=self.params.get("n_estimators", 200),
            max_depth=self.params.get("max_depth", 16),
            min_samples_leaf=self.params.get("min_samples_leaf", 50),
            max_samples=self.params.get("max_samples", None),
            bootstrap=True,
            n_jobs=self.params.get("n_jobs", -1),
            random_state=self.params.get("random_state", 42),
        )
        self.model_.fit(X.to_numpy(dtype=np.float32), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X.to_numpy(dtype=np.float32))

    def feature_importance(self) -> dict[str, float] | None:
        if self.model_ is None or self.feature_names_ is None:
            return None
        imp = self.model_.feature_importances_
        return {n: float(v) for n, v in zip(self.feature_names_, imp)}
