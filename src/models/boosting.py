from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.base import BaseModel


class LightGBMModel(BaseModel):
    name = "lightgbm"
    family = "trees"

    def fit(self, X: pd.DataFrame, y: np.ndarray,
            eval_set: tuple[pd.DataFrame, np.ndarray] | None = None) -> "LightGBMModel":
        self.feature_names_ = X.columns.tolist()
        params = dict(
            objective=self.params.get("objective", "regression"),
            metric=self.params.get("metric", "rmse"),
            num_leaves=self.params.get("num_leaves", 63),
            learning_rate=self.params.get("learning_rate", 0.05),
            feature_fraction=self.params.get("feature_fraction", 0.9),
            bagging_fraction=self.params.get("bagging_fraction", 0.9),
            bagging_freq=self.params.get("bagging_freq", 5),
            min_data_in_leaf=self.params.get("min_data_in_leaf", 50),
            verbose=-1,
            seed=self.params.get("random_state", 42),
        )

        train_set = lgb.Dataset(X, label=y)
        valid_sets = [train_set]
        valid_names = ["train"]
        if eval_set is not None:
            xv, yv = eval_set
            valid_sets.append(lgb.Dataset(xv, label=yv, reference=train_set))
            valid_names.append("val")

        callbacks = []
        early = self.params.get("early_stopping_rounds", 50)
        if eval_set is not None and early:
            callbacks.append(lgb.early_stopping(early, verbose=False))
        callbacks.append(lgb.log_evaluation(0))

        self.model_ = lgb.train(
            params,
            train_set,
            num_boost_round=self.params.get("n_estimators", 1500),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("model not fit")
        return self.model_.predict(X, num_iteration=self.model_.best_iteration)

    def feature_importance(self) -> dict[str, float] | None:
        if self.model_ is None:
            return None
        imp = self.model_.feature_importance(importance_type="gain")
        return {n: float(v) for n, v in zip(self.model_.feature_name(), imp)}
