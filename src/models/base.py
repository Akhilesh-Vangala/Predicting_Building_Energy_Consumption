from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelResult:
    name: str
    family: str
    train_seconds: float
    val_predictions: np.ndarray
    val_targets: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)
    feature_importance: dict[str, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    name: str = "base"
    family: str = "base"

    def __init__(self, **kwargs: Any) -> None:
        self.params = kwargs
        self.model_: Any = None
        self.feature_names_: list[str] | None = None
        self.train_seconds_: float = 0.0

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def fit_predict(self, X_train: pd.DataFrame, y_train: np.ndarray,
                    X_val: pd.DataFrame) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        self.fit(X_train, y_train)
        self.train_seconds_ = time.perf_counter() - t0
        preds = self.predict(X_val)
        return preds, self.train_seconds_

    def feature_importance(self) -> dict[str, float] | None:
        return None
