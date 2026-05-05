from __future__ import annotations

import numpy as np


def _validate(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def cv_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate(y_true, y_pred)
    mean_true = np.mean(y_true)
    if mean_true == 0:
        return float("nan")
    return float(rmse(y_true, y_pred) / mean_true)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate(y_true, y_pred)
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "cv_rmse": cv_rmse(y_true, y_pred),
        "rmsle": rmsle(y_true, y_pred),
    }
