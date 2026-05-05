from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import cv_rmse, mae, rmse, rmsle, score_predictions


def test_rmse_zero_when_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0


def test_rmse_known_value() -> None:
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    assert rmse(y_true, y_pred) == pytest.approx(1.0)


def test_cv_rmse_matches_definition() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 35.0])
    expected = rmse(y_true, y_pred) / np.mean(y_true)
    assert cv_rmse(y_true, y_pred) == pytest.approx(expected)


def test_score_predictions_keys() -> None:
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 2.0])
    out = score_predictions(y_true, y_pred)
    assert {"rmse", "mae", "cv_rmse", "rmsle"} <= out.keys()


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        rmse(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
