from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import score_predictions

METER_NAMES = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}


def per_group_metrics(
    df: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    group_col: str = "primary_use",
) -> pd.DataFrame:
    rows: list[dict] = []
    for key, grp in df.groupby(group_col, observed=True):
        if len(grp) == 0:
            continue
        m = score_predictions(grp[y_true_col].to_numpy(), grp[y_pred_col].to_numpy())
        m[group_col] = key
        m["n"] = len(grp)
        rows.append(m)
    out = pd.DataFrame(rows)
    cols = [group_col, "n", "rmse", "mae", "cv_rmse", "rmsle"]
    return out[[c for c in cols if c in out.columns]].sort_values("rmse")


def per_building_type(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return per_group_metrics(df, group_col="primary_use", **kwargs)


def per_meter_type(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    if "meter_name" not in df.columns and "meter" in df.columns:
        df = df.copy()
        df["meter_name"] = df["meter"].map(METER_NAMES)
    return per_group_metrics(df, group_col="meter_name", **kwargs)


def per_site(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return per_group_metrics(df, group_col="site_id", **kwargs)


def per_building(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return per_group_metrics(df, group_col="building_id", **kwargs)
