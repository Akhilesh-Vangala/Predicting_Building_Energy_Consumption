from __future__ import annotations

import numpy as np
import pandas as pd


def add_building_features(df: pd.DataFrame) -> pd.DataFrame:
    if "primary_use" in df.columns:
        if df["primary_use"].dtype.name == "category":
            df["primary_use_code"] = df["primary_use"].cat.codes.astype(np.int8)
        else:
            df["primary_use_code"] = pd.Categorical(df["primary_use"]).codes.astype(np.int8)

    if "building_age" in df.columns:
        df["building_age"] = df["building_age"].astype(np.float32).fillna(
            df["building_age"].median()
        )
    return df
