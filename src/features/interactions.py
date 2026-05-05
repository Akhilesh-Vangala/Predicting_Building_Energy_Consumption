from __future__ import annotations

import numpy as np
import pandas as pd


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    if "hour" in df.columns and "is_weekend" in df.columns:
        df["hour_x_weekend"] = (df["hour"] * df["is_weekend"]).astype(np.int16)

    if "log_square_feet" in df.columns and "air_temperature" in df.columns:
        df["sqft_x_temp"] = (
            df["log_square_feet"] * df["air_temperature"].fillna(0.0)
        ).astype(np.float32)

    return df
