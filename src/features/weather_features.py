from __future__ import annotations

import numpy as np
import pandas as pd


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    if "air_temperature" in df.columns:
        df["temp_squared"] = (df["air_temperature"] ** 2).astype(np.float32)
    return df
