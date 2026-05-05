from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import AppConfig

logger = logging.getLogger(__name__)

METER_NAMES = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}

_TRAIN_DTYPES = {
    "building_id": np.int16,
    "meter": np.int8,
    "meter_reading": np.float32,
}

_META_DTYPES = {
    "site_id": np.int8,
    "building_id": np.int16,
    "square_feet": np.float32,
    "year_built": np.float32,
    "floor_count": np.float32,
}

_WEATHER_DTYPES = {
    "site_id": np.int8,
    "air_temperature": np.float32,
    "dew_temperature": np.float32,
    "cloud_coverage": np.float32,
    "precip_depth_1_hr": np.float32,
    "sea_level_pressure": np.float32,
    "wind_direction": np.float32,
    "wind_speed": np.float32,
}


def load_building_metadata(cfg: AppConfig) -> pd.DataFrame:
    path = cfg.data.meta_csv
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Could not find building metadata at {path}. See data/raw/README.md."
        )
    meta = pd.read_csv(path, dtype=_META_DTYPES)
    meta["primary_use"] = meta["primary_use"].astype("category")
    ref_year = cfg.data.reference_year
    meta["building_age"] = (ref_year - meta["year_built"]).astype(np.float32)
    meta["log_square_feet"] = np.log1p(meta["square_feet"]).astype(np.float32)
    logger.info("Loaded metadata for %d buildings across %d sites",
                len(meta), meta["site_id"].nunique())
    return meta


def load_weather(cfg: AppConfig) -> pd.DataFrame:
    path = cfg.data.weather_csv
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Could not find weather data at {path}. See data/raw/README.md."
        )
    weather = pd.read_csv(path, dtype=_WEATHER_DTYPES, parse_dates=["timestamp"])
    logger.info("Loaded %d hourly weather records across %d sites",
                len(weather), weather["site_id"].nunique())
    return weather


def load_meter_readings(
    cfg: AppConfig,
    site_ids: Iterable[int] | None = None,
    chunk_size: int = 2_000_000,
) -> pd.DataFrame:
    path = cfg.data.train_csv
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Could not find training data at {path}. See data/raw/README.md."
        )

    site_set = set(int(s) for s in site_ids) if site_ids is not None else None
    if site_set is not None:
        meta = load_building_metadata(cfg)
        keep_buildings = set(
            meta.loc[meta["site_id"].isin(site_set), "building_id"].astype(int).tolist()
        )
        logger.info("Filtering to %d buildings from %d sites",
                    len(keep_buildings), len(site_set))
    else:
        keep_buildings = None

    frames: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        dtype=_TRAIN_DTYPES,
        parse_dates=["timestamp"],
        chunksize=chunk_size,
    )
    rows_in = 0
    rows_kept = 0
    for chunk in reader:
        rows_in += len(chunk)
        if keep_buildings is not None:
            chunk = chunk[chunk["building_id"].isin(keep_buildings)]
        rows_kept += len(chunk)
        if len(chunk) > 0:
            frames.append(chunk)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    logger.info("Read %s rows, kept %s after site filter",
                f"{rows_in:,}", f"{rows_kept:,}")
    return df


def load_subset(cfg: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = load_building_metadata(cfg)
    weather = load_weather(cfg)

    if cfg.data.use_full:
        readings = load_meter_readings(cfg, site_ids=None)
    else:
        readings = load_meter_readings(cfg, site_ids=cfg.data.site_subset)

    return readings, meta, weather


def merge_with_context(
    readings: pd.DataFrame,
    meta: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    df = readings.merge(meta, on="building_id", how="left")
    df = df.merge(weather, on=["site_id", "timestamp"], how="left")
    return df
