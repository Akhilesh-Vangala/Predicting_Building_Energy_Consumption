from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Paths:
    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    results: Path = Path("results")
    models: Path = Path("results/models")
    plots: Path = Path("results/plots")
    metrics: Path = Path("results/metrics")
    tables: Path = Path("results/tables")

    def make_dirs(self) -> None:
        for p in (self.processed, self.results, self.models, self.plots,
                  self.metrics, self.tables):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    train_csv: Path = Path("data/raw/train.csv")
    meta_csv: Path = Path("data/raw/building_metadata.csv")
    weather_csv: Path = Path("data/raw/weather_train.csv")
    site_subset: list[int] = field(default_factory=lambda: [2, 4, 13, 14])
    use_full: bool = False
    reference_year: int = 2017
    outlier_quantile: float = 0.999
    zero_streak_min_hours: int = 48


@dataclass
class SplitConfig:
    train_months: list[int] = field(default_factory=lambda: list(range(1, 10)))
    val_months: list[int] = field(default_factory=lambda: [10, 11, 12])


@dataclass
class FeaturesConfig:
    lag_hours: list[int] = field(default_factory=lambda: [24, 168])
    rolling_windows: list[int] = field(default_factory=lambda: [24, 168])
    use_target_encoding: bool = True
    target_encoding_smoothing: int = 30


@dataclass
class ClusteringConfig:
    k_range: list[int] = field(default_factory=lambda: list(range(2, 11)))
    random_state: int = 42


@dataclass
class AppConfig:
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    models: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    random_state: int = 42


def _coerce_paths(d: dict, base: Path | None = None) -> Paths:
    return Paths(**{k: _resolve(Path(v), base) for k, v in d.items()})


def _resolve(p: Path, base: Path | None) -> Path:
    if base is None or p.is_absolute():
        return p
    return (base / p).resolve()


def load_config(path: str | Path) -> AppConfig:
    cfg_path = Path(path).resolve()
    base = cfg_path.parent.parent
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = AppConfig()
    if "paths" in raw:
        cfg.paths = _coerce_paths(raw["paths"], base=base)
    if "data" in raw:
        d = dict(raw["data"])
        d["train_csv"] = _resolve(Path(d.get("train_csv", cfg.data.train_csv)), base)
        d["meta_csv"] = _resolve(Path(d.get("meta_csv", cfg.data.meta_csv)), base)
        d["weather_csv"] = _resolve(Path(d.get("weather_csv", cfg.data.weather_csv)), base)
        cfg.data = DataConfig(**d)
    if "split" in raw:
        cfg.split = SplitConfig(**raw["split"])
    if "features" in raw:
        f_raw = dict(raw["features"])
        f_raw.pop("cyclical", None)
        cfg.features = FeaturesConfig(**f_raw)
    if "clustering" in raw:
        cfg.clustering = ClusteringConfig(**raw["clustering"])
    if "models" in raw:
        cfg.models = raw["models"]
    if "evaluation" in raw:
        cfg.evaluation = raw["evaluation"]
    if "random_state" in raw:
        cfg.random_state = int(raw["random_state"])

    cfg.paths.make_dirs()
    return cfg
