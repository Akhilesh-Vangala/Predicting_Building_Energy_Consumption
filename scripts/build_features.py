from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib

from src.config import load_config
from src.pipeline import prepare_data
from src.utils import save_json, set_seed, setup_logging, timer


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.random_state)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with timer("prepare_data"):
        prep = prepare_data(cfg, target_log=True, feature_set="engineered")

    prep.train_full.to_parquet(out / "train_features.parquet")
    prep.val_full.to_parquet(out / "val_features.parquet")
    joblib.dump(prep.encoder, out / "target_encoder.joblib")
    save_json(prep.summary, out / "feature_summary.json")
    logging.getLogger(__name__).info("Wrote features to %s", out)


if __name__ == "__main__":
    main()
