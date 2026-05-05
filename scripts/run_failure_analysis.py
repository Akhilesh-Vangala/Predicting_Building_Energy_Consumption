from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.metrics import score_predictions
from src.evaluation.per_group import per_building, per_building_type, per_meter_type, per_site
from src.utils import load_json, save_json, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-set", default="engineered")
    args = parser.parse_args()

    cfg = load_config(args.config)
    detailed_path = cfg.paths.metrics / f"models_{args.feature_set}.json"
    if not detailed_path.exists():
        raise FileNotFoundError(f"missing {detailed_path}; run train_all first")
    detailed = load_json(detailed_path)

    rows = []
    for model_name, payload in detailed["models"].items():
        for grp in payload.get("by_primary_use", []):
            grp = dict(grp); grp["model"] = model_name; grp["group_kind"] = "primary_use"
            rows.append(grp)
        for grp in payload.get("by_meter", []):
            grp = dict(grp); grp["model"] = model_name; grp["group_kind"] = "meter"
            rows.append(grp)
        for grp in payload.get("by_site", []):
            grp = dict(grp); grp["model"] = model_name; grp["group_kind"] = "site"
            rows.append(grp)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("no per-group rows; nothing to do")
        return

    table = df.melt(id_vars=["model", "group_kind"], value_vars=["rmse", "cv_rmse", "mae"],
                    var_name="metric", value_name="value")
    table.to_csv(cfg.paths.tables / f"failure_long_{args.feature_set}.csv", index=False)

    pivot_use = (
        df[df["group_kind"] == "primary_use"]
          .pivot_table(index="primary_use", columns="model", values="rmse", aggfunc="mean")
    )
    pivot_meter = (
        df[df["group_kind"] == "meter"]
          .pivot_table(index="meter_name", columns="model", values="rmse", aggfunc="mean")
    )
    pivot_use.to_csv(cfg.paths.tables / f"failure_by_primary_use_{args.feature_set}.csv")
    pivot_meter.to_csv(cfg.paths.tables / f"failure_by_meter_{args.feature_set}.csv")

    summary = {
        "feature_set": args.feature_set,
        "worst_use_per_model": {
            m: pivot_use[m].sort_values(ascending=False).head(3).to_dict()
            for m in pivot_use.columns
        },
        "worst_meter_per_model": {
            m: pivot_meter[m].sort_values(ascending=False).head(3).to_dict()
            for m in pivot_meter.columns
        },
    }
    save_json(summary, cfg.paths.metrics / f"failure_analysis_{args.feature_set}.json")
    logger.info("Wrote failure analysis to %s", cfg.paths.tables)


if __name__ == "__main__":
    main()
