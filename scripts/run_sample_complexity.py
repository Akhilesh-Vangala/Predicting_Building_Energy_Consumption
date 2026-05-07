"""
Sample complexity curves: val RMSE vs. training-set size for Ridge, LightGBM, MLP.

Loads data once, then trains each model on chronologically-ordered fractions of
the training set (10 % → 100 %) while evaluating on the fixed validation set.

Outputs:
  results/metrics/sample_complexity.csv
  results/figures/sample_complexity.{pdf,png}
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.models import LightGBMModel, MLPModel, RidgeModel
from src.pipeline import evaluate_predictions, prepare_data
from src.utils import setup_logging

logger = logging.getLogger(__name__)

FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--feature-set", default="engineered")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    logger.info("Loading data...")
    prep = prepare_data(cfg, feature_set=args.feature_set, target_log=True)

    # Sort train chronologically so each fraction is the earliest N rows
    order = prep.train_full["timestamp"].argsort().to_numpy()
    X_full = prep.train_full[prep.feature_cols].iloc[order].reset_index(drop=True)
    y_full = prep.target_log_train[order]

    X_val = prep.val_full[prep.feature_cols]
    y_val = prep.target_log_val
    n_full = len(X_full)

    lgb_params = cfg.models.get("lightgbm", {})
    mlp_params = cfg.models.get("mlp", {})
    linear_params = cfg.models.get("linear", {})

    results = []
    for frac in FRACTIONS:
        n = max(1000, int(n_full * frac))
        X_tr = X_full.iloc[:n]
        y_tr = y_full[:n]
        logger.info("fraction=%.2f  n_train=%d", frac, n)

        for model_name, model in [
            ("ridge", RidgeModel(**linear_params)),
            ("lightgbm", LightGBMModel(**lgb_params)),
            ("mlp", MLPModel(**mlp_params)),
        ]:
            t0 = time.perf_counter()
            try:
                if isinstance(model, (LightGBMModel, MLPModel)):
                    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
                else:
                    model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
            except Exception as e:
                logger.warning("%s frac=%.2f failed: %s", model_name, frac, e)
                continue
            elapsed = time.perf_counter() - t0

            ev = evaluate_predictions(prep.val_full, preds, y_val, target_log=True)
            row = {
                "fraction": frac,
                "n_train": n,
                "model": model_name,
                "rmse": ev["overall"]["rmse"],
                "mae": ev["overall"]["mae"],
                "cv_rmse": ev["overall"]["cv_rmse"],
                "rmsle": ev["overall"]["rmsle"],
                "train_seconds": round(elapsed, 1),
            }
            results.append(row)
            logger.info("  [%s] RMSE=%.2f in %.1fs", model_name, row["rmse"], elapsed)

    df = pd.DataFrame(results)
    out_csv = cfg.paths.metrics / "sample_complexity.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Saved %s", out_csv)

    _plot(df, cfg.paths.plots)


def _plot(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from src.viz.style import PRIMARY, SUPPORT, save_figure, set_style

    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    palette = {"ridge": SUPPORT[0], "lightgbm": SUPPORT[1], "mlp": SUPPORT[3]}
    markers = {"ridge": "o", "lightgbm": "s", "mlp": "^"}

    for model_name, grp in df.groupby("model"):
        grp = grp.sort_values("n_train")
        ax.plot(
            grp["n_train"] / 1_000,
            grp["rmse"],
            marker=markers.get(model_name, "o"),
            color=palette.get(model_name, PRIMARY),
            label=model_name,
            linewidth=1.8,
            markersize=6,
        )

    ax.set_xlabel("Training set size (thousands of rows)")
    ax.set_ylabel("Validation RMSE (kWh)")
    ax.set_title("Sample complexity curves")
    ax.legend(title="model")
    ax.grid(True)
    fig.tight_layout()
    save_figure(fig, out_dir / "sample_complexity", formats=("pdf", "png"))
    plt.close(fig)
    logger.info("sample_complexity figure saved to %s", out_dir)


if __name__ == "__main__":
    main()
