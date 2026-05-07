"""Generate predicted-vs-actual scatter and residual plots for the paper."""
from __future__ import annotations
import logging, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline import prepare_data
from src.viz.style import set_style, save_figure

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

LATEX_FIGURES = Path("/Users/akhileshvangala/Desktop/ml latex/figures")
PLOT_DIR = ROOT / "results" / "plots" / "analysis"
LATEX_FIGURES.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

METER_LABELS = {0: "Electricity", 1: "Chilled water", 2: "Steam", 3: "Hot water"}
METER_COLORS = {
    "Electricity":   "#2a9d8f",
    "Chilled water": "#e9c46a",
    "Steam":         "#e76f51",
    "Hot water":     "#264653",
}
CAP = np.expm1(14)   # 1.2M kWh — same cap used in evaluation


def save(fig: plt.Figure, stem: str) -> None:
    for dest in [PLOT_DIR, LATEX_FIGURES]:
        fig.savefig(dest / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(dest / f"{stem}.png", bbox_inches="tight", dpi=200)
    log.info("Saved %s", stem)


def main() -> None:
    set_style()

    cfg  = load_config(ROOT / "configs" / "default.yaml")
    prep = prepare_data(cfg, feature_set="engineered")

    y_true_raw = np.load(ROOT / "results/metrics/predictions_engineered/y_true_raw.npy")
    rf_log     = np.load(ROOT / "results/metrics/predictions_engineered/random_forest_log_preds.npy")
    lgb_log    = np.load(ROOT / "results/metrics/predictions_engineered/lightgbm_log_preds.npy")

    rf_pred  = np.clip(np.expm1(np.clip(rf_log,  0, 14)), 0, CAP)
    lgb_pred = np.clip(np.expm1(np.clip(lgb_log, 0, 14)), 0, CAP)
    y_true   = np.clip(y_true_raw, 0, CAP)

    meter = prep.val_full["meter"].to_numpy()
    meter_name = pd.Series(meter).map(METER_LABELS).to_numpy()

    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------ #
    # Figure A: log-scale pred vs actual scatter (RF), stratified sample
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    for ax, (preds, model_label) in zip(axes, [(rf_pred, "Random Forest"), (lgb_pred, "LightGBM")]):
        for mname, mcolor in METER_COLORS.items():
            mask = meter_name == mname
            n = mask.sum()
            k = min(4000, n)
            idx = rng.choice(np.where(mask)[0], size=k, replace=False)

            act = y_true[idx]
            pr  = preds[idx]
            # use log10 so axis labels are readable
            ax.scatter(
                np.log10(act + 1), np.log10(pr + 1),
                s=2, alpha=0.25, color=mcolor, label=mname, rasterized=True,
            )

        # Perfect prediction line
        lim = [0, np.log10(CAP + 1)]
        ax.plot(lim, lim, "k--", lw=0.8, label="Perfect")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("log₁₀(Actual + 1)  [kWh]", fontsize=9)
        ax.set_ylabel("log₁₀(Predicted + 1)  [kWh]", fontsize=9)
        ax.set_title(model_label, fontsize=10, fontweight="bold")
        ax.set_aspect("equal")

    # Shared legend on first axis
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, markerscale=4, fontsize=8,
                   loc="upper left", frameon=False)

    fig.suptitle("Predicted vs.\ Actual Consumption by Meter Type",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "pred_vs_actual_scatter")
    plt.close("all")

    # ------------------------------------------------------------------ #
    # Figure B: residual (pred − actual) distribution by meter type
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.2))
    order = ["Electricity", "Hot water", "Chilled water", "Steam"]

    for ax, mname in zip(axes, order):
        mask = meter_name == mname
        err = rf_pred[mask] - y_true[mask]
        # clip for readability; note what fraction is outside
        xlim = np.percentile(np.abs(err), 99)
        ax.hist(err, bins=80, color=METER_COLORS[mname], alpha=0.85,
                range=(-xlim, xlim), edgecolor="none")
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.axvline(err.mean(), color="red", lw=1.0, ls="--",
                   label=f"Bias = {err.mean():.0f}")
        rmse = np.sqrt(np.mean(err**2))
        ax.set_title(f"{mname}\nRMSE {rmse:,.0f} kWh", fontsize=8, fontweight="bold")
        ax.set_xlabel("Residual (kWh)", fontsize=7)
        ax.set_yticklabels([])
        ax.legend(fontsize=7, frameon=False)

    fig.suptitle("RF Residual Distributions by Meter Type",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    save(fig, "residual_by_meter")
    plt.close("all")

    log.info("Done.")


if __name__ == "__main__":
    main()
