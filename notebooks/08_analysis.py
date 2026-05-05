# %% [markdown]
# # 08 — Analysis: ablations, failure modes, and the remaining gap
#
# This notebook ties everything together for the report. It expects that
# `scripts/train_all.py` has already produced the metrics JSONs for both feature
# sets:
#
# - `results/metrics/models_engineered.json`
# - `results/metrics/models_raw.json`
#
# Run them with:
#
# ```
# python -m scripts.train_all --config configs/default.yaml --feature-set engineered
# python -m scripts.train_all --config configs/default.yaml --feature-set raw \
#     --skip arima lstm
# ```
#
# (We skip ARIMA and LSTM in the raw ablation: ARIMA does not use feature columns,
# and LSTM with only the 5 raw columns has no signal worth measuring.)

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
os.environ.setdefault("MPLCONFIGDIR", str(__import__("pathlib").Path.cwd() / ".mplcache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.viz.style import set_style, save_figure, PRIMARY, ACCENT, SUPPORT

set_style()
PLOT_DIR = Path("../results/plots/analysis")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


eng = load_metrics("../results/metrics/models_engineered.json")
raw = load_metrics("../results/metrics/models_raw.json")

# %% [markdown]
# ## 1. Headline comparison
#
# Every model in one table — RMSE, CV-RMSE, training time.

# %%
def metrics_table(payload: dict) -> pd.DataFrame:
    rows = []
    for name, m in payload["models"].items():
        rows.append({
            "model": name, "family": m["family"],
            "rmse": m["metrics"]["rmse"],
            "cv_rmse": m["metrics"]["cv_rmse"],
            "mae": m["metrics"]["mae"],
            "train_seconds": m["train_seconds"],
        })
    bl = payload.get("baselines", {})
    for k, v in bl.items():
        rows.append({
            "model": f"baseline_{k}", "family": "baseline",
            "rmse": v["metrics"].get("rmse"),
            "cv_rmse": v["metrics"].get("cv_rmse"),
            "mae": v["metrics"].get("mae"),
            "train_seconds": 0.0,
        })
    return pd.DataFrame(rows).sort_values("rmse")


eng_table = metrics_table(eng)
raw_table = metrics_table(raw)
eng_table

# %%
fig, ax = plt.subplots(figsize=(9, max(3.5, 0.4 * len(eng_table))))
fam_colors = {"baseline": "#888", "linear": SUPPORT[0], "trees": SUPPORT[1],
              "time_series": SUPPORT[2], "neural": SUPPORT[3]}
colors = [fam_colors.get(f, PRIMARY) for f in eng_table["family"]]
ax.barh(eng_table["model"][::-1], eng_table["rmse"][::-1], color=colors[::-1])
ax.set_xlabel("RMSE on validation (kWh)")
ax.set_title("All models on engineered features (lower is better)")
plt.tight_layout()
save_figure(fig, PLOT_DIR / "all_models_rmse")
plt.show()

# %% [markdown]
# ## 2. Ablation: 5 raw features vs. 28 engineered features
#
# The proposal predicts that the gap between linear models and LightGBM shrinks
# substantially with engineered features, because most of what feature engineering
# encodes (lags, rolling stats, target encoding, cyclical time) is something tree
# ensembles can also learn from raw inputs to some degree.

# %%
def ablation_pivot(eng_table: pd.DataFrame, raw_table: pd.DataFrame) -> pd.DataFrame:
    e = eng_table.set_index("model")["rmse"].rename("engineered")
    r = raw_table.set_index("model")["rmse"].rename("raw")
    out = pd.concat([e, r], axis=1)
    out["delta"] = out["raw"] - out["engineered"]
    out["pct_improvement"] = out["delta"] / out["raw"] * 100
    return out.dropna(subset=["engineered", "raw"]).sort_values("pct_improvement", ascending=False)


ablation = ablation_pivot(eng_table, raw_table)
ablation

# %%
fig, ax = plt.subplots(figsize=(10, max(3.5, 0.4 * len(ablation))))
order = ablation.index[::-1]
x = np.arange(len(order))
width = 0.4
ax.barh(x + width / 2, ablation.loc[order, "raw"], height=width,
        color=ACCENT, label="raw (5 features)")
ax.barh(x - width / 2, ablation.loc[order, "engineered"], height=width,
        color=PRIMARY, label="engineered (28 features)")
ax.set_yticks(x)
ax.set_yticklabels(order)
ax.set_xlabel("RMSE")
ax.set_title("Feature ablation — RMSE on validation")
ax.legend()
plt.tight_layout()
save_figure(fig, PLOT_DIR / "ablation_rmse")
plt.show()

# %% [markdown]
# ## 3. Per-primary-use breakdown (engineered)
#
# Where do the wins concentrate? We expect Education and Office (regular schedules)
# to be cheap to predict; Lodging and Healthcare to expose where simpler models
# fail.

# %%
def per_use_pivot(payload: dict) -> pd.DataFrame:
    rows = []
    for name, m in payload["models"].items():
        for r in m.get("by_primary_use", []):
            rows.append({"model": name, **r})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.pivot_table(index="primary_use", columns="model", values="rmse")


pu_eng = per_use_pivot(eng)
pu_eng.round(0)

# %%
if not pu_eng.empty:
    fig, ax = plt.subplots(figsize=(min(13, 1.2 + 0.7 * len(pu_eng.columns)),
                                    max(4, 0.4 * len(pu_eng))))
    sns.heatmap(pu_eng, annot=True, fmt=".0f", cmap="Reds", ax=ax,
                cbar_kws={"label": "RMSE"})
    ax.set_title("RMSE by primary use × model (engineered features)")
    plt.tight_layout()
    save_figure(fig, PLOT_DIR / "per_primary_use_heatmap")
    plt.show()

# %% [markdown]
# ## 4. Per-meter-type breakdown (engineered)

# %%
def per_meter_pivot(payload: dict) -> pd.DataFrame:
    rows = []
    for name, m in payload["models"].items():
        for r in m.get("by_meter", []):
            rows.append({"model": name, **r})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.pivot_table(index="meter_name", columns="model", values="rmse")


pm_eng = per_meter_pivot(eng)
pm_eng.round(0)

# %%
if not pm_eng.empty:
    fig, ax = plt.subplots(figsize=(min(13, 1.2 + 0.7 * len(pm_eng.columns)),
                                    max(3.5, 0.5 * len(pm_eng))))
    sns.heatmap(pm_eng, annot=True, fmt=".0f", cmap="Reds", ax=ax,
                cbar_kws={"label": "RMSE"})
    ax.set_title("RMSE by meter type × model")
    plt.tight_layout()
    save_figure(fig, PLOT_DIR / "per_meter_heatmap")
    plt.show()

# %% [markdown]
# ## 5. Worst-predicted buildings for the best model
#
# What is the gap that LightGBM cannot close? The buildings with the largest
# residuals tell us what extra information would matter. Common culprits: missing
# occupancy schedules, renovations, equipment replacements, and metering errors.

# %%
y_true = np.load("../results/metrics/predictions_engineered/y_true_raw.npy")
lgbm_log = np.load("../results/metrics/predictions_engineered/lightgbm_log_preds.npy")
lgbm_pred = np.expm1(np.clip(lgbm_log, 0.0, 14.0))

val_full_path = Path("../data/processed/val_features.parquet")
if val_full_path.exists():
    val = pd.read_parquet(val_full_path)
    val = val.iloc[:len(y_true)].copy()
    val["abs_err"] = np.abs(lgbm_pred - y_true)

    by_building = (
        val.groupby(["building_id", "primary_use"], observed=True)["abs_err"]
           .mean().sort_values(ascending=False).head(20)
           .reset_index()
    )
    print(by_building.to_string(index=False))
else:
    print("(parquet not found; run scripts/build_features.py first)")

# %% [markdown]
# ## 6. Take-aways
#
# - Engineered features close most of the gap between linear and tree-based
#   models, replicating the GEPIII competition finding (Miller et al. 2022a).
# - Tree ensembles dominate; LightGBM is the best single model.
# - Sequence models (ARIMA, LSTM) are at most competitive with LightGBM, not
#   ahead — consistent with the Grinsztajn et al. (2022) result that engineered
#   tabular features encode most of what RNNs can learn.
# - Hot water and steam meters are the consistent failure mode across families.
# - Per-cluster LightGBM (notebook 07) gives a small uplift but does not change
#   the overall ranking; clustering helps mostly on the noisier sites.

# %%
out_dir = Path("../results/tables")
out_dir.mkdir(parents=True, exist_ok=True)
eng_table.to_csv(out_dir / "all_models_engineered.csv", index=False)
raw_table.to_csv(out_dir / "all_models_raw.csv", index=False)
ablation.to_csv(out_dir / "ablation_table.csv")
print(f"wrote summary tables to {out_dir}")
