# %% [markdown]
# # 07 — Clustering experiment
#
# K-means on each meter's normalised hour-of-day consumption profile (24-vector,
# row-summed to 1). The proposal hypothesis: reducing the within-cluster
# heterogeneity should let a per-cluster LightGBM beat a single global LightGBM,
# especially on sites with diverse building mixes.
#
# We pick `k` with the elbow method (largest distance from the line connecting
# the smallest and largest `k` in the inertia curve), train one LightGBM per
# cluster on the same features and split, then compare per-cluster aggregate
# RMSE against the single global LightGBM.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(__import__("pathlib").Path.cwd() / ".mplcache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import load_config
from src.pipeline import prepare_data, evaluate_predictions
from src.clustering.kmeans_profiles import (
    build_consumption_profiles, fit_kmeans, elbow_curve, assign_clusters,
)
from src.evaluation.metrics import score_predictions
from src.models import LightGBMModel
from src.viz.style import set_style, save_figure, PRIMARY, ACCENT, SUPPORT

set_style()
PLOT_DIR = Path("../results/plots/clustering")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

cfg = load_config("../configs/default.yaml")
prep = prepare_data(cfg, feature_set="engineered")

# %% [markdown]
# ## Step 1 — build profiles
#
# Per `(building_id, meter)`, average meter_reading at each hour of the day, then
# divide by the row sum. The resulting 24-vector tells us *when* during the day a
# meter draws its load, regardless of total magnitude.

# %%
profiles = build_consumption_profiles(prep.train_full)
print(f"profiles for {len(profiles):,} meters, {profiles.shape[1]} hour bins")
profiles.head()

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
sample = profiles.sample(min(60, len(profiles)), random_state=42)
for i, (_, row) in enumerate(sample.iterrows()):
    ax.plot(row.values, color=PRIMARY, alpha=0.18, linewidth=0.7)
ax.plot(profiles.mean().values, color=ACCENT, linewidth=2.5, label="mean profile")
ax.set_xlabel("hour")
ax.set_ylabel("normalised consumption")
ax.set_title("Normalised hour-of-day profiles (sample of 60 meters)")
ax.legend()
plt.tight_layout()
save_figure(fig, PLOT_DIR / "profile_sample")
plt.show()

# %% [markdown]
# ## Step 2 — elbow method for `k`

# %%
curve = elbow_curve(profiles, cfg.clustering.k_range, random_state=cfg.clustering.random_state)
curve

# %%
def pick_elbow(curve: pd.DataFrame) -> int:
    ks = curve["k"].to_numpy(dtype=float)
    inertias = curve["inertia"].to_numpy(dtype=float)
    if len(ks) < 3:
        return int(ks[0])
    p1 = np.array([ks[0], inertias[0]])
    p2 = np.array([ks[-1], inertias[-1]])
    line = (p2 - p1) / np.linalg.norm(p2 - p1)
    distances = []
    for x, y in zip(ks, inertias):
        v = np.array([x, y]) - p1
        proj = v - (v @ line) * line
        distances.append(np.linalg.norm(proj))
    return int(ks[int(np.argmax(distances))])

k_star = pick_elbow(curve)
print(f"elbow k* = {k_star}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(curve["k"], curve["inertia"], marker="o", color=PRIMARY)
ax.axvline(k_star, color=ACCENT, linestyle="--", label=f"k* = {k_star}")
ax.set_xlabel("k")
ax.set_ylabel("inertia")
ax.set_title("Elbow method")
ax.legend()
plt.tight_layout()
save_figure(fig, PLOT_DIR / "elbow")
plt.show()

# %% [markdown]
# ## Step 3 — fit K-means and inspect cluster centroids
#
# Each centroid is itself a 24-vector — a "canonical" daily profile. We expect to
# see a few archetypes: working-hours buildings (peak around 10–17), nightshift
# operations (flat or inverted), and 24/7 facilities (hospital, lodging).

# %%
km = fit_kmeans(profiles, k_star, random_state=cfg.clustering.random_state)
labels = assign_clusters(km, profiles).reset_index()
labels["cluster"] = labels["cluster"].astype(int)
print(labels["cluster"].value_counts().sort_index())

# %%
centroids = pd.DataFrame(km.cluster_centers_, columns=profiles.columns)
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(k_star):
    ax.plot(centroids.iloc[i].values, label=f"cluster {i}",
            color=SUPPORT[i % len(SUPPORT)], linewidth=2)
ax.set_xlabel("hour")
ax.set_ylabel("normalised consumption")
ax.set_title(f"K-means centroids (k = {k_star})")
ax.legend()
plt.tight_layout()
save_figure(fig, PLOT_DIR / "centroids")
plt.show()

# %% [markdown]
# Match clusters to primary use to see what archetypes the algorithm found.

# %%
labels_meta = labels.merge(
    prep.train_full[["building_id", "primary_use"]].drop_duplicates(),
    on="building_id", how="left",
)
breakdown = (
    labels_meta.pivot_table(index="cluster", columns="primary_use",
                            values="building_id", aggfunc="count", fill_value=0)
)
breakdown_pct = breakdown.div(breakdown.sum(axis=1), axis=0).round(3) * 100

fig, ax = plt.subplots(figsize=(13, 4.5))
sns.heatmap(breakdown_pct, annot=True, fmt=".0f", cmap="Blues", ax=ax,
            cbar_kws={"label": "% of cluster"})
ax.set_title("Primary-use composition of each cluster (%)")
plt.tight_layout()
save_figure(fig, PLOT_DIR / "cluster_x_primary_use")
plt.show()

# %% [markdown]
# ## Step 4 — per-cluster LightGBM vs. global LightGBM

# %%
train = prep.train_full.merge(labels, on=["building_id", "meter"], how="left").reset_index(drop=True)
val = prep.val_full.merge(labels, on=["building_id", "meter"], how="left").reset_index(drop=True)

cluster_metrics: list[dict] = []
val_preds_real = np.full(len(val), np.nan, dtype=np.float64)
val_true_real = np.full(len(val), np.nan, dtype=np.float64)
val_cluster_arr = val["cluster"].to_numpy()
train_cluster_arr = train["cluster"].to_numpy()

for c in sorted(pd.Series(train_cluster_arr).dropna().unique()):
    train_pos = np.where(train_cluster_arr == c)[0]
    val_pos = np.where(val_cluster_arr == c)[0]
    if len(train_pos) < 1000 or len(val_pos) == 0:
        print(f"skipping cluster {int(c)} (train={len(train_pos)} val={len(val_pos)})")
        continue
    train_c = train.iloc[train_pos]
    val_c = val.iloc[val_pos]
    m = LightGBMModel(**cfg.models.get("lightgbm", {}))
    Xt = train_c[prep.feature_cols]; Xv = val_c[prep.feature_cols]
    yt = np.log1p(np.clip(train_c["meter_reading"].to_numpy(), 0, None))
    yv = np.log1p(np.clip(val_c["meter_reading"].to_numpy(), 0, None))
    t0 = time.time()
    m.fit(Xt, yt, eval_set=(Xv, yv))
    secs = time.time() - t0
    preds = m.predict(Xv)
    pred_real = np.expm1(np.clip(preds, 0.0, 14.0))
    actual_real = np.expm1(np.clip(yv, 0.0, 14.0))
    rmse = float(np.sqrt(np.mean((pred_real - actual_real) ** 2)))
    val_preds_real[val_pos] = pred_real
    val_true_real[val_pos] = actual_real
    cluster_metrics.append({
        "cluster": int(c), "n_train": int(len(train_pos)), "n_val": int(len(val_pos)),
        "rmse": rmse, "train_seconds": secs,
    })
    print(f"cluster {int(c)}: RMSE {rmse:.1f}  (train {secs:.1f}s)")

per_cluster = pd.DataFrame(cluster_metrics)
per_cluster

# %%
valid = ~np.isnan(val_preds_real) & ~np.isnan(val_true_real)
overall_per_cluster = score_predictions(val_true_real[valid], val_preds_real[valid])
print(f"per-cluster aggregate RMSE: {overall_per_cluster['rmse']:.1f}")

# Global LightGBM for comparison
global_lgbm = LightGBMModel(**cfg.models.get("lightgbm", {}))
global_lgbm.fit(prep.train_full[prep.feature_cols], prep.target_log_train,
                eval_set=(prep.val_full[prep.feature_cols], prep.target_log_val))
global_preds = global_lgbm.predict(prep.val_full[prep.feature_cols])
global_real = np.expm1(np.clip(global_preds, 0.0, 14.0))
global_rmse = float(np.sqrt(np.mean((global_real - prep.target_raw_val) ** 2)))
print(f"global LightGBM RMSE: {global_rmse:.1f}")
print(f"clustering uplift: {(global_rmse - overall_per_cluster['rmse']) / global_rmse * 100:+.1f}%")

# %%
out_dir = Path("../results/tables")
out_dir.mkdir(parents=True, exist_ok=True)
curve.to_csv(out_dir / "cluster_elbow.csv", index=False)
labels.to_csv(out_dir / "cluster_labels.csv", index=False)
per_cluster.to_csv(out_dir / "cluster_per_cluster_lgbm.csv", index=False)
pd.DataFrame([
    {"model": "per_cluster_lightgbm", "rmse": overall_per_cluster["rmse"]},
    {"model": "global_lightgbm", "rmse": global_rmse},
]).to_csv(out_dir / "cluster_vs_global.csv", index=False)
print(f"wrote tables to {out_dir}")
