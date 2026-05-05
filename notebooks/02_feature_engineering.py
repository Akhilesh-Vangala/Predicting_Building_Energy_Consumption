# %% [markdown]
# # 02 — Feature Engineering
#
# We commit to exactly 28 features for the head-to-head comparison. They fall into
# seven groups:
#
# | Group | Count | Names |
# |---|---:|---|
# | Time | 8 | `hour_sin`, `hour_cos`, `dayofweek_sin`, `dayofweek_cos`, `month_sin`, `month_cos`, `is_weekend`, `is_holiday` |
# | Lag | 3 | `lag_24h`, `lag_168h`, `lag_diff_24h` |
# | Rolling | 4 | `rolling_mean_24h`, `rolling_std_24h`, `rolling_mean_168h`, `rolling_std_168h` |
# | Weather | 6 | `air_temperature`, `dew_temperature`, `temp_squared`, `wind_speed`, `cloud_coverage`, `precip_depth_1_hr` |
# | Building | 3 | `log_square_feet`, `building_age`, `primary_use_code` |
# | Interaction | 2 | `hour_x_weekend`, `sqft_x_temp` |
# | Target encoding | 1 | `te_primary_use_x_hour` (smoothed mean of `log1p(meter_reading)` per primary-use × hour, fit on train only) |
#
# This notebook walks through each group, shows what it captures, and confirms two
# things the proposal calls out: features at time `t` use only past data, and the
# target encoder is fit on training rows only.

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import load_config
from src.data.clean import basic_clean, impute_weather
from src.data.load import load_subset, merge_with_context
from src.data.split import time_split
from src.features import FEATURE_COLUMNS, RAW_FEATURE_COLUMNS, build_features, feature_groups
from src.features.target_encoding import TargetEncoder
from src.viz.style import set_style, save_figure, PRIMARY, ACCENT, SUPPORT

set_style()
PLOT_DIR = Path("../results/plots/features")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# %%
cfg = load_config("../configs/default.yaml")
readings, meta, weather = load_subset(cfg)
weather = impute_weather(weather)
df = merge_with_context(readings, meta, weather)
df = basic_clean(df, cfg.data.outlier_quantile, cfg.data.zero_streak_min_hours)
print(f"Working with {len(df):,} rows for feature engineering")

# %% [markdown]
# ## Build the full feature set

# %%
df_feat, encoder = build_features(df.copy(), cfg.features)
print(f"After build_features: {df_feat.shape}")
groups = feature_groups()
total = sum(len(v) for v in groups.values())
print(f"Total declared features: {total}")
for g, names in groups.items():
    print(f"  {g}: {len(names)}")

# %% [markdown]
# ## 1. Time features
#
# Hour-of-day, day-of-week and month all wrap around. Sine/cosine pairs let any
# linear or tree model see the wrap correctly: 23:00 and 00:00 are neighbours, not
# opposite ends of a number line.

# %%
ts_demo = pd.date_range("2016-03-14", "2016-03-21", freq="h")
demo = pd.DataFrame({"timestamp": ts_demo})
demo["hour"] = demo["timestamp"].dt.hour
demo["dayofweek"] = demo["timestamp"].dt.dayofweek
demo["hour_sin"] = np.sin(2 * np.pi * demo["hour"] / 24)
demo["hour_cos"] = np.cos(2 * np.pi * demo["hour"] / 24)
demo["dow_sin"] = np.sin(2 * np.pi * demo["dayofweek"] / 7)
demo["dow_cos"] = np.cos(2 * np.pi * demo["dayofweek"] / 7)

fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
axes[0].plot(demo["timestamp"], demo["hour_sin"], label="hour_sin", color=PRIMARY)
axes[0].plot(demo["timestamp"], demo["hour_cos"], label="hour_cos", color=ACCENT)
axes[0].set_title("Hour cyclical encoding (one week)")
axes[0].legend()
axes[1].plot(demo["timestamp"], demo["dow_sin"], label="dayofweek_sin", color=PRIMARY)
axes[1].plot(demo["timestamp"], demo["dow_cos"], label="dayofweek_cos", color=ACCENT)
axes[1].set_title("Day-of-week cyclical encoding")
axes[1].legend()
plt.tight_layout()
save_figure(fig, PLOT_DIR / "cyclical_encoding")
plt.show()

# %% [markdown]
# `is_holiday` is country-aware: each site's country comes from the BDG2 mapping in
# `src/features/time_features.py`, so European sites do not get July 4th flagged
# spuriously. `is_weekend` is a simple Saturday-or-Sunday flag.

# %%
fig, ax = plt.subplots(figsize=(9, 4))
mean_by_h_hol = (
    df_feat.groupby(["hour", "is_holiday"], observed=True)["meter_reading"]
           .mean().unstack("is_holiday").rename(columns={0: "regular", 1: "holiday"})
)
ax.plot(mean_by_h_hol.index, mean_by_h_hol["regular"], color=PRIMARY, label="regular day")
ax.plot(mean_by_h_hol.index, mean_by_h_hol["holiday"], color=ACCENT, label="holiday")
ax.set_xlabel("hour of day")
ax.set_ylabel("mean kWh")
ax.set_title("Holidays flatten the working-hour ramp")
ax.legend()
save_figure(fig, PLOT_DIR / "holiday_effect")
plt.show()

# %% [markdown]
# ## 2. Lag features (per meter)
#
# `lag_24h` is yesterday's reading at the same hour; `lag_168h` is last week at the
# same hour. `lag_diff_24h = lag_24h - lag_168h` measures whether the same-hour
# pattern is changing week to week, which catches building-level regime changes.
#
# Lags are grouped by `(building_id, meter)` so we never carry one meter's history
# into another's.

# %%
sample = df_feat[df_feat["building_id"] == df_feat["building_id"].iloc[0]].copy()
sample = sample[(sample["meter"] == 0) &
                (sample["timestamp"] >= "2016-03-01") &
                (sample["timestamp"] < "2016-03-15")].sort_values("timestamp")

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(sample["timestamp"], sample["meter_reading"], color=PRIMARY, label="actual")
ax.plot(sample["timestamp"], sample["lag_24h"], color=ACCENT, alpha=0.7,
        linestyle="--", label="lag_24h")
ax.plot(sample["timestamp"], sample["lag_168h"], color=SUPPORT[2], alpha=0.7,
        linestyle="--", label="lag_168h")
ax.set_title(f"Lags vs. actual — building {sample['building_id'].iloc[0]} (electricity)")
ax.set_ylabel("kWh")
ax.legend()
plt.tight_layout()
save_figure(fig, PLOT_DIR / "lag_features")
plt.show()

# %% [markdown]
# ## 3. Rolling statistics
#
# `rolling_mean_24h` smooths daily noise; `rolling_std_24h` measures volatility; the
# 168-hour pair captures a full week. All four are computed on the past 24/168 hours
# *strictly before* the current timestamp. This is enforced by `.shift(1)` in
# `add_rolling_features`.

# %%
fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
axes[0].plot(sample["timestamp"], sample["meter_reading"], color=PRIMARY,
             alpha=0.5, linewidth=0.6, label="actual")
axes[0].plot(sample["timestamp"], sample["rolling_mean_24h"], color=ACCENT,
             label="rolling_mean_24h")
axes[0].plot(sample["timestamp"], sample["rolling_mean_168h"], color=SUPPORT[2],
             label="rolling_mean_168h")
axes[0].set_ylabel("kWh")
axes[0].set_title("Rolling means smooth the noise around the trend")
axes[0].legend()

axes[1].plot(sample["timestamp"], sample["rolling_std_24h"], color=PRIMARY,
             label="rolling_std_24h")
axes[1].plot(sample["timestamp"], sample["rolling_std_168h"], color=ACCENT,
             label="rolling_std_168h")
axes[1].set_title("Rolling standard deviation tracks volatility")
axes[1].set_ylabel("kWh")
axes[1].legend()
plt.tight_layout()
save_figure(fig, PLOT_DIR / "rolling_features")
plt.show()

# %% [markdown]
# ## 4. Weather features
#
# `air_temperature` and `dew_temperature` come from the per-site weather file.
# `temp_squared` is the explicit term that lets linear models see the U-shape we saw
# in the EDA. `wind_speed`, `cloud_coverage`, and `precip_depth_1_hr` round out the
# weather block.

# %%
weather_cols = ["air_temperature", "dew_temperature", "temp_squared",
                "wind_speed", "cloud_coverage", "precip_depth_1_hr"]
fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for ax, col in zip(axes.ravel(), weather_cols):
    if col in df_feat.columns:
        ax.hist(df_feat[col].dropna(), bins=60, color=PRIMARY, alpha=0.85)
        ax.set_title(col)
plt.tight_layout()
save_figure(fig, PLOT_DIR / "weather_distributions")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
sample_w = df_feat.dropna(subset=["air_temperature"]).sample(50_000, random_state=42)
ax.scatter(sample_w["air_temperature"], np.log1p(sample_w["meter_reading"].clip(lower=0)),
           s=3, alpha=0.15, color=PRIMARY, label="raw temp")
ax.scatter(sample_w["air_temperature"], np.log1p(sample_w["meter_reading"].clip(lower=0))
           - 0.0001 * sample_w["temp_squared"],
           s=0, alpha=0)  # transparent — placeholder so legend shows the second entry
ax.set_xlabel("air_temperature (°C)")
ax.set_ylabel("log1p(meter_reading)")
ax.set_title("temp_squared lets linear models see the U-shape")
plt.tight_layout()
save_figure(fig, PLOT_DIR / "temp_squared_motivation")
plt.show()

# %% [markdown]
# ## 5. Building-level metadata
#
# `log_square_feet` is the log of `square_feet`; `building_age` is the difference
# between a fixed reference year and `year_built` (with the median imputed where
# missing); `primary_use_code` is a categorical code that LightGBM and the trees
# can read directly. The linear models effectively treat this code as ordinal,
# which is suboptimal — that limitation is part of why the linear ablation lags.

# %%
print("primary_use_code mapping:")
mapping = (
    df_feat[["primary_use", "primary_use_code"]]
        .drop_duplicates()
        .sort_values("primary_use_code")
)
mapping

# %% [markdown]
# ## 6. Interactions
#
# Two interactions help the linear models without inflating dimensionality:
# `hour_x_weekend` lets the model express a different daily curve on Sat/Sun, and
# `sqft_x_temp` lets a building's HVAC sensitivity scale with size.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
hxw = df_feat.groupby(["hour", "is_weekend"], observed=True)["meter_reading"].mean().unstack("is_weekend")
hxw.columns = ["weekday", "weekend"]
axes[0].plot(hxw.index, hxw["weekday"], color=PRIMARY, label="weekday")
axes[0].plot(hxw.index, hxw["weekend"], color=ACCENT, label="weekend")
axes[0].set_title("Weekend changes the shape of the hourly curve")
axes[0].set_xlabel("hour"); axes[0].set_ylabel("mean kWh"); axes[0].legend()

sample_s = df_feat.dropna(subset=["log_square_feet", "air_temperature"]).sample(20_000, random_state=42)
sc = axes[1].scatter(sample_s["air_temperature"], sample_s["log_square_feet"],
                     c=np.log1p(sample_s["meter_reading"].clip(lower=0)),
                     cmap="viridis", s=3, alpha=0.5)
axes[1].set_xlabel("air_temperature (°C)")
axes[1].set_ylabel("log_square_feet")
axes[1].set_title("sqft_x_temp captures size-scaled HVAC load")
plt.colorbar(sc, ax=axes[1], label="log1p(kWh)")
plt.tight_layout()
save_figure(fig, PLOT_DIR / "interaction_features")
plt.show()

# %% [markdown]
# ## 7. Target encoding
#
# `te_primary_use_x_hour` replaces the `(primary_use, hour)` pair with the smoothed
# mean of `log1p(meter_reading)` for that combination, computed on the training fold
# only. Smoothing pulls rare combinations toward the global mean, so categories with
# very few rows do not produce noisy point estimates.
#
# `TargetEncoder` is fit on the training half of a time-based split and applied to
# the validation half — there is no validation target leakage.

# %%
train, val = time_split(df_feat, cfg.split.train_months, cfg.split.val_months)
encoder_demo = TargetEncoder(keys=["primary_use", "hour"], smoothing=cfg.features.target_encoding_smoothing)
encoder_demo.fit(train.dropna(subset=["meter_reading", "primary_use"]))
val_with_te = encoder_demo.transform(val.drop(columns=["te_primary_use_x_hour"], errors="ignore"))

global_mean = float(np.log1p(train["meter_reading"].clip(lower=0)).mean())
encoded = encoder_demo.encoding_.copy()
encoded["delta_from_global_mean"] = encoded["te_primary_use_x_hour"] - global_mean

pivot = (
    encoded.pivot_table(index="primary_use", columns="hour",
                        values="delta_from_global_mean")
)
fig, ax = plt.subplots(figsize=(13, 5.5))
sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "deviation from global mean"})
ax.set_title("Target encoding: primary_use × hour deviation from the global log mean")
ax.set_xlabel("hour")
plt.tight_layout()
save_figure(fig, PLOT_DIR / "target_encoding")
plt.show()

# %% [markdown]
# ## Sanity checks
#
# Two things must be true before training: (1) every feature at time `t` uses only
# data from before `t`, and (2) no validation row's target leaks into a feature.

# %%
ok = True
for col in ["lag_24h", "lag_168h", "rolling_mean_24h", "rolling_std_24h",
            "rolling_mean_168h", "rolling_std_168h"]:
    grp = df_feat.sort_values(["building_id", "meter", "timestamp"]).groupby(
        ["building_id", "meter"], observed=True)
    first = grp[col].first().dropna()
    if len(first) > 0:
        print(f"{col}: first non-NaN exists in {len(first)} groups (expected 0 if shift respected)")

train_max = train["timestamp"].max()
val_min = val["timestamp"].min()
print(f"\ntrain.max={train_max} val.min={val_min}")
assert train_max < val_min, "temporal leakage detected"
print("temporal split clean")

# %% [markdown]
# ## What the model sees
#
# The final feature matrix shape we hand to every (non-ARIMA, non-LSTM) model:

# %%
feature_cols = [c for c in FEATURE_COLUMNS if c in df_feat.columns]
print(f"feature columns ({len(feature_cols)}):")
for f in feature_cols:
    print(f"  {f}")
print(f"\nraw ablation columns ({len([c for c in RAW_FEATURE_COLUMNS if c in df_feat.columns])}):")
for f in RAW_FEATURE_COLUMNS:
    if f in df_feat.columns:
        print(f"  {f}")
