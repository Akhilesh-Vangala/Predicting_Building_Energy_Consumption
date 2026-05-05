from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.viz.style import PRIMARY, ACCENT, SUPPORT

METER_LABELS = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}


def plot_target_distribution(df: pd.DataFrame, target_col: str = "meter_reading") -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    y = df[target_col].dropna()
    y_pos = y[y > 0]

    axes[0].hist(y, bins=80, color=PRIMARY, alpha=0.85)
    axes[0].set_yscale("log")
    axes[0].set_title("Meter readings (raw scale, log y)")
    axes[0].set_xlabel("kWh")
    axes[0].set_ylabel("count (log)")

    axes[1].hist(np.log1p(y_pos), bins=80, color=ACCENT, alpha=0.85)
    axes[1].set_title("Meter readings (log1p)")
    axes[1].set_xlabel("log1p(kWh)")
    axes[1].set_ylabel("count")
    fig.suptitle("Right-skewed target — log1p stabilises the variance", y=1.02)
    fig.tight_layout()
    return fig


def plot_missingness(df: pd.DataFrame, top_n: int = 25) -> plt.Figure:
    miss = df.isna().mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(miss))))
    ax.barh(miss.index[::-1], (miss.values * 100)[::-1], color=PRIMARY)
    ax.set_xlabel("% missing")
    ax.set_title("Column-level missingness")
    fig.tight_layout()
    return fig


def plot_meter_breakdown(df: pd.DataFrame, target_col: str = "meter_reading") -> plt.Figure:
    counts = df["meter"].value_counts().sort_index()
    means = df.groupby("meter")[target_col].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(
        [METER_LABELS.get(int(i), str(i)) for i in counts.index],
        counts.values, color=SUPPORT[:len(counts)],
    )
    axes[0].set_title("Row count by meter type")
    axes[0].set_ylabel("rows")

    axes[1].bar(
        [METER_LABELS.get(int(i), str(i)) for i in means.index],
        means.values, color=SUPPORT[:len(means)],
    )
    axes[1].set_title("Mean consumption by meter type (kWh)")
    axes[1].set_ylabel("mean kWh")
    fig.tight_layout()
    return fig


def plot_primary_use_breakdown(df: pd.DataFrame, target_col: str = "meter_reading") -> plt.Figure:
    if "primary_use" not in df.columns:
        return plt.figure()
    counts = df["primary_use"].value_counts()
    means = df.groupby("primary_use", observed=True)[target_col].mean().reindex(counts.index)

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.25 * len(counts))))
    axes[0].barh(counts.index[::-1], counts.values[::-1], color=PRIMARY)
    axes[0].set_title("Row count by primary use")
    axes[0].set_xlabel("rows")
    axes[1].barh(means.index[::-1], means.values[::-1], color=ACCENT)
    axes[1].set_title("Mean consumption by primary use")
    axes[1].set_xlabel("mean kWh")
    fig.tight_layout()
    return fig


def plot_hourly_profile(df: pd.DataFrame, target_col: str = "meter_reading",
                        timestamp_col: str = "timestamp") -> plt.Figure:
    work = df[[timestamp_col, target_col, "meter"]].copy()
    work["hour"] = pd.DatetimeIndex(work[timestamp_col]).hour
    profile = work.groupby(["meter", "hour"])[target_col].mean().unstack("meter")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, col in enumerate(profile.columns):
        ax.plot(profile.index, profile[col], label=METER_LABELS.get(int(col), str(col)),
                color=SUPPORT[i % len(SUPPORT)], linewidth=2)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("mean kWh")
    ax.set_title("Hour-of-day consumption profile by meter")
    ax.legend(title="meter")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    return fig


def plot_weekly_profile(df: pd.DataFrame, target_col: str = "meter_reading",
                        timestamp_col: str = "timestamp") -> plt.Figure:
    work = df[[timestamp_col, target_col, "primary_use"]].copy()
    ts = pd.DatetimeIndex(work[timestamp_col])
    work["dayofweek"] = ts.dayofweek
    work["hour"] = ts.hour
    work["bucket"] = work["dayofweek"] * 24 + work["hour"]

    if "primary_use" in work.columns:
        top_uses = work["primary_use"].value_counts().head(4).index.tolist()
        sub = work[work["primary_use"].isin(top_uses)]
        profile = sub.groupby(["primary_use", "bucket"], observed=True)[target_col].mean().unstack("primary_use")
    else:
        profile = work.groupby("bucket")[target_col].mean().to_frame("all")

    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, col in enumerate(profile.columns):
        ax.plot(profile.index, profile[col], label=col, color=SUPPORT[i % len(SUPPORT)], linewidth=1.6)
    for d in range(1, 7):
        ax.axvline(d * 24, color="#aaa", linestyle="--", linewidth=0.5)
    ax.set_xticks([12 + d * 24 for d in range(7)])
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_ylabel("mean kWh")
    ax.set_title("Weekly hour-by-hour profile, top primary uses")
    ax.legend(title="primary use", loc="upper right")
    fig.tight_layout()
    return fig


def plot_temperature_vs_consumption(df: pd.DataFrame,
                                    target_col: str = "meter_reading") -> plt.Figure:
    if "air_temperature" not in df.columns:
        return plt.figure()
    sample = df.sample(min(len(df), 50_000), random_state=42)
    work = sample[["air_temperature", target_col, "meter"]].dropna()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(sorted(work["meter"].unique())):
        sub = work[work["meter"] == m]
        ax.scatter(
            sub["air_temperature"],
            np.log1p(sub[target_col].clip(lower=0)),
            s=4, alpha=0.18, color=SUPPORT[i % len(SUPPORT)],
            label=METER_LABELS.get(int(m), str(m)),
        )
    ax.set_xlabel("air temperature (\u00b0C)")
    ax.set_ylabel("log1p(meter_reading)")
    ax.set_title("Temperature vs. consumption \u2014 U-shape across meters")
    ax.legend(title="meter")
    fig.tight_layout()
    return fig


def plot_correlation_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> plt.Figure:
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(min(13, 0.5 * len(columns) + 4),
                                    min(11, 0.5 * len(columns) + 3)))
    sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                ax=ax, square=True, cbar_kws={"shrink": 0.7})
    ax.set_title("Pearson correlation \u2014 numeric features")
    fig.tight_layout()
    return fig


def plot_per_site_volume(df: pd.DataFrame) -> plt.Figure:
    counts = df.groupby("site_id").size()
    bldgs = df.groupby("site_id")["building_id"].nunique()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(counts.index, counts.values, color=PRIMARY)
    axes[0].set_title("Rows per site")
    axes[0].set_xlabel("site_id")
    axes[0].set_ylabel("rows")
    axes[1].bar(bldgs.index, bldgs.values, color=ACCENT)
    axes[1].set_title("Buildings per site")
    axes[1].set_xlabel("site_id")
    axes[1].set_ylabel("buildings")
    fig.tight_layout()
    return fig
