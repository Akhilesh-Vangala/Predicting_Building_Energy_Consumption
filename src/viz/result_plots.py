from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.viz.style import PRIMARY, ACCENT, SUPPORT


def plot_model_comparison(metrics_df: pd.DataFrame, metric: str = "rmse") -> plt.Figure:
    df = metrics_df.sort_values(metric)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(df))))
    colors = []
    for fam in df["family"]:
        idx = ["linear", "trees", "time_series", "neural"].index(fam) if fam in [
            "linear", "trees", "time_series", "neural"
        ] else 0
        colors.append(SUPPORT[idx % len(SUPPORT)])
    ax.barh(df["model"][::-1], df[metric][::-1], color=colors[::-1])
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Model comparison \u2014 {metric.upper()} on validation")
    fig.tight_layout()
    return fig


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                             title: str = "predicted vs. actual") -> plt.Figure:
    sample = min(len(y_true), 30_000)
    idx = np.random.default_rng(42).choice(len(y_true), size=sample, replace=False)
    yt = y_true[idx]; yp = y_pred[idx]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(np.log1p(np.clip(yt, 0, None)), np.log1p(np.clip(yp, 0, None)),
               s=3, alpha=0.18, color=PRIMARY)
    lo = min(np.log1p(yt.min()) if yt.min() > 0 else 0, np.log1p(yp.min()) if yp.min() > 0 else 0)
    hi = max(np.log1p(yt.max()), np.log1p(yp.max()))
    ax.plot([lo, hi], [lo, hi], color=ACCENT, linewidth=1)
    ax.set_xlabel("log1p(actual)")
    ax.set_ylabel("log1p(predicted)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   title: str = "residuals") -> plt.Figure:
    res = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(res, bins=80, color=PRIMARY, alpha=0.85)
    axes[0].set_title(f"{title} \u2014 distribution")
    axes[0].set_xlabel("predicted - actual")
    axes[1].scatter(np.log1p(np.clip(y_true, 0, None)), res, s=3, alpha=0.15, color=PRIMARY)
    axes[1].axhline(0, color=ACCENT, linewidth=1)
    axes[1].set_title(f"{title} \u2014 residual vs actual")
    axes[1].set_xlabel("log1p(actual)")
    axes[1].set_ylabel("residual")
    fig.tight_layout()
    return fig


def plot_feature_importance(importance: dict[str, float], top_n: int = 25,
                            title: str = "feature importance") -> plt.Figure:
    series = pd.Series(importance).sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(series))))
    ax.barh(series.index, series.values, color=PRIMARY)
    ax.set_xlabel("importance (gain or |coef|)")
    ax.set_title(title)
    fig.tight_layout()
    return fig
