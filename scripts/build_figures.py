from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import load_config
from src.utils import setup_logging
from src.viz.style import ACCENT, PRIMARY, SUPPORT, save_figure, set_style

logger = logging.getLogger(__name__)

METER_NAMES = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}

MODEL_ORDER = [
    "ols", "ridge", "lasso", "elasticnet",
    "decision_tree", "random_forest", "lightgbm",
    "arima", "lstm", "mlp",
]

FAMILY_COLOR = {
    "linear": SUPPORT[0],
    "trees": SUPPORT[1],
    "time_series": SUPPORT[2],
    "neural": SUPPORT[3],
    "baseline": "#aaaaaa",
}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    save_figure(fig, out_dir / name, formats=("pdf", "png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Results figures — read from saved CSV / JSON artifacts
# ---------------------------------------------------------------------------

def fig_model_rmse_bar(eng_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(eng_csv).sort_values("rmse")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(df) + 1.0)))
    colors = [FAMILY_COLOR.get(f, PRIMARY) for f in df["family"]]
    bars = ax.barh(df["model"], df["rmse"], color=colors)
    ax.set_xlabel("RMSE (kWh)")
    ax.set_title("Model comparison — validation RMSE")
    from matplotlib.patches import Patch
    seen = {}
    for fam, color in zip(df["family"], colors):
        if fam not in seen:
            seen[fam] = color
    handles = [Patch(color=c, label=l) for l, c in seen.items()]
    ax.legend(handles=handles, loc="lower right", title="family")
    for bar, val in zip(bars, df["rmse"]):
        ax.text(val * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    _save(fig, out_dir, "model_rmse_bar")
    logger.info("model_rmse_bar done")


def fig_ablation_delta(eng_csv: Path, raw_csv: Path, out_dir: Path) -> None:
    eng = pd.read_csv(eng_csv).set_index("model")
    raw = pd.read_csv(raw_csv).set_index("model")
    common = [m for m in MODEL_ORDER
              if m in eng.index and m in raw.index
              and eng.loc[m, "family"] not in ("baseline",)]
    if not common:
        logger.warning("ablation_delta: no common models")
        return
    delta = (eng.loc[common, "rmse"] - raw.loc[common, "rmse"]) / raw.loc[common, "rmse"] * 100
    delta = delta.sort_values()
    colors = [ACCENT if v < 0 else PRIMARY for v in delta.values]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(delta) + 1.0)))
    ax.barh(delta.index, delta.values, color=colors)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("RMSE change, engineered vs. raw (%)")
    ax.set_title("Feature engineering ablation")
    fig.tight_layout()
    _save(fig, out_dir, "ablation_delta")
    logger.info("ablation_delta done")


def fig_rmse_by_meter_type(fail_meter_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(fail_meter_csv, index_col=0)
    models = [m for m in MODEL_ORDER if m in df.columns]
    if not models:
        logger.warning("rmse_by_meter_type: no models found in CSV")
        return
    x = np.arange(len(df.index))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(7, len(models) * 0.85), 5))
    for i, m in enumerate(models):
        ax.bar(x + i * width, df[m], width=width, label=m,
               color=SUPPORT[i % len(SUPPORT)])
    ax.set_xticks(x + width * len(models) / 2)
    ax.set_xticklabels(df.index, rotation=15)
    ax.set_ylabel("RMSE (kWh)")
    ax.set_title("RMSE by meter type")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    _save(fig, out_dir, "rmse_by_meter_type")
    logger.info("rmse_by_meter_type done")


def fig_failure_heatmap(fail_use_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(fail_use_csv, index_col=0)
    models = [m for m in MODEL_ORDER if m in df.columns]
    if not models:
        logger.warning("failure_heatmap: no models found in CSV")
        return
    sub = df[models].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    fig, ax = plt.subplots(figsize=(max(7, len(models) * 0.85), max(4, len(sub) * 0.45)))
    annot = sub.map(lambda v: f"{v:,.0f}" if pd.notna(v) else "")
    sns.heatmap(np.log1p(sub.fillna(0)), ax=ax, cmap="YlOrRd",
                annot=annot, fmt="", linewidths=0.3,
                cbar_kws={"label": "log1p(RMSE)"})
    ax.set_title("RMSE by primary use x model (values in kWh)")
    ax.set_xlabel("")
    fig.tight_layout()
    _save(fig, out_dir, "failure_heatmap_use_x_model")
    logger.info("failure_heatmap_use_x_model done")


def fig_lgbm_feature_importance(eng_json: Path, out_dir: Path, top_n: int = 15) -> None:
    d = json.loads(eng_json.read_text())
    imp = d["models"].get("lightgbm", {}).get("feature_importance") or {}
    if not imp:
        logger.warning("lgbm_feature_importance: no data")
        return
    s = pd.Series(imp).sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(s))))
    ax.barh(s.index, s.values, color=PRIMARY)
    ax.set_xlabel("importance (gain)")
    ax.set_title(f"LightGBM feature importance — top {top_n}")
    fig.tight_layout()
    _save(fig, out_dir, "lightgbm_feature_importance")
    logger.info("lightgbm_feature_importance done")


def fig_cluster_elbow(elbow_csv: Path, clustering_json: Path, out_dir: Path) -> None:
    df = pd.read_csv(elbow_csv)
    d = json.loads(clustering_json.read_text())
    k_star = d["k_star"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["k"], df["inertia"], marker="o", color=PRIMARY, linewidth=2)
    ax.axvline(k_star, color=ACCENT, linestyle="--", linewidth=1.5, label=f"k*={k_star}")
    ax.set_xlabel("k")
    ax.set_ylabel("inertia")
    ax.set_title("K-means elbow — consumption profile clusters")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "cluster_elbow")
    logger.info("cluster_elbow done")


def fig_per_cluster_vs_global(clustering_json: Path, out_dir: Path) -> None:
    d = json.loads(clustering_json.read_text())
    cm = pd.DataFrame(d["per_cluster_metrics"])
    global_rmse = d["global_lightgbm"]["rmse"]
    overall_rmse = d["per_cluster_overall"]["rmse"]
    fig, ax = plt.subplots(figsize=(max(5, len(cm) * 0.9), 4.5))
    ax.bar(cm["cluster"].astype(str), cm["rmse"], color=PRIMARY, label="per-cluster LightGBM")
    ax.axhline(global_rmse, color=ACCENT, linestyle="--", linewidth=1.5,
               label=f"global LightGBM RMSE={global_rmse:,.0f}")
    ax.axhline(overall_rmse, color=SUPPORT[2], linestyle=":", linewidth=1.5,
               label=f"overall cluster RMSE={overall_rmse:,.0f}")
    ax.set_xlabel("cluster")
    ax.set_ylabel("RMSE (kWh)")
    ax.set_title("Per-cluster LightGBM vs. global LightGBM")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "per_cluster_vs_global")
    logger.info("per_cluster_vs_global done")


# ---------------------------------------------------------------------------
# Data-dependent figures — require prepare_data() to be called first
# ---------------------------------------------------------------------------

def fig_eda_target_distribution(train_df: pd.DataFrame, out_dir: Path) -> None:
    from src.viz.eda_plots import plot_target_distribution
    fig = plot_target_distribution(train_df)
    _save(fig, out_dir, "eda_target_distribution")
    logger.info("eda_target_distribution done")


def fig_eda_hourly_profile_by_use(train_df: pd.DataFrame, out_dir: Path) -> None:
    if "primary_use" not in train_df.columns:
        return
    top_uses = train_df["primary_use"].value_counts().head(5).index.tolist()
    sub = train_df[train_df["primary_use"].isin(top_uses)].copy()
    sub["hour"] = pd.DatetimeIndex(sub["timestamp"]).hour
    profile = (sub.groupby(["primary_use", "hour"], observed=True)["meter_reading"]
               .mean().unstack("primary_use"))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, col in enumerate(profile.columns):
        ax.plot(profile.index, profile[col], label=str(col),
                color=SUPPORT[i % len(SUPPORT)], linewidth=2)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("mean kWh")
    ax.set_title("Hour-of-day consumption by primary use (top 5)")
    ax.legend(title="primary use", fontsize=8)
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    _save(fig, out_dir, "eda_hourly_profile_by_use")
    logger.info("eda_hourly_profile_by_use done")


def fig_eda_meter_type_distributions(train_df: pd.DataFrame, out_dir: Path) -> None:
    from src.viz.eda_plots import plot_meter_breakdown
    fig = plot_meter_breakdown(train_df)
    _save(fig, out_dir, "eda_meter_type_distributions")
    logger.info("eda_meter_type_distributions done")


def fig_eda_weather_correlation(train_df: pd.DataFrame, out_dir: Path) -> None:
    cols = [c for c in [
        "meter_reading", "air_temperature", "dew_temperature", "wind_speed",
        "cloud_coverage", "temp_squared", "hour_sin", "hour_cos",
        "is_weekend", "lag_24h", "rolling_mean_24h", "log_square_feet",
    ] if c in train_df.columns]
    sample = train_df[cols].sample(min(len(train_df), 100_000), random_state=42)
    corr = sample.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(9, 0.5 * len(cols) + 3),
                                    max(8, 0.5 * len(cols) + 2)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, square=True,
                cbar_kws={"shrink": 0.7}, annot_kws={"size": 7})
    ax.set_title("Pearson correlation — numeric features")
    fig.tight_layout()
    _save(fig, out_dir, "eda_weather_correlation")
    logger.info("eda_weather_correlation done")


def fig_eda_missingness(train_df: pd.DataFrame, out_dir: Path) -> None:
    from src.viz.eda_plots import plot_missingness
    fig = plot_missingness(train_df)
    _save(fig, out_dir, "eda_missingness")
    logger.info("eda_missingness done")


def fig_eda_site_coverage(train_df: pd.DataFrame, out_dir: Path) -> None:
    from src.viz.eda_plots import plot_per_site_volume
    fig = plot_per_site_volume(train_df)
    _save(fig, out_dir, "eda_site_coverage")
    logger.info("eda_site_coverage done")


def fig_residuals_by_hour(val_df: pd.DataFrame, pred_dir: Path,
                          eng_json: Path, out_dir: Path) -> None:
    d = json.loads(eng_json.read_text())
    models_rmse = {m: d["models"][m]["metrics"]["rmse"] for m in d["models"]}
    best_model = min(models_rmse, key=models_rmse.get)
    pred_file = pred_dir / f"{best_model}_log_preds.npy"
    if not pred_file.exists():
        logger.warning("residuals_by_hour: %s not found", pred_file)
        return
    preds_log = np.load(pred_file)
    y_true_raw = np.load(pred_dir / "y_true_raw.npy")
    if len(preds_log) != len(y_true_raw):
        logger.warning("residuals_by_hour: length mismatch %d vs %d",
                       len(preds_log), len(y_true_raw))
        return
    preds_real = np.expm1(np.clip(preds_log, 0.0, 14.0))
    residuals = preds_real - y_true_raw

    hour = pd.DatetimeIndex(val_df["timestamp"]).hour
    by_hour = (pd.DataFrame({"hour": hour, "residual": residuals})
               .groupby("hour")["residual"]
               .agg(["mean", "std"])
               .reset_index())

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(by_hour["hour"],
                    by_hour["mean"] - by_hour["std"],
                    by_hour["mean"] + by_hour["std"],
                    alpha=0.2, color=PRIMARY)
    ax.plot(by_hour["hour"], by_hour["mean"], color=PRIMARY, linewidth=2,
            label="mean residual")
    ax.axhline(0, color=ACCENT, linewidth=1)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("residual (kWh)")
    ax.set_title(f"Residuals by hour of day — {best_model}")
    ax.set_xticks(range(0, 24, 2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "residuals_by_hour")
    logger.info("residuals_by_hour done (model=%s)", best_model)


def fig_cluster_centroids(train_df: pd.DataFrame, clustering_json: Path,
                           out_dir: Path) -> None:
    from src.clustering.kmeans_profiles import build_consumption_profiles, fit_kmeans

    d = json.loads(clustering_json.read_text())
    k_star = d["k_star"]
    profiles = build_consumption_profiles(train_df)
    km = fit_kmeans(profiles, k_star, random_state=42)
    centroids = pd.DataFrame(
        km.cluster_centers_,
        columns=[f"h{h:02d}" for h in range(24)],
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (_, row) in enumerate(centroids.iterrows()):
        ax.plot(range(24), row.values, label=f"cluster {i}",
                color=SUPPORT[i % len(SUPPORT)], linewidth=2)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("normalised consumption")
    ax.set_title("K-means cluster centroids — 24-hour consumption profile")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(title="cluster")
    fig.tight_layout()
    _save(fig, out_dir, "cluster_centroids")
    logger.info("cluster_centroids done (k=%d)", k_star)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-eda", action="store_true",
                        help="skip figures that require loading the full dataset")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_style()

    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = cfg.paths.metrics
    tables_dir = cfg.paths.tables
    eng_csv = tables_dir / "models_engineered.csv"
    raw_csv = tables_dir / "models_raw.csv"
    eng_json = metrics_dir / "models_engineered.json"
    clustering_json = metrics_dir / "clustering.json"
    elbow_csv = tables_dir / "cluster_elbow.csv"
    fail_meter_csv = tables_dir / "failure_by_meter_engineered.csv"
    fail_use_csv = tables_dir / "failure_by_primary_use_engineered.csv"
    pred_dir = metrics_dir / "predictions_engineered"

    # --- results figures (no raw data) ---
    if eng_csv.exists():
        fig_model_rmse_bar(eng_csv, out_dir)
    if eng_csv.exists() and raw_csv.exists():
        fig_ablation_delta(eng_csv, raw_csv, out_dir)
    if fail_meter_csv.exists():
        fig_rmse_by_meter_type(fail_meter_csv, out_dir)
    if fail_use_csv.exists():
        fig_failure_heatmap(fail_use_csv, out_dir)
    if eng_json.exists():
        fig_lgbm_feature_importance(eng_json, out_dir)
    if elbow_csv.exists() and clustering_json.exists():
        fig_cluster_elbow(elbow_csv, clustering_json, out_dir)
    if clustering_json.exists():
        fig_per_cluster_vs_global(clustering_json, out_dir)

    if args.skip_eda:
        logger.info("--skip-eda: skipping data-dependent figures")
    else:
        logger.info("Loading dataset for EDA and residual figures (~2 min)...")
        from src.pipeline import prepare_data
        prep = prepare_data(cfg, feature_set="engineered", target_log=True)
        train_df = prep.train_full
        val_df = prep.val_full

        fig_eda_target_distribution(train_df, out_dir)
        fig_eda_hourly_profile_by_use(train_df, out_dir)
        fig_eda_meter_type_distributions(train_df, out_dir)
        fig_eda_weather_correlation(train_df, out_dir)
        fig_eda_missingness(train_df, out_dir)
        fig_eda_site_coverage(train_df, out_dir)
        if pred_dir.exists() and eng_json.exists():
            fig_residuals_by_hour(val_df, pred_dir, eng_json, out_dir)
        if clustering_json.exists():
            fig_cluster_centroids(train_df, clustering_json, out_dir)

    files = sorted(out_dir.glob("*"))
    logger.info("results/figures/ has %d files", len(files))


if __name__ == "__main__":
    main()
