from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src.clustering.kmeans_profiles import (
    assign_clusters, build_consumption_profiles, elbow_curve, fit_kmeans,
)
from src.config import load_config
from src.evaluation.metrics import score_predictions
from src.models import LightGBMModel
from src.pipeline import evaluate_predictions, prepare_data
from src.utils import save_json, set_seed, setup_logging, timer

logger = logging.getLogger(__name__)


def _pick_k_elbow(curve: pd.DataFrame) -> int:
    ks = curve["k"].to_numpy(dtype=float)
    inertias = curve["inertia"].to_numpy(dtype=float)
    if len(ks) < 3:
        return int(ks[0])
    p1 = np.array([ks[0], inertias[0]])
    p2 = np.array([ks[-1], inertias[-1]])
    line_vec = p2 - p1
    line_norm = line_vec / np.linalg.norm(line_vec)
    distances = []
    for x, y in zip(ks, inertias):
        v = np.array([x, y]) - p1
        proj = v - (v @ line_norm) * line_norm
        distances.append(np.linalg.norm(proj))
    return int(ks[int(np.argmax(distances))])


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.random_state)

    with timer("prepare_data"):
        prep = prepare_data(cfg, feature_set="engineered", target_log=True)

    profiles = build_consumption_profiles(prep.train_full)
    logger.info("Built consumption profiles for %d meters", len(profiles))

    curve = elbow_curve(profiles, cfg.clustering.k_range, random_state=cfg.clustering.random_state)
    k_star = _pick_k_elbow(curve)
    logger.info("Elbow k = %d", k_star)

    km = fit_kmeans(profiles, k_star, random_state=cfg.clustering.random_state)
    cluster_labels = assign_clusters(km, profiles).reset_index()

    train = prep.train_full.merge(cluster_labels, on=["building_id", "meter"], how="left")
    val = prep.val_full.merge(cluster_labels, on=["building_id", "meter"], how="left")

    # Buildings that appear only in val (zero training rows after cleaning) get NaN
    # cluster. Assign them to the nearest centroid using their val hourly profile.
    missing_mask = val["cluster"].isna()
    if missing_mask.any():
        missing_keys = val.loc[missing_mask, ["building_id", "meter"]].drop_duplicates()
        logger.info(
            "Assigning %d val-only meters to nearest cluster via val profile",
            len(missing_keys),
        )
        val_profiles = build_consumption_profiles(
            val.loc[missing_mask, ["building_id", "meter", "timestamp", "meter_reading"]]
        )
        if len(val_profiles):
            assigned = assign_clusters(km, val_profiles).reset_index()
            val = val.merge(
                assigned.rename(columns={"cluster": "_cluster_fallback"}),
                on=["building_id", "meter"],
                how="left",
            )
            val["cluster"] = val["cluster"].fillna(val["_cluster_fallback"])
            val = val.drop(columns=["_cluster_fallback"])
            cluster_labels = pd.concat(
                [cluster_labels, assigned], ignore_index=True
            ).drop_duplicates(subset=["building_id", "meter"])

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    feature_cols = prep.feature_cols
    cluster_metrics: list[dict] = []
    cluster_preds_real = np.full(len(val), np.nan, dtype=np.float64)
    cluster_true_real = np.full(len(val), np.nan, dtype=np.float64)

    val_cluster_arr = val["cluster"].to_numpy()
    train_cluster_arr = train["cluster"].to_numpy()

    for c in sorted(pd.Series(train_cluster_arr).dropna().unique()):
        train_pos = np.where(train_cluster_arr == c)[0]
        val_pos = np.where(val_cluster_arr == c)[0]
        if len(train_pos) < 1000 or len(val_pos) == 0:
            logger.info("Skipping tiny cluster %s (train=%d val=%d)", c, len(train_pos), len(val_pos))
            continue
        train_c = train.iloc[train_pos]
        val_c = val.iloc[val_pos]
        model = LightGBMModel(**cfg.models.get("lightgbm", {}))
        Xt = train_c[feature_cols]; Xv = val_c[feature_cols]
        yt = np.log1p(np.clip(train_c["meter_reading"].to_numpy(), 0, None))
        yv = np.log1p(np.clip(val_c["meter_reading"].to_numpy(), 0, None))
        with timer(f"cluster_{int(c)}_lgbm"):
            model.fit(Xt, yt, eval_set=(Xv, yv))
        preds_log = model.predict(Xv)
        eval_payload = evaluate_predictions(val_c, preds_log, yv, target_log=True)
        cluster_metrics.append({
            "cluster": int(c),
            "n_train": int(len(train_pos)),
            "n_val": int(len(val_pos)),
            **eval_payload["overall"],
        })
        cluster_preds_real[val_pos] = np.expm1(np.clip(preds_log, a_min=0.0, a_max=14.0))
        cluster_true_real[val_pos] = np.expm1(np.clip(yv, a_min=0.0, a_max=14.0))

    valid = ~np.isnan(cluster_preds_real) & ~np.isnan(cluster_true_real)
    overall_cluster = score_predictions(cluster_true_real[valid], cluster_preds_real[valid])

    feature_cols_global = prep.feature_cols
    global_lgbm = LightGBMModel(**cfg.models.get("lightgbm", {}))
    Xt = prep.train_full[feature_cols_global]
    Xv = prep.val_full[feature_cols_global]
    yt = prep.target_log_train
    yv = prep.target_log_val
    with timer("global_lgbm"):
        global_lgbm.fit(Xt, yt, eval_set=(Xv, yv))
    global_preds_log = global_lgbm.predict(Xv)
    global_eval = evaluate_predictions(prep.val_full, global_preds_log, yv, target_log=True)

    payload = {
        "elbow": curve.to_dict(orient="records"),
        "k_star": int(k_star),
        "per_cluster_metrics": cluster_metrics,
        "per_cluster_overall": overall_cluster,
        "global_lightgbm": global_eval["overall"],
    }
    save_json(payload, cfg.paths.metrics / "clustering.json")

    cluster_labels.to_csv(cfg.paths.tables / "cluster_labels.csv", index=False)
    pd.DataFrame(cluster_metrics).to_csv(cfg.paths.tables / "cluster_metrics.csv", index=False)
    curve.to_csv(cfg.paths.tables / "cluster_elbow.csv", index=False)
    logger.info("Clustering done: per-cluster RMSE=%.2f vs global=%.2f",
                overall_cluster["rmse"], global_eval["overall"]["rmse"])


if __name__ == "__main__":
    main()
