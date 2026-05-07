"""Train the LSTM on all meters using chunked streaming.

The stock fit_predict_sequences materialises one tensor for all windows before
training begins.  For 970 meters at stride=6 that is ~17 GB — too large for
a workstation.  This script keeps memory bounded by processing one chunk of
`METER_CHUNK` meters at a time inside each epoch:

  per-chunk memory ≈ METER_CHUNK * (train_len / stride) * lookback * n_feat * 4 B
                   ≈  20          * 1000                 * 168       * 27     * 4
                   ≈  360 MB

All 970 meters are visited every epoch, so the model sees the full dataset.
Validation predictions are collected in the same chunked manner.

Usage (from the repo root, venv active):
    python -m scripts.run_lstm_full --config configs/default.yaml

Outputs
    results/metrics/predictions_engineered/lstm_full_log_preds.npy
    results/metrics/models_engineered.json   (updated with lstm_full row)
    results/tables/models_engineered.csv     (updated)
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd

from src.config import load_config
from src.pipeline import evaluate_predictions, prepare_data
from src.utils import load_json, save_json, set_seed, setup_logging

logger = logging.getLogger(__name__)

# Tunable constants — keep METER_CHUNK small enough to fit in RAM.
# 20 meters × ~1,000 strided windows × 168 × 27 features × 4 bytes ≈ 360 MB.
METER_CHUNK = 20
LOOKBACK = 168
HIDDEN = 96
LAYERS = 1
DROPOUT = 0.0        # dropout only helps when LAYERS > 1
BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 12
PATIENCE = 3
STRIDE = 6           # sample every 6th window; same as the 60-meter run


def _iter_meter_chunks(
    df: pd.DataFrame,
    all_keys: pd.DataFrame,
    chunk_size: int,
    keys: list[str],
    timestamp_col: str,
):
    """Yield consecutive slices of meters from df."""
    for start in range(0, len(all_keys), chunk_size):
        chunk_keys = all_keys.iloc[start:start + chunk_size]
        yield df.merge(chunk_keys, on=keys, how="inner")


def _build_windows_fast(
    f: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    lookback: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised window builder using numpy sliding_window_view.

    Returns X (n_windows, lookback, n_feat), y (n_windows,), orig_idx (n_windows,).
    All windows start at position `lookback` and are spaced `stride` apart.
    """
    n = len(f)
    if n <= lookback:
        return (
            np.empty((0, lookback, f.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    # sliding_window_view shape: (n, lookback, n_feat)
    windows = np.lib.stride_tricks.sliding_window_view(f, (lookback, f.shape[1]))
    # windows[i] is the window ending at row i+lookback-1; target is row i+lookback.
    # Valid target positions: lookback .. n-1  (step=stride)
    positions = np.arange(lookback, n, stride)   # indices into the original array
    # windows indexed by (position - lookback)
    X = windows[positions - lookback].squeeze(1).astype(np.float32)  # (nw, lookback, nf)
    return X, y[positions], idx[positions]


def _build_windows(
    chunk_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    keys: list[str],
    timestamp_col: str,
    lookback: int,
    stride: int,
    target_log: bool,
    train_cutoff: pd.Timestamp | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build (X, y, original_idx) windows from one meter chunk."""
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_idx: list[np.ndarray] = []

    for _, grp in chunk_df.groupby(keys, observed=True, sort=False):
        grp = grp.sort_values(timestamp_col)
        f = grp[list(feature_cols)].to_numpy(dtype=np.float32)
        y_raw = grp[target_col].to_numpy(dtype=np.float64)
        idx = grp.index.to_numpy(dtype=np.int64)

        if len(grp) <= lookback:
            continue

        if target_log:
            y = np.log1p(np.clip(y_raw, 0, None)).astype(np.float32)
        else:
            y = y_raw.astype(np.float32)

        if train_cutoff is not None:
            # Training mode: only windows whose target timestamp > train_cutoff.
            ts = pd.to_datetime(grp[timestamp_col].to_numpy())
            cutoff_pos = np.searchsorted(ts, pd.Timestamp(train_cutoff), side="right")
            positions = np.arange(cutoff_pos, len(grp), stride)
            positions = positions[positions >= lookback]
            if len(positions) == 0:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(f, (lookback, f.shape[1]))
            X_m = windows[positions - lookback].squeeze(1).astype(np.float32)
            y_m = y[positions]
            idx_m = idx[positions]
        else:
            X_m, y_m, idx_m = _build_windows_fast(f, y, idx, lookback, stride)

        if len(X_m) == 0:
            continue
        all_X.append(X_m)
        all_y.append(y_m)
        all_idx.append(idx_m)

    if not all_X:
        n_feat = len(feature_cols)
        return (
            np.empty((0, lookback, n_feat), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            [],
        )
    return (
        np.concatenate(all_X, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_idx, axis=0).tolist(),
    )


def _build_val_windows(
    chunk_df: pd.DataFrame,
    full_df: pd.DataFrame,  # train + val merged for context
    feature_cols: list[str],
    target_col: str,
    keys: list[str],
    timestamp_col: str,
    lookback: int,
    train_cutoff: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build validation windows (need lookback from train side)."""
    chunk_keys = chunk_df[keys].drop_duplicates()
    chunk_full = full_df.merge(chunk_keys, on=keys, how="inner")

    all_X: list[np.ndarray] = []
    all_y: list[float] = []
    all_idx: list[int] = []

    for _, grp in chunk_full.groupby(keys, observed=True, sort=False):
        grp = grp.sort_values(timestamp_col).reset_index(drop=True)
        f = grp[list(feature_cols)].to_numpy(dtype=np.float32)
        y_raw = grp[target_col].to_numpy(dtype=np.float64)
        y = np.log1p(np.clip(y_raw, 0, None)).astype(np.float32)
        ts = grp[timestamp_col].values
        orig_idx = grp.index.to_numpy()  # these are reset — need original

        if len(grp) <= lookback:
            continue

        # Map back to the original dataframe index via timestamp + keys
        cutoff_pos = np.searchsorted(
            pd.to_datetime(ts),
            pd.Timestamp(train_cutoff),
            side="right",
        )

        for i in range(max(lookback, cutoff_pos), len(grp)):
            all_X.append(f[i - lookback:i])
            all_y.append(float(y[i]))
            # Use the original val_full index if present, else internal
            all_idx.append(int(orig_idx[i]))

    if not all_X:
        n_feat = len(feature_cols)
        return (
            np.empty((0, lookback, n_feat), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            [],
        )
    return np.stack(all_X), np.array(all_y, dtype=np.float32), all_idx


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--meter-chunk", type=int, default=METER_CHUNK,
        help="Number of meters to load per training batch (default: %(default)s).",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
    )
    parser.add_argument(
        "--patience", type=int, default=PATIENCE,
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.random_state)

    logger.info("Loading data (feature_set=engineered)...")
    prep = prepare_data(cfg, feature_set="engineered", target_log=True)

    feature_cols = prep.feature_cols
    keys = ["building_id", "meter"]
    ts_col = "timestamp"

    train_cutoff = prep.train_full[ts_col].max()
    logger.info("Train cutoff: %s", train_cutoff)

    # All unique (building_id, meter) pairs in the FULL dataset.
    all_keys = (
        prep.train_full[keys]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    n_meters = len(all_keys)
    logger.info("Total meters: %d  |  chunk size: %d", n_meters, args.meter_chunk)

    # ------------------------------------------------------------------
    # Fit a global feature scaler on a random sample of train windows.
    # We sample up to 20,000 rows from the raw training data directly
    # (no windowing needed — Standard Scaler operates per-feature).
    # ------------------------------------------------------------------
    logger.info("Fitting feature scaler on random sample of train rows...")
    from sklearn.preprocessing import StandardScaler

    sample_size = min(100_000, len(prep.train_full))
    sample_df = prep.train_full.sample(n=sample_size, random_state=42)
    scaler = StandardScaler()
    scaler.fit(sample_df[list(feature_cols)].to_numpy(dtype=np.float32))
    logger.info("Scaler fitted on %d rows.", sample_size)

    # ------------------------------------------------------------------
    # Build the PyTorch model.
    # ------------------------------------------------------------------
    import torch
    from torch import nn

    n_feat = len(feature_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    lstm_layer = nn.LSTM(
        input_size=n_feat, hidden_size=HIDDEN,
        num_layers=LAYERS, dropout=DROPOUT if LAYERS > 1 else 0.0,
        batch_first=True,
    ).to(device)
    head = nn.Linear(HIDDEN, 1).to(device)

    def forward(x: torch.Tensor) -> torch.Tensor:
        out, _ = lstm_layer(x)
        return head(out[:, -1, :])

    opt = torch.optim.Adam(
        list(lstm_layer.parameters()) + list(head.parameters()), lr=LR,
    )
    loss_fn = nn.MSELoss()

    rng = np.random.default_rng(cfg.random_state)

    # ------------------------------------------------------------------
    # Training loop: each epoch iterates over all meter chunks.
    # ------------------------------------------------------------------
    # Concatenate train + val for building validation windows that need
    # lookback context from the training period.
    full_df = pd.concat(
        [prep.train_full, prep.val_full], ignore_index=False
    ).sort_values(keys + [ts_col])

    best_val_loss = float("inf")
    bad_epochs = 0
    best_state: dict | None = None

    t_start = time.perf_counter()

    for ep in range(args.epochs):
        lstm_layer.train()
        head.train()

        # Shuffle the meter order every epoch.
        shuffled_keys = all_keys.sample(frac=1, random_state=int(rng.integers(1 << 31)))

        epoch_loss_sum = 0.0
        epoch_n_batches = 0

        n_chunks = (n_meters + args.meter_chunk - 1) // args.meter_chunk
        for chunk_i, chunk_df in enumerate(_iter_meter_chunks(
            prep.train_full, shuffled_keys, args.meter_chunk, keys, ts_col
        )):
            if chunk_i % 10 == 0:
                logger.info("ep=%d  chunk %d/%d  elapsed=%.1f min",
                            ep, chunk_i, n_chunks,
                            (time.perf_counter() - t_start) / 60)
            X_c, y_c, _ = _build_windows(
                chunk_df, feature_cols, "meter_reading",
                keys, ts_col, LOOKBACK, STRIDE,
                target_log=True, train_cutoff=None,
            )
            if len(X_c) == 0:
                continue

            # Apply scaler in-place.
            orig_shape = X_c.shape
            X_c = scaler.transform(X_c.reshape(-1, n_feat)).reshape(orig_shape)

            Xt = torch.from_numpy(np.ascontiguousarray(X_c, dtype=np.float32))
            yt = torch.from_numpy(np.ascontiguousarray(y_c, dtype=np.float32)).unsqueeze(1)
            n = Xt.shape[0]
            order = rng.permutation(n)

            for start in range(0, n, BATCH_SIZE):
                idx = order[start:start + BATCH_SIZE]
                xb = Xt[idx].to(device)
                yb = yt[idx].to(device)
                opt.zero_grad()
                p = forward(xb)
                loss = loss_fn(p, yb)
                loss.backward()
                opt.step()
                epoch_loss_sum += loss.item()
                epoch_n_batches += 1

        # ------------------------------------------------------------------
        # Validation: compute loss on a representative subset of val meters
        # (first `meter_chunk` meters in sorted order) to keep it fast.
        # ------------------------------------------------------------------
        lstm_layer.eval()
        head.eval()

        val_chunk_keys = all_keys.head(args.meter_chunk)
        val_chunk_df = prep.val_full.merge(val_chunk_keys, on=keys, how="inner")
        chunk_full_df = full_df.merge(val_chunk_keys, on=keys, how="inner")

        val_X_list: list[np.ndarray] = []
        val_y_list: list[np.ndarray] = []
        for _, grp in chunk_full_df.groupby(keys, observed=True, sort=False):
            grp = grp.sort_values(ts_col).reset_index(drop=True)
            f = grp[list(feature_cols)].to_numpy(dtype=np.float32)
            y_raw = grp["meter_reading"].to_numpy(dtype=np.float64)
            y = np.log1p(np.clip(y_raw, 0, None)).astype(np.float32)
            ts = pd.to_datetime(grp[ts_col].values)
            if len(grp) <= LOOKBACK:
                continue
            cutoff_pos = np.searchsorted(ts, pd.Timestamp(train_cutoff), side="right")
            for i in range(max(LOOKBACK, cutoff_pos), len(grp)):
                val_X_list.append(f[i - LOOKBACK:i])
                val_y_list.append(float(y[i]))

        if val_X_list:
            vX = np.stack(val_X_list)
            vY = np.array(val_y_list, dtype=np.float32)
            orig_shape = vX.shape
            vX = scaler.transform(vX.reshape(-1, n_feat)).reshape(orig_shape)
            vX_t = torch.from_numpy(np.ascontiguousarray(vX, dtype=np.float32))
            vY_t = torch.from_numpy(np.ascontiguousarray(vY, dtype=np.float32)).unsqueeze(1)
            with torch.no_grad():
                pv_chunks = []
                for i in range(0, len(vX_t), BATCH_SIZE):
                    pv_chunks.append(forward(vX_t[i:i + BATCH_SIZE].to(device)).cpu())
                pv = torch.cat(pv_chunks)
                val_loss = loss_fn(pv, vY_t).item()
        else:
            val_loss = float("inf")

        avg_train_loss = epoch_loss_sum / max(epoch_n_batches, 1)
        elapsed = (time.perf_counter() - t_start) / 60
        logger.info(
            "ep=%d  train_mse=%.4f  val_mse=%.4f  elapsed=%.1f min",
            ep, avg_train_loss, val_loss, elapsed,
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            bad_epochs = 0
            best_state = {
                "lstm": {k: v.detach().cpu().clone() for k, v in lstm_layer.state_dict().items()},
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                logger.info("Early stopping at ep=%d (patience=%d)", ep, args.patience)
                break

    if best_state is not None:
        lstm_layer.load_state_dict(best_state["lstm"])
        head.load_state_dict(best_state["head"])
    lstm_layer.eval()
    head.eval()

    # ------------------------------------------------------------------
    # Full validation pass — iterate over ALL meters, chunk by chunk.
    # Collect predictions aligned to prep.val_full original index.
    # ------------------------------------------------------------------
    logger.info("Running full validation inference over all %d meters...", n_meters)

    # We need original val_full index for each prediction.
    # Strategy: for each meter, the val windows start after train_cutoff.
    # We rebuild windows from (train_full ∪ val_full) per chunk and record
    # which original val_full rows each prediction corresponds to.

    # Build a lookup: (building_id, meter, timestamp) → original val_full index.
    val_lookup = (
        prep.val_full
        .reset_index()   # brings the index into a column named 'index'
        .set_index(keys + [ts_col])["index"]
        .to_dict()
    )

    all_preds: dict[int, float] = {}  # orig_idx → log-space prediction

    for chunk_df in _iter_meter_chunks(
        full_df, all_keys, args.meter_chunk, keys, ts_col
    ):
        for _, grp in chunk_df.groupby(keys, observed=True, sort=False):
            grp = grp.sort_values(ts_col).reset_index(drop=True)
            f = grp[list(feature_cols)].to_numpy(dtype=np.float32)
            ts = pd.to_datetime(grp[ts_col].values)
            ts_raw = grp[ts_col].values
            bid = int(grp["building_id"].iloc[0])
            mtr = int(grp["meter"].iloc[0])

            if len(grp) <= LOOKBACK:
                continue

            cutoff_pos = np.searchsorted(ts, pd.Timestamp(train_cutoff), side="right")

            chunk_X: list[np.ndarray] = []
            chunk_info: list[tuple] = []  # (building_id, meter, timestamp)

            for i in range(max(LOOKBACK, cutoff_pos), len(grp)):
                chunk_X.append(f[i - LOOKBACK:i])
                chunk_info.append((bid, mtr, ts_raw[i]))

            if not chunk_X:
                continue

            cX = np.stack(chunk_X)
            orig_shape = cX.shape
            cX = scaler.transform(cX.reshape(-1, n_feat)).reshape(orig_shape)
            cX_t = torch.from_numpy(np.ascontiguousarray(cX, dtype=np.float32))

            with torch.no_grad():
                out_chunks = []
                for i in range(0, len(cX_t), BATCH_SIZE):
                    out_chunks.append(forward(cX_t[i:i + BATCH_SIZE].to(device)).cpu().numpy())
            preds = np.concatenate(out_chunks).squeeze(-1)

            for (b, m, t), pred_val in zip(chunk_info, preds):
                key = (b, m, pd.Timestamp(t))
                orig_idx = val_lookup.get(key)
                if orig_idx is not None:
                    all_preds[orig_idx] = float(pred_val)

    if not all_preds:
        logger.error("No predictions collected — check meter key alignment.")
        return

    # Align predictions to val_full row order.
    val_idx = sorted(all_preds.keys())
    preds_log = np.array([all_preds[i] for i in val_idx], dtype=np.float32)
    val_subset = prep.val_full.loc[val_idx]
    y_true_log = np.log1p(
        np.clip(val_subset["meter_reading"].to_numpy(dtype=np.float64), 0, None)
    ).astype(np.float32)

    coverage = len(val_idx) / len(prep.val_full) * 100
    logger.info(
        "Collected %d predictions (%.1f%% of val_full).",
        len(val_idx), coverage,
    )

    # ------------------------------------------------------------------
    # Score and save.
    # ------------------------------------------------------------------
    elapsed_total = time.perf_counter() - t_start
    eval_payload = evaluate_predictions(val_subset, preds_log, y_true_log, target_log=True)
    overall = eval_payload["overall"]

    logger.info(
        "[lstm_full] RMSE=%.2f | MAE=%.2f | CV-RMSE=%.3f | coverage=%.1f%%",
        overall["rmse"], overall["mae"], overall["cv_rmse"], coverage,
    )

    # Save predictions.
    pred_dir = cfg.paths.metrics / "predictions_engineered"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "lstm_full_log_preds.npy", preds_log)
    logger.info("Saved predictions to %s", pred_dir / "lstm_full_log_preds.npy")

    # Patch into models_engineered.json / .csv.
    metrics_path = cfg.paths.metrics / "models_engineered.json"
    table_path = cfg.paths.tables / "models_engineered.csv"

    new_row = {
        "model": "lstm",
        "family": "time_series",
        **overall,
        "train_seconds": float(elapsed_total),
    }

    # Load and update existing results.
    if metrics_path.exists():
        existing = load_json(metrics_path)
    else:
        existing = {"models": {}}

    existing.setdefault("models", {})["lstm"] = {
        "family": "time_series",
        "metrics": overall,
        "by_primary_use": eval_payload["by_primary_use"],
        "by_meter": eval_payload["by_meter"],
        "by_site": eval_payload["by_site"],
        "feature_importance": None,
        "train_seconds": float(elapsed_total),
        "n_predicted_rows": len(val_idx),
        "coverage_pct": round(coverage, 2),
    }
    save_json(existing, metrics_path)

    if table_path.exists():
        df = pd.read_csv(table_path)
        # Replace or append the lstm row.
        df = df[df["model"] != "lstm"]
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df.sort_values("rmse")
    df.to_csv(table_path, index=False)
    logger.info("Updated %s and %s", table_path, metrics_path)


if __name__ == "__main__":
    main()
