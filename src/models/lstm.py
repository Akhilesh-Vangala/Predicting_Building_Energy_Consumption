from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseModel


class LSTMModel(BaseModel):
    """Sequence model with a 168-hour lookback per meter.

    Operates on per-meter sequences: for each (building_id, meter) we form
    overlapping windows of length `lookback`. The training tensor is
    (windows, lookback, n_features); the target is the meter_reading at the
    timestamp immediately after the window. Static metadata features are
    broadcast across the time dimension.
    """

    name = "lstm"
    family = "time_series"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.scaler_x_ = None
        self.scaler_y_ = None

    def _make_windows(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        meter_keys: list[str],
        timestamp_col: str,
        lookback: int,
        train_cutoff: pd.Timestamp | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        df = df.sort_values(meter_keys + [timestamp_col])
        all_X: list[np.ndarray] = []
        all_y: list[float] = []
        all_t: list[pd.Timestamp] = []
        all_idx: list[int] = []

        feat_arr_per_group = df.groupby(meter_keys, observed=True)

        for _, grp in feat_arr_per_group:
            f = grp[feature_cols].to_numpy(dtype=np.float32)
            y = grp[target_col].to_numpy(dtype=np.float32)
            ts = grp[timestamp_col].to_numpy()
            idx = grp.index.to_numpy()
            if len(grp) <= lookback:
                continue
            for i in range(lookback, len(grp)):
                if train_cutoff is not None and pd.Timestamp(ts[i]) > train_cutoff:
                    continue
                all_X.append(f[i - lookback:i])
                all_y.append(y[i])
                all_t.append(pd.Timestamp(ts[i]))
                all_idx.append(int(idx[i]))

        if not all_X:
            return np.empty((0, lookback, len(feature_cols)), dtype=np.float32), \
                   np.empty(0, dtype=np.float32), np.empty(0, dtype=object), []
        X = np.stack(all_X)
        y = np.array(all_y, dtype=np.float32)
        t = np.array(all_t, dtype=object)
        return X, y, t, all_idx

    def fit_predict_sequences(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "meter_reading",
        meter_keys: tuple[str, str] = ("building_id", "meter"),
        timestamp_col: str = "timestamp",
        target_log: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        import logging
        import torch
        from torch import nn
        from sklearn.preprocessing import StandardScaler

        log = logging.getLogger(__name__)

        lookback = int(self.params.get("lookback", 168))
        hidden = int(self.params.get("hidden_size", 128))
        layers = int(self.params.get("num_layers", 2))
        dropout = float(self.params.get("dropout", 0.2))
        bs = int(self.params.get("batch_size", 256))
        lr = float(self.params.get("learning_rate", 1e-3))
        epochs = int(self.params.get("epochs", 30))
        patience = int(self.params.get("patience", 5))
        max_meters = self.params.get("max_meters", None)
        stride = int(self.params.get("stride", 1))

        keys = list(meter_keys)
        all_keys = train_df[keys].drop_duplicates()
        if max_meters is not None and len(all_keys) > int(max_meters):
            n = int(max_meters)
            meter_col = "meter" if "meter" in keys else None
            if meter_col is not None:
                # Stratified sample so every meter type is represented
                parts = [
                    g.sample(n=max(1, round(n * len(g) / len(all_keys))), random_state=42)
                    for _, g in all_keys.groupby(meter_col, group_keys=False)
                ]
                import pandas as _pd
                all_keys = _pd.concat(parts).drop_duplicates()
                if len(all_keys) > n:
                    all_keys = all_keys.sample(n=n, random_state=42)
            else:
                all_keys = all_keys.sample(n=n, random_state=42)
        log.info("LSTM operating on %d meters (stride=%d, lookback=%d)",
                 len(all_keys), stride, lookback)

        train_df = train_df.merge(all_keys, on=keys, how="inner")
        val_df = val_df.merge(all_keys, on=keys, how="inner")
        train_df = train_df.sort_values(keys + [timestamp_col]).reset_index(drop=True)
        val_df = val_df.sort_values(keys + [timestamp_col]).reset_index(drop=True)

        train_X: list[np.ndarray] = []
        train_y: list[float] = []
        for _, grp in train_df.groupby(keys, observed=True):
            f = grp[list(feature_cols)].to_numpy(dtype=np.float32)
            y = grp[target_col].to_numpy(dtype=np.float32)
            if len(grp) <= lookback:
                continue
            for i in range(lookback, len(grp), stride):
                train_X.append(f[i - lookback:i])
                train_y.append(y[i])
        if not train_X:
            log.warning("LSTM: no training windows; skipping")
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), []
        X_train = np.stack(train_X)
        y_train = np.array(train_y, dtype=np.float32)
        if target_log:
            y_train = np.log1p(np.clip(y_train, 0, None)).astype(np.float32)
        log.info("LSTM training tensor shape=%s (%.1f MB)", X_train.shape,
                 X_train.nbytes / 1e6)

        full = pd.concat([train_df, val_df], ignore_index=True)
        full = full.sort_values(keys + [timestamp_col]).reset_index(drop=True)
        train_max = train_df[timestamp_col].max()

        all_X: list[np.ndarray] = []
        all_y: list[float] = []
        val_idx: list[int] = []
        for _, grp in full.groupby(keys, observed=True):
            f = grp[list(feature_cols)].to_numpy(dtype=np.float32)
            y = grp[target_col].to_numpy(dtype=np.float32)
            if target_log:
                y = np.log1p(np.clip(y, 0, None)).astype(np.float32)
            ts = grp[timestamp_col].to_numpy()
            if len(grp) <= lookback:
                continue
            cutoff_idx = np.searchsorted(ts, np.datetime64(train_max), side="right")
            for i in range(max(lookback, cutoff_idx), len(grp)):
                all_X.append(f[i - lookback:i])
                all_y.append(y[i])
                val_idx.append(int(grp.index[i]))
        if not all_X:
            log.warning("LSTM: no validation windows; skipping")
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), []
        X_val = np.stack(all_X)
        y_val_actual = np.array(all_y, dtype=np.float32)
        log.info("LSTM validation tensor shape=%s (%.1f MB)", X_val.shape,
                 X_val.nbytes / 1e6)

        n_feat = X_train.shape[2]
        flat = X_train.reshape(-1, n_feat)
        self.scaler_x_ = StandardScaler().fit(flat)
        X_train_s = self.scaler_x_.transform(flat).reshape(X_train.shape)
        X_val_s = self.scaler_x_.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = nn.Sequential().to(device)
        net.lstm = nn.LSTM(
            input_size=n_feat, hidden_size=hidden,
            num_layers=layers, dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        ).to(device)
        net.head = nn.Linear(hidden, 1).to(device)

        def forward(x: "torch.Tensor") -> "torch.Tensor":
            out, _ = net.lstm(x)
            return net.head(out[:, -1, :])

        opt = torch.optim.Adam(list(net.lstm.parameters()) + list(net.head.parameters()), lr=lr)
        loss_fn = nn.MSELoss()

        Xt = torch.from_numpy(np.ascontiguousarray(X_train_s, dtype=np.float32))
        yt = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32)).unsqueeze(1)
        n_train = Xt.shape[0]
        rng = np.random.default_rng(int(self.params.get("random_state", 42)))

        bad = 0
        best_state: dict | None = None
        best_val = float("inf")
        Xv = torch.from_numpy(np.ascontiguousarray(X_val_s, dtype=np.float32))
        Yv = torch.from_numpy(np.ascontiguousarray(y_val_actual, dtype=np.float32)).unsqueeze(1)

        for ep in range(epochs):
            net.lstm.train(); net.head.train()
            order = rng.permutation(n_train)
            for start in range(0, n_train, bs):
                idx = order[start:start + bs]
                xb = Xt[idx].to(device); yb = yt[idx].to(device)
                opt.zero_grad()
                p = forward(xb)
                loss = loss_fn(p, yb)
                loss.backward()
                opt.step()

            net.lstm.eval(); net.head.eval()
            with torch.no_grad():
                pv_chunks = []
                for i in range(0, len(Xv), bs):
                    pv_chunks.append(forward(Xv[i:i+bs].to(device)).cpu())
                pv = torch.cat(pv_chunks)
                v = loss_fn(pv, Yv).item()
            log.info("LSTM ep=%d val_mse=%.4f", ep, v)
            if v < best_val - 1e-4:
                best_val = v
                bad = 0
                best_state = {
                    "lstm": {k: v_.detach().cpu().clone() for k, v_ in net.lstm.state_dict().items()},
                    "head": {k: v_.detach().cpu().clone() for k, v_ in net.head.state_dict().items()},
                }
            else:
                bad += 1
                if bad >= patience:
                    log.info("LSTM early-stopped at ep=%d", ep)
                    break

        if best_state is not None:
            net.lstm.load_state_dict(best_state["lstm"])
            net.head.load_state_dict(best_state["head"])

        net.lstm.eval(); net.head.eval()
        with torch.no_grad():
            pv_chunks = []
            for i in range(0, len(Xv), bs):
                pv_chunks.append(forward(Xv[i:i+bs].to(device)).cpu().numpy())
            preds = np.concatenate(pv_chunks).squeeze(-1)
        self.model_ = (net, device)
        return preds, y_val_actual, val_idx

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LSTMModel":
        raise NotImplementedError("Use LSTMModel.fit_predict_sequences instead.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Use LSTMModel.fit_predict_sequences instead.")
