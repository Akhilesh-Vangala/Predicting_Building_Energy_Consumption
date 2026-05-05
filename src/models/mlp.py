from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseModel


class MLPModel(BaseModel):
    name = "mlp"
    family = "neural"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.scaler_ = None
        self._best_state: dict | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray,
            eval_set: tuple[pd.DataFrame, np.ndarray] | None = None) -> "MLPModel":
        import torch
        from torch import nn

        from sklearn.preprocessing import StandardScaler

        n_threads = int(self.params.get("torch_threads", 1))
        if n_threads >= 1:
            torch.set_num_threads(n_threads)

        self.feature_names_ = X.columns.tolist()
        Xn = X.to_numpy(dtype=np.float64)
        stds = Xn.std(axis=0)
        self.kept_idx_ = np.where(stds > 1e-8)[0]
        Xk = Xn[:, self.kept_idx_]
        self.scaler_ = StandardScaler()
        Xs = np.ascontiguousarray(self.scaler_.fit_transform(Xk), dtype=np.float32)
        ya = np.ascontiguousarray(np.asarray(y), dtype=np.float32).reshape(-1, 1)
        Xt = torch.from_numpy(Xs)
        yt = torch.from_numpy(ya)

        if eval_set is not None:
            xv, yv = eval_set
            Xvn = xv.to_numpy(dtype=np.float64)[:, self.kept_idx_]
            Xvs = np.ascontiguousarray(self.scaler_.transform(Xvn), dtype=np.float32)
            Yva = np.ascontiguousarray(np.asarray(yv), dtype=np.float32).reshape(-1, 1)
            Xv = torch.from_numpy(Xvs)
            Yv = torch.from_numpy(Yva)
        else:
            Xv = Yv = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden = self.params.get("hidden_sizes", [256, 128, 64])
        dropout = float(self.params.get("dropout", 0.3))

        layers: list[nn.Module] = []
        prev = Xt.shape[1]
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        net = nn.Sequential(*layers).to(device)

        opt = torch.optim.Adam(
            net.parameters(),
            lr=float(self.params.get("learning_rate", 1e-3)),
            weight_decay=float(self.params.get("weight_decay", 0.0)),
        )
        loss_fn = nn.MSELoss()

        bs = int(self.params.get("batch_size", 1024))
        n = Xt.shape[0]
        rng = np.random.default_rng(int(self.params.get("random_state", 42)))

        epochs = int(self.params.get("epochs", 60))
        patience = int(self.params.get("patience", 8))
        best_val = float("inf")
        bad = 0
        best_state: dict | None = None

        for ep in range(epochs):
            net.train()
            order = rng.permutation(n)
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                xb = Xt[idx].to(device)
                yb = yt[idx].to(device)
                opt.zero_grad()
                p = net(xb)
                loss = loss_fn(p, yb)
                loss.backward()
                opt.step()

            if Xv is not None:
                net.eval()
                with torch.no_grad():
                    pv = net(Xv.to(device))
                    v = loss_fn(pv, Yv.to(device)).item()
                if v < best_val - 1e-4:
                    best_val = v
                    bad = 0
                    best_state = {k: v_.detach().cpu().clone() for k, v_ in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break

        if best_state is not None:
            net.load_state_dict(best_state)

        self.model_ = net
        self._device = device
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        if self.model_ is None or self.scaler_ is None or self.kept_idx_ is None:
            raise RuntimeError("model not fit")
        self.model_.eval()
        Xn = X.to_numpy(dtype=np.float64)[:, self.kept_idx_]
        Xs = self.scaler_.transform(Xn).astype(np.float32)
        with torch.no_grad():
            t = torch.from_numpy(Xs).to(self._device)
            out = self.model_(t).cpu().numpy().squeeze(-1)
        return out
