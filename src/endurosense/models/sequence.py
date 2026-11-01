"""Sequence Model A predictors: LSTM and GRU (PyTorch).

Each reads the last 60 s of battery signals (2 Hz: voltage, current, power,
energy drawn, motors-on, plus a padding mask) with a recurrent layer, then
joins the recurrent summary with the same 25 tabular features the other
models get, so every algorithm sees the same information. A small MLP head
predicts remaining Wh.

Training: Huber loss (in Wh), AdamW, early stopping on validation MAE. The
early-stopping validation set is 15% of the *training* chains (grouped), so
the cross-validation fold being scored is never touched during training.
All normalisation statistics come from the training portion only.
"""
from __future__ import annotations

import copy
import io

import numpy as np
import pandas as pd
import torch
from torch import nn

from endurosense.config import load_config
from endurosense.features.model_a import FEATURES


class WindowStore:
    """Pre-built sequence windows for every row of the Model A table (aligned by row index)."""

    def __init__(self, X: np.ndarray, mask: np.ndarray):
        self.X = np.concatenate([X, mask[..., None]], axis=2).astype(np.float32)   # mask as an extra channel

    def get(self, index) -> np.ndarray:
        return self.X[np.asarray(index)]


class _Net(nn.Module):
    def __init__(self, kind: str, n_channels: int, n_static: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(n_channels, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, seq, static):
        out, _ = self.rnn(seq)
        return self.head(torch.cat([out[:, -1], static], dim=1)).squeeze(1)


class SequenceModel:
    def __init__(self, kind: str, windows: WindowStore, hidden: int = 64, layers: int = 1,
                 dropout: float = 0.1, seed: int = 42, features: list[str] = FEATURES, device: str | None = None):
        self.kind, self.windows, self.features = kind, windows, list(features)
        self.hidden, self.layers, self.dropout, self.seed = hidden, layers, dropout, seed
        self.name = "LSTM" if kind == "lstm" else "GRU"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = load_config()["model_a_training"]["nn"]
        self.epochs_trained = 0

    # -------------------------------------------------------------- preparation
    def _static(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.features].to_numpy(dtype=np.float64)
        x = np.where(np.isnan(x), self.med, x)
        return ((x - self.s_mu) / self.s_sd).astype(np.float32)

    def _seq(self, df: pd.DataFrame) -> np.ndarray:
        return ((self.windows.get(df.index) - self.q_mu) / self.q_sd).astype(np.float32)

    def _fit_scalers(self, tr: pd.DataFrame, target: str):
        x = tr[self.features].to_numpy(dtype=np.float64)
        self.med = np.nanmedian(x, axis=0)
        x = np.where(np.isnan(x), self.med, x)
        self.s_mu, self.s_sd = x.mean(0), x.std(0) + 1e-8
        q = self.windows.get(tr.index)
        self.q_mu, self.q_sd = q.mean(axis=(0, 1)), q.std(axis=(0, 1)) + 1e-8
        self.q_mu[-1], self.q_sd[-1] = 0.0, 1.0                     # leave the mask channel as 0/1
        self.y_mu, self.y_sd = float(tr[target].mean()), float(tr[target].std())

    # ------------------------------------------------------------------ fitting
    def fit(self, train: pd.DataFrame, target: str) -> "SequenceModel":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        chains = train["battery_chain"].unique()
        val_chains = set(rng.choice(chains, max(1, int(round(0.15 * len(chains)))), replace=False))
        is_val = train["battery_chain"].isin(val_chains).to_numpy()
        tr, va = train[~is_val], train[is_val]
        self._fit_scalers(tr, target)

        def tensors(d):
            return (torch.from_numpy(self._seq(d)), torch.from_numpy(self._static(d)),
                    torch.from_numpy(d[target].to_numpy(np.float32)))

        Xs, Xt, y = tensors(tr)
        Vs, Vt, vy = (t.to(self.device) for t in tensors(va))
        self.net = _Net(self.kind, Xs.shape[2], Xt.shape[1], self.hidden, self.layers, self.dropout).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        loss_fn = nn.HuberLoss(delta=self.cfg["huber_delta_wh"])
        best, best_state, bad = np.inf, None, 0
        for epoch in range(self.cfg["max_epochs"]):
            self.net.train()
            perm = torch.randperm(len(y))
            for b in range(0, len(y), self.cfg["batch"]):
                idx = perm[b:b + self.cfg["batch"]]
                pred = self.net(Xs[idx].to(self.device), Xt[idx].to(self.device)) * self.y_sd + self.y_mu
                loss = loss_fn(pred, y[idx].to(self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                mae = float((self.net(Vs, Vt) * self.y_sd + self.y_mu - vy).abs().mean())
            if mae < best - 1e-4:
                best, best_state, bad = mae, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.cfg["patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.epochs_trained, self.val_mae = epoch + 1, best
        return self

    # --------------------------------------------------------------- inference
    def predict(self, df: pd.DataFrame, batch: int = 4096) -> np.ndarray:
        self.net.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(df), batch):
                d = df.iloc[b:b + batch]
                s = torch.from_numpy(self._seq(d)).to(self.device)
                t = torch.from_numpy(self._static(d)).to(self.device)
                out.append((self.net(s, t) * self.y_sd + self.y_mu).cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    def size_bytes(self) -> int:
        buf = io.BytesIO()
        torch.save(self.net.state_dict(), buf)
        return buf.getbuffer().nbytes
