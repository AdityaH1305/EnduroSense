"""Probabilistic Model A: an ensemble of GRUs that each predict a mean and a spread.

Plain idea: instead of one number, each network says "about 52 Wh, give or take
4". Five networks trained from different starting points are then combined:

- each network's own spread captures how noisy the estimate is in the current
  situation;
- disagreement *between* networks captures uncertainty about the model itself.

This is the "deep ensemble" idea (Lakshminarayanan et al., 2017), with one
change found necessary here: the mean and the spread are trained by separate
objectives.

- The **mean** is trained exactly like the Phase 3 GRU (Huber loss on Wh, early
  stopping on validation error). Training mean and spread together on the
  likelihood made the mean worse (2.51 vs 2.39 Wh on held-out batteries).
- The **spread head** reads the recurrent summary and the tabular features but
  passes no gradient back to them, so it cannot disturb the mean. It learns the
  size of the mean's errors with the Gaussian negative log-likelihood.

The spread is learned on training errors, which are smaller than held-out
errors, so it is treated as a *relative* signal ("more or less certain than
usual") and put on the right absolute scale afterwards by conformal calibration
(``endurosense.uncertainty``).

The early-stopping validation set is 15% of the *training* chains, a different
15% per ensemble member (which adds diversity).
"""
from __future__ import annotations

import copy
import io

import numpy as np
import pandas as pd
import torch
from torch import nn

from endurosense.models.sequence import SequenceModel, WindowStore

LOGVAR_RANGE = (-7.0, 5.0)        # in standardised target units; keeps the loss numerically stable


class _ProbNet(nn.Module):
    def __init__(self, kind: str, n_channels: int, n_static: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(n_channels, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.mean_head = nn.Sequential(nn.Linear(hidden + n_static, hidden), nn.ReLU(), nn.Dropout(dropout),
                                       nn.Linear(hidden, 1))
        self.spread_head = nn.Sequential(nn.Linear(hidden + n_static, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, seq, static):
        out, _ = self.rnn(seq)
        feats = torch.cat([out[:, -1], static], dim=1)
        mean = self.mean_head(feats).squeeze(1)
        logvar = self.spread_head(feats.detach()).squeeze(1).clamp(*LOGVAR_RANGE)   # no gradient into the mean path
        return mean, logvar


class ProbabilisticSequenceModel(SequenceModel):
    """One ensemble member: predicts mean and standard deviation of remaining Wh."""

    def fit(self, train: pd.DataFrame, target: str) -> "ProbabilisticSequenceModel":
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
        self.net = _ProbNet(self.kind, Xs.shape[2], Xt.shape[1], self.hidden, self.layers, self.dropout).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        huber = nn.HuberLoss(delta=self.cfg["huber_delta_wh"])

        best, best_state, bad = np.inf, None, 0
        for epoch in range(self.cfg["max_epochs"]):
            self.net.train()
            perm = torch.randperm(len(y))
            for b in range(0, len(y), self.cfg["batch"]):
                idx = perm[b:b + self.cfg["batch"]]
                yb = y[idx].to(self.device)
                mean, logvar = self.net(Xs[idx].to(self.device), Xt[idx].to(self.device))
                pred = mean * self.y_sd + self.y_mu
                err_z = ((yb - pred) / self.y_sd).detach()                  # the mean's error, in standard units
                loss = huber(pred, yb) + 0.5 * (logvar + err_z ** 2 / logvar.exp()).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                mae = float((self.net(Vs, Vt)[0] * self.y_sd + self.y_mu - vy).abs().mean())
            if mae < best - 1e-4:
                best, best_state, bad = mae, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.cfg["patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.epochs_trained, self.val_mae = epoch + 1, best
        return self

    def predict_dist(self, df: pd.DataFrame, batch: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Mean and standard deviation (Wh) for every row."""
        self.net.eval()
        mus, sds = [], []
        with torch.no_grad():
            for b in range(0, len(df), batch):
                d = df.iloc[b:b + batch]
                m, lv = self.net(torch.from_numpy(self._seq(d)).to(self.device),
                                 torch.from_numpy(self._static(d)).to(self.device))
                mus.append((m * self.y_sd + self.y_mu).cpu().numpy())
                sds.append(((0.5 * lv).exp() * self.y_sd).cpu().numpy())
        return np.concatenate(mus), np.concatenate(sds)

    def predict(self, df: pd.DataFrame, batch: int = 4096) -> np.ndarray:
        return self.predict_dist(df, batch)[0]


class SequenceEnsemble:
    """``n`` probabilistic members; combined as a mixture of their Gaussians."""

    def __init__(self, kind: str, windows: WindowStore, n: int = 5, seed: int = 42, **config):
        self.name = f"{'LSTM' if kind == 'lstm' else 'GRU'} ensemble ({n})"
        self.members = [ProbabilisticSequenceModel(kind, windows, seed=seed + 1000 * i, **config) for i in range(n)]

    def fit(self, train: pd.DataFrame, target: str) -> "SequenceEnsemble":
        for m in self.members:
            m.fit(train, target)
        return self

    def predict_dist(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Mixture mean and standard deviation: member noise plus disagreement between members."""
        mu, sd = zip(*(m.predict_dist(df) for m in self.members))
        mu, sd = np.stack(mu), np.stack(sd)
        mean = mu.mean(axis=0)
        var = (sd ** 2 + mu ** 2).mean(axis=0) - mean ** 2
        return mean, np.sqrt(np.maximum(var, 1e-12))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_dist(df)[0]

    def size_bytes(self) -> int:
        total = 0
        for m in self.members:
            buf = io.BytesIO()
            torch.save(m.net.state_dict(), buf)
            total += buf.getbuffer().nbytes
        return total
