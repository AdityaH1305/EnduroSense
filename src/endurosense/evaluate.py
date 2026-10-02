"""Shared evaluation: metrics, grouped cross-validation, speed and size.

Cross-validation uses the locked development folds (grouped by battery chain),
so every prediction is made by a model that never saw that battery.
Out-of-fold (OOF) predictions are kept for error analysis and for later
phases (uncertainty, decisions).
"""
from __future__ import annotations

import io
import pickle
import time
from typing import Callable

import numpy as np
import pandas as pd

from endurosense.data.split import dev_folds


def metrics(y_true, y_pred) -> dict:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = p - y
    ss_res, ss_tot = np.sum(err ** 2), np.sum((y - y.mean()) ** 2)
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1 - ss_res / ss_tot), "bias": float(np.mean(err))}


def cross_validate(factory: Callable[[], object], df: pd.DataFrame, target: str) -> tuple[pd.Series, list[float]]:
    """OOF predictions over the dev folds (a fresh model per fold) and per-fold MAE."""
    oof = pd.Series(np.nan, index=df.index)
    fold_mae = []
    for _, train, val in dev_folds(df):
        pred = factory().fit(train, target).predict(val)
        oof.loc[val.index] = pred
        fold_mae.append(float(np.mean(np.abs(pred - val[target].to_numpy()))))
    return oof, fold_mae


def paired_comparison(y, pred_a, pred_b, groups, n_boot: int = 2000, seed: int = 42) -> dict:
    """Is model A really better than model B? MAE difference (A - B) with a 95%
    bootstrap interval that resamples whole groups (battery chains), because
    rows from one battery are not independent."""
    err_a = np.abs(np.asarray(pred_a) - np.asarray(y))
    err_b = np.abs(np.asarray(pred_b) - np.asarray(y))
    g = pd.DataFrame({"g": np.asarray(groups), "a": err_a, "b": err_b}).groupby("g").agg(["sum", "count"])
    sa, sb, n = g[("a", "sum")].to_numpy(), g[("b", "sum")].to_numpy(), g[("a", "count")].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(n_boot, len(n)))
    diffs = (sa[idx].sum(1) - sb[idx].sum(1)) / n[idx].sum(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mae_diff": float(err_a.mean() - err_b.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "share_groups_better": float(np.mean(sa / n < sb / n))}


def single_threaded(model):
    """Set every estimator inside ``model`` to one thread, so latency is comparable."""
    for obj in (model, getattr(model, "model", None)):
        pipe = getattr(obj, "pipe", None)
        if pipe is not None and hasattr(pipe[-1], "n_jobs"):
            pipe[-1].set_params(n_jobs=1)
    return model


def latency_ms(predict: Callable[[pd.DataFrame], np.ndarray], one_row: pd.DataFrame, repeats: int) -> float:
    """Median wall time (ms) of a single-row prediction."""
    predict(one_row)                                   # warm-up
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        predict(one_row)
        times.append(time.perf_counter() - t)
    return float(np.median(times) * 1000)


def size_mb(model) -> float:
    """Serialised size of a fitted model."""
    if hasattr(model, "size_bytes"):
        return model.size_bytes() / 1e6
    buf = io.BytesIO()
    pickle.dump(model, buf)
    return buf.getbuffer().nbytes / 1e6
