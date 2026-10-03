"""How good are predicted ranges? Coverage, width, pinball loss, reliability.

- **Coverage**: share of true values inside the range. A 90% range should cover
  about 90%. Reported per row and per group (battery chain), because rows from
  one battery rise and fall together.
- **Width**: narrower is better *at the same coverage*.
- **Pinball loss**: one number scoring all predicted quantiles (lower = better).
- **Reliability**: for each level q, how often the truth fell below the predicted
  q-quantile. Perfect calibration gives exactly q.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from endurosense.uncertainty.quantiles import group_weights


def _interp_rows(q_values: np.ndarray, levels: np.ndarray, q: float) -> np.ndarray:
    """Row-wise quantile at level ``q`` from an (rows, levels) array of quantiles."""
    j = np.searchsorted(levels, q)
    if j < len(levels) and np.isclose(levels[j], q):
        return q_values[:, j]
    j = np.clip(j, 1, len(levels) - 1)
    t = (q - levels[j - 1]) / (levels[j] - levels[j - 1])
    return q_values[:, j - 1] + t * (q_values[:, j] - q_values[:, j - 1])


def interval_table(y, q_values, levels, groups, coverages=(0.5, 0.8, 0.9, 0.95)) -> pd.DataFrame:
    """Coverage (row- and group-weighted) and mean width of central intervals."""
    y, q_values, levels = np.asarray(y, float), np.asarray(q_values, float), np.asarray(levels, float)
    w = group_weights(groups)
    rows = []
    for c in coverages:
        lo, hi = _interp_rows(q_values, levels, (1 - c) / 2), _interp_rows(q_values, levels, 1 - (1 - c) / 2)
        inside = (y >= lo) & (y <= hi)
        rows.append({"nominal": c, "coverage": float(inside.mean()),
                     "coverage_by_group": float(np.average(inside, weights=w)),
                     "mean_width": float(np.mean(hi - lo)),
                     "too_high": float(np.mean(y < lo)), "too_low": float(np.mean(y > hi))})
    return pd.DataFrame(rows)


def pinball_loss(y, q_values, levels) -> float:
    y, q_values, levels = np.asarray(y, float)[:, None], np.asarray(q_values, float), np.asarray(levels, float)[None, :]
    d = y - q_values
    return float(np.mean(np.maximum(levels * d, (levels - 1) * d)))


def reliability(y, q_values, levels, groups) -> pd.DataFrame:
    """Observed frequency of ``y <= predicted q-quantile`` for each level (group-weighted)."""
    y, q_values = np.asarray(y, float)[:, None], np.asarray(q_values, float)
    w = group_weights(groups)
    return pd.DataFrame({"level": np.asarray(levels, float),
                         "observed": np.average(y <= q_values, axis=0, weights=w)})


def coverage_by(y, q_values, levels, by, coverage: float = 0.9) -> pd.DataFrame:
    """Coverage and width of the central ``coverage`` interval within each subgroup of ``by``."""
    y, q_values, levels = np.asarray(y, float), np.asarray(q_values, float), np.asarray(levels, float)
    lo, hi = _interp_rows(q_values, levels, (1 - coverage) / 2), _interp_rows(q_values, levels, 1 - (1 - coverage) / 2)
    d = pd.DataFrame({"by": np.asarray(by), "inside": (y >= lo) & (y <= hi), "width": hi - lo,
                      "over": y < lo})          # truth below the range: the model was too optimistic
    return d.groupby("by").agg(rows=("inside", "size"), coverage=("inside", "mean"),
                               width=("width", "mean"), truth_below_range=("over", "mean")).reset_index()
