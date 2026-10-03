"""Group-aware conformal quantiles and predictive distributions.

Plain idea: a model's prediction comes with a range. To make the range honest
we look at how wrong the model was on data it had not seen (out-of-fold
errors) and set the range from those errors. This is conformal prediction.

Two details matter here:

- **Groups, not rows.** Thousands of one-second readings from one battery are
  not independent: if the model is off for that battery, it is off for all of
  them. So every battery chain gets the same total weight when error quantiles
  are computed, and the small-sample correction uses the number of *chains*.
- **Small-sample correction is optional.** Conformal theory widens the range
  when few groups are available (``smooth``: level q -> q (1 + 1/G); ``ceil``:
  ceil((G + 1) q) / G). With ~40 chains that widening is large: measured on
  held-out chains, the nominal 90% range covered 89.5% uncorrected, 93.9% with
  ``smooth`` and 96.8% with ``ceil`` (ranges 11, 14 and 19 Wh wide). The default
  is therefore ``none``: calibrated to the nominal level, with caution applied
  explicitly by the decision threshold in Phase 6 instead of hidden in the range.

``PredictiveDistribution`` stores quantiles at fixed levels and can be sampled
or queried; the feasibility layer (Phase 6) combines two of them.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from endurosense.config import load_config


def group_weights(groups) -> np.ndarray:
    """Row weights such that every group has total weight 1."""
    g = pd.Series(np.asarray(groups))
    return (1.0 / g.map(g.value_counts())).to_numpy()


def weighted_quantile(values, q, weights) -> float:
    """Quantile of ``values`` under ``weights`` (inverse of the weighted CDF, lower interpolation).

    The smallest value whose cumulative weight reaches ``q``. A cumulative weight that
    equals ``q`` up to rounding counts as reaching it, so exact ties (common with
    group weights such as 1/2, 1/3) do not depend on floating-point rounding.
    """
    v, w = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(v, kind="stable")
    v, cw = v[order], np.cumsum(w[order])
    return float(v[min(np.searchsorted(cw, (q - 1e-9) * cw[-1], side="left"), len(v) - 1)])


def adjusted_level(q: float, n_groups: int, correction: str | None = None) -> float:
    """Probability level after the small-sample correction (pushed away from the median)."""
    correction = correction or load_config()["uncertainty"]["finite_sample_correction"]
    if correction == "none":
        return q
    if correction == "smooth":
        return min(1.0, q * (1 + 1 / n_groups)) if q >= 0.5 else max(0.0, 1 - (1 - q) * (1 + 1 / n_groups))
    if correction == "ceil":
        if q >= 0.5:
            return min(1.0, math.ceil((n_groups + 1) * q) / n_groups)
        return max(0.0, math.floor((n_groups + 1) * q) / n_groups)
    raise ValueError(f"unknown finite_sample_correction {correction!r}")


def conformal_quantiles(scores, groups, levels, correction: str | None = None) -> np.ndarray:
    """Calibrated quantile of the error ``scores`` at each level, group-weighted."""
    w = group_weights(groups)
    n_groups = len(pd.unique(np.asarray(groups)))
    return np.array([weighted_quantile(scores, adjusted_level(q, n_groups, correction), w) for q in levels])


class PredictiveDistribution:
    """A predictive distribution given by its quantiles at fixed levels.

    Between the stored levels the quantile function is interpolated linearly;
    beyond the outermost levels it is continued with a normal tail matched to
    the outermost quantiles (so extreme draws remain finite and sensible).
    """

    def __init__(self, levels, values):
        self.levels = np.asarray(levels, float)
        self.values = np.maximum.accumulate(np.asarray(values, float))      # enforce non-crossing quantiles
        z = norm.ppf(self.levels)
        spread = (self.values[-1] - self.values[0]) / (z[-1] - z[0])
        self._tail_sd = max(float(spread), 1e-9)

    def quantile(self, q):
        q = np.atleast_1d(np.asarray(q, float))
        out = np.interp(q, self.levels, self.values)
        lo, hi = q < self.levels[0], q > self.levels[-1]
        out[lo] = self.values[0] + self._tail_sd * (norm.ppf(q[lo]) - norm.ppf(self.levels[0]))
        out[hi] = self.values[-1] + self._tail_sd * (norm.ppf(q[hi]) - norm.ppf(self.levels[-1]))
        return out if out.size > 1 else float(out[0])

    def cdf(self, x: float) -> float:
        """P(value <= x)."""
        if x <= self.values[0]:
            return float(norm.cdf(norm.ppf(self.levels[0]) + (x - self.values[0]) / self._tail_sd))
        if x >= self.values[-1]:
            return float(norm.cdf(norm.ppf(self.levels[-1]) + (x - self.values[-1]) / self._tail_sd))
        return float(np.interp(x, self.values, self.levels))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.atleast_1d(self.quantile(rng.uniform(1e-6, 1 - 1e-6, n)))

    def interval(self, coverage: float | None = None) -> tuple[float, float]:
        """Central range; by default the headline level from the config (90%)."""
        coverage = load_config()["uncertainty"]["interval"] if coverage is None else coverage
        a = (1 - coverage) / 2
        return float(self.quantile(a)), float(self.quantile(1 - a))

    @property
    def median(self) -> float:
        return float(self.quantile(0.5))
