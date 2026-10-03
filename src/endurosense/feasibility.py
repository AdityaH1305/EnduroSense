"""Mission feasibility: the probability that a battery can complete a mission.

Plain idea: Model A says how much energy the battery can still give (a range),
Model B says how much the mission will take (a range). The mission succeeds if
the battery's energy is at least the mission's energy. Because both are ranges,
the answer is a probability:

    P(success) = P(energy available >= energy required)

"Energy available" already excludes the safety reserve, so success means the
mission ends with the reserve intact.

Computation: with A and B independent,

    P(A >= B) = average over A's distribution of  P(B <= a)

which is evaluated on a fine grid of A's quantiles (deterministic, no random
sampling, so the same inputs always give the same probability).

Independence of the two models' errors is an assumption; it is checked on real
flights in ``scripts/08_decisions.py`` (their correlation is reported there).

Decision rule: approve the mission if P(success) >= tau (default 0.95).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from endurosense.config import load_config
from endurosense.uncertainty.quantiles import PredictiveDistribution


def _quantile_grid(q_rows: np.ndarray, levels: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Quantile function of each row evaluated at probabilities ``u`` -> (rows, len(u)).

    Linear interpolation between stored levels, normal-shaped tails beyond them
    (same rule as ``PredictiveDistribution``)."""
    z = norm.ppf(levels)
    sd = np.maximum((q_rows[:, -1] - q_rows[:, 0]) / (z[-1] - z[0]), 1e-9)[:, None]
    out = np.stack([np.interp(u, levels, row) for row in q_rows])
    lo, hi = u < levels[0], u > levels[-1]
    out[:, lo] = q_rows[:, [0]] + sd * (norm.ppf(u[lo]) - z[0])[None, :]
    out[:, hi] = q_rows[:, [-1]] + sd * (norm.ppf(u[hi]) - z[-1])[None, :]
    return out


def _cdf_rows(q_rows: np.ndarray, levels: np.ndarray, x: np.ndarray) -> np.ndarray:
    """CDF of each row's distribution at its own points ``x`` (rows, k) -> (rows, k)."""
    z = norm.ppf(levels)
    sd = np.maximum((q_rows[:, -1] - q_rows[:, 0]) / (z[-1] - z[0]), 1e-9)
    out = np.empty_like(x, dtype=float)
    for i in range(len(q_rows)):
        xi, v = x[i], q_rows[i]
        c = np.interp(xi, v, levels)
        lo, hi = xi <= v[0], xi >= v[-1]
        c[lo] = norm.cdf(z[0] + (xi[lo] - v[0]) / sd[i])
        c[hi] = norm.cdf(z[-1] + (xi[hi] - v[-1]) / sd[i])
        out[i] = c
    return out


def p_success_batch(q_available: np.ndarray, q_required: np.ndarray, levels, n_grid: int = 400) -> np.ndarray:
    """P(available >= required) for many pairs at once.

    ``q_available`` and ``q_required`` are (pairs, levels) arrays of predictive
    quantiles (non-decreasing along each row).
    """
    levels = np.asarray(levels, float)
    qa = np.maximum.accumulate(np.asarray(q_available, float), axis=1)
    qb = np.maximum.accumulate(np.asarray(q_required, float), axis=1)
    u = (np.arange(n_grid) + 0.5) / n_grid
    a = _quantile_grid(qa, levels, u)                    # equally likely values of energy available
    return _cdf_rows(qb, levels, a).mean(axis=1)         # how often the mission needs no more than that


def p_success(available: PredictiveDistribution, required: PredictiveDistribution, n_grid: int = 400) -> float:
    """P(available >= required) for one battery state and one mission."""
    u = (np.arange(n_grid) + 0.5) / n_grid
    return float(np.mean([required.cdf(a) for a in np.atleast_1d(available.quantile(u))]))


def decide(p: float | None, tau: float | None = None) -> bool:
    """GO only if the success probability reaches the threshold. ``None`` (the
    battery model abstained: no pre-flight voltage reading) is always NO-GO."""
    tau = load_config()["decision"]["tau"] if tau is None else tau
    return p is not None and p >= tau
