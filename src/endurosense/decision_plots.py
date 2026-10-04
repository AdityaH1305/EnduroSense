"""Figures of the decision evaluation, shared by the development run (Phase 6)
and the final test run (Phase 7) so both are drawn the same way."""
from __future__ import annotations

import numpy as np
import pandas as pd

from endurosense import whatif as W
from endurosense.plots import INK_MUTED, SERIES, plt, save

COLOR = {"P1": SERIES[1], "P2": SERIES[3], "P3": SERIES[0], "P4": INK_MUTED}


def tradeoff(all_scores: dict, tau: float, path, sets=("all pairs", "borderline pairs"), limits=None) -> None:
    """Unsafe approvals vs wasted refusals as each policy's threshold moves, with the
    policies' as-specified operating points marked."""
    limits = limits or {"all pairs": (30, 40), "borderline pairs": (60, 100), "pre-flight states": (30, 40)}
    fig, axes = plt.subplots(1, len(sets), figsize=(5.5 * len(sets), 4))
    for ax, name in zip(np.atleast_1d(axes), sets):
        sc, p = all_scores[name]
        f = p["feasible"].to_numpy()
        for c in ("P1", "P2", "P3"):
            cur = W.tradeoff_curve(sc[c].to_numpy(), f)
            ax.plot(100 * cur["unsafe_approval_rate"], 100 * cur["wasted_refusal_rate"], color=COLOR[c], lw=2, label=W.POLICIES[c])
        r = W.rates(sc["P3"] >= tau, f)
        ax.scatter([100 * r["unsafe_approval_rate"]], [100 * r["wasted_refusal_rate"]], s=50, color=COLOR["P3"], zorder=5,
                   edgecolor="white", label=f"EnduroSense at tau = {tau}")
        for c, thr, lab in (("P1", 1.0, "P1 as in the brief"), ("P2", 0.0, "P2 with no margin")):
            r = W.rates(sc[c] >= thr, f)
            ax.scatter([100 * r["unsafe_approval_rate"]], [100 * r["wasted_refusal_rate"]], s=50, color=COLOR[c], zorder=5,
                       marker="s", edgecolor="white", label=lab)
        ax.set_xlim(0, limits[name][0]); ax.set_ylim(0, limits[name][1])
        ax.set_xlabel("unsafe approvals (% of missions that would fail)")
        ax.set_ylabel("wasted refusals (% of missions that would succeed)")
        ax.set_title(f"{name}: closer to the bottom-left corner is better")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    save(fig, path)


def probability_reliability(rel: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, ls="--", lw=1)
    ax.plot(rel["mean_predicted"], rel["observed_success"], "o-", color=SERIES[0], label="all pairs")
    ax.plot(rel["mean_predicted"], rel["observed_by_chain"], "o-", color=SERIES[2], label="each battery chain counted equally")
    ax.set_xlabel("predicted P(success)"); ax.set_ylabel("share that actually succeeded")
    ax.set_title("Is P(success) honest?"); ax.legend(fontsize=8)
    save(fig, path)


def fleet(table: pd.DataFrame, path, title: str = "Fleet simulation on real battery chains (blue = EnduroSense)") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    y = np.arange(len(table))
    for ax, (col, ci, name) in zip(axes, (("unsafe_per_100_missions", None, "unsafe missions per 100 flown"),
                                          ("completed_per_day", "completed_ci", "missions completed per day"),
                                          ("swaps_per_day", "swaps_ci", "battery swaps per day"))):
        ax.barh(y, table[col], xerr=table[ci] if ci else None,
                color=[SERIES[0] if p.startswith("P3") else INK_MUTED for p in table["policy"]], capsize=3)
        ax.set_yticks(y, table["policy"] if ax is axes[0] else [""] * len(y), fontsize=7)
        ax.invert_yaxis(); ax.set_title(name, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left")
    save(fig, path)
