"""Shared figure style so every chart in reports and slides looks consistent.

Colours are a colour-blind-checked categorical palette; series colours are
assigned in fixed order and text always uses the ink colours, never a series
colour.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PHASE_COLORS = {"ground": "#b8b6ae", "climb": SERIES[0], "cruise": SERIES[2],
                "descent": SERIES[1], "hover": SERIES[3]}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "Calibri", "font.size": 9,
        "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titlelocation": "left",
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
