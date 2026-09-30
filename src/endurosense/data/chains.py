"""Battery chains: consecutive flights flown on the same battery.

The dataset does not record which battery was used. When a flight starts at
almost the voltage the previous flight ended at, the same battery was flown
again; a larger jump means a freshly charged battery was swapped in. Linked
flights trace a battery's discharge from full towards the reserve, which gives
Model A measured "energy remaining" labels.
"""
from __future__ import annotations

import pandas as pd

from endurosense.config import load_config


def assign_battery_chains(summary: pd.DataFrame, tol_v: float | None = None) -> pd.Series:
    """Chain id for each flight in a per-flight summary.

    ``summary`` needs ``date``, ``local_time``, ``v_start`` and ``v_end``. A
    flight continues the previous (same-day) flight's chain when
    ``|v_start - previous v_end| <= tol_v``.
    """
    tol_v = load_config()["chains"]["tol_v"] if tol_v is None else tol_v
    s = summary.sort_values(["date", "local_time"])
    new_chain = (s["date"] != s["date"].shift()) | ((s["v_start"] - s["v_end"].shift()).abs() > tol_v)
    return new_chain.cumsum().reindex(summary.index)
