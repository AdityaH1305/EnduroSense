"""Electrical energy drawn from the battery.

Energy is power integrated over time: E = sum(V * I * dt). Watt-hours are used
throughout because they do not depend on what the drone does next, unlike
"minutes remaining".
"""
from __future__ import annotations

import numpy as np


def energy_wh(time_s: np.ndarray, voltage_v: np.ndarray, current_a: np.ndarray) -> float:
    """Total energy (Wh) drawn over one recording.

    Each reading's power is held until the next timestamp; the first reading
    contributes zero because it has no preceding interval.
    """
    t = np.asarray(time_s, dtype=float)
    dt = np.diff(t, prepend=t[0])
    return float(np.sum(np.asarray(voltage_v) * np.asarray(current_a) * dt) / 3600.0)
