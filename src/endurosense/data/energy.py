"""Electrical energy drawn from the battery.

Energy is power integrated over time: E = sum(V * I * dt). Watt-hours are used
throughout because they do not depend on what the drone does next, unlike
"minutes remaining".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from endurosense.config import load_config

EARTH_RADIUS_M = 6_371_000.0


def energy_wh(time_s: np.ndarray, voltage_v: np.ndarray, current_a: np.ndarray) -> float:
    """Total energy (Wh) drawn over one recording.

    Each reading's power is held until the next timestamp; the first reading
    contributes zero because it has no preceding interval.
    """
    t = np.asarray(time_s, dtype=float)
    dt = np.diff(t, prepend=t[0])
    return float(np.sum(np.asarray(voltage_v) * np.asarray(current_a) * dt) / 3600.0)


def step_distance_m(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Horizontal distance (m) between consecutive GPS fixes; first element is 0.

    Equirectangular approximation, accurate to well under 0.1% over the
    sub-kilometre steps in this dataset.
    """
    lon, lat = np.deg2rad(np.asarray(lon_deg)), np.deg2rad(np.asarray(lat_deg))
    dx = np.diff(lon, prepend=lon[0]) * np.cos(lat) * EARTH_RADIUS_M
    dy = np.diff(lat, prepend=lat[0]) * EARTH_RADIUS_M
    return np.hypot(dx, dy)


def add_energy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-reading power, energy and distance columns (``df`` sorted by flight, time).

    - ``power_w``: V x I
    - ``energy_wh``: energy of the interval ending at this reading (same rule as
      ``energy_wh()``, so per-reading values sum to the flight total). Recording
      gaps are integrated across rather than dropped, so no energy goes missing.
    - ``cum_energy_wh``: running total within the flight
    - ``step_dist_m`` / ``cum_dist_m``: horizontal distance flown; GPS movement
      below ``distance_min_ground_speed`` is treated as position noise
    """
    min_gs = load_config()["phases"]["distance_min_ground_speed"]
    out = df.copy()
    dt = out.groupby("flight")["time"].diff().fillna(0.0)
    out["power_w"] = out["battery_voltage"] * out["battery_current"]
    out["energy_wh"] = out["power_w"] * dt / 3600.0
    out["cum_energy_wh"] = out.groupby("flight")["energy_wh"].cumsum()

    step = np.zeros(len(out))
    for _, idx in out.groupby("flight").indices.items():
        f = out.iloc[idx]
        step[idx] = step_distance_m(f["position_x"].to_numpy(), f["position_y"].to_numpy())
    moving = np.hypot(out["velocity_x"], out["velocity_y"]) >= min_gs
    out["step_dist_m"] = np.where(moving, step, 0.0)
    out["cum_dist_m"] = out.groupby("flight")["step_dist_m"].cumsum()
    return out
