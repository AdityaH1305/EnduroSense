"""Ambient wind estimation from the drone-mounted anemometer.

The anemometer rides on the drone, so it measures air moving *relative to the
drone*. Rebuilding the true wind vector during cruise was too noisy
(within-flight std ~4 m/s). Instead, ambient wind is read while the drone is
airborne but not moving sideways (climbing or descending), when relative
airflow ~= ambient wind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from endurosense.config import load_config


def ground_speed(flight: pd.DataFrame) -> pd.Series:
    """Horizontal ground speed (m/s) from the north/east velocity components."""
    return np.hypot(flight["velocity_x"], flight["velocity_y"])


def height_above_takeoff(flight: pd.DataFrame) -> pd.Series:
    """Altitude relative to the first reading (the take-off point), in metres."""
    return flight["position_z"] - flight["position_z"].iloc[0]


def _stationary_aloft(flight: pd.DataFrame) -> pd.Series:
    cfg = load_config()["wind"]
    return (ground_speed(flight) < cfg["stationary_max_ground_speed"]) & (
        height_above_takeoff(flight) > cfg["stationary_min_alt_m"]
    )


def ambient_wind(flight: pd.DataFrame) -> float:
    """Mean ambient wind speed (m/s) for one flight, or NaN if never stationary aloft."""
    stationary = _stationary_aloft(flight)
    return float(flight.loc[stationary, "wind_speed"].mean()) if stationary.any() else np.nan


def ambient_wind_stats(flight: pd.DataFrame) -> dict:
    """Ambient wind mean, spread (gustiness / estimate quality) and sample count."""
    w = flight.loc[_stationary_aloft(flight), "wind_speed"]
    return {"ambient_wind": float(w.mean()) if len(w) else np.nan,
            "ambient_wind_std": float(w.std()) if len(w) > 1 else np.nan,
            "ambient_wind_n": int(len(w))}
