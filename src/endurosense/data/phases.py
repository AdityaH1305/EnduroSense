"""Flight-phase segmentation: ground, climb, cruise, descent, hover.

Each phase draws power differently, so Model B predicts energy per phase and
assembles missions from the parts. Phases are assigned per reading from three
signals:

- height above take-off (smoothed),
- vertical speed ``velocity_z``. NOTE: in this dataset it is positive *upwards*
  (correlation with d(altitude)/dt is +0.77), despite the README describing
  a north-east-down frame,
- horizontal ground speed,
- battery current: with the motors off (~0 A) the drone must be on the ground.
  This matters after landing, where the altitude reading drifts up to ~14 m.

Fragments shorter than ``min_duration_s`` are merged into the preceding phase
so brief sensor wobbles do not create spurious phase changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from endurosense.config import load_config

GROUND, CLIMB, CRUISE, DESCENT, HOVER = "ground", "climb", "cruise", "descent", "hover"
PHASES = [GROUND, CLIMB, CRUISE, DESCENT, HOVER]


def raw_phase(alt_rel: np.ndarray, vz_up: np.ndarray, ground_speed: np.ndarray,
              current_a: np.ndarray) -> np.ndarray:
    """Per-reading phase from instantaneous signals, before smoothing."""
    cfg = load_config()["phases"]
    vmin = cfg["vertical_min_speed"]
    return np.select(
        [(alt_rel < cfg["ground_max_alt_m"]) | (current_a < cfg["ground_max_current_a"]),
         ground_speed >= cfg["move_min_ground_speed"],
         vz_up > vmin,
         vz_up < -vmin],
        [GROUND, CRUISE, CLIMB, DESCENT],
        default=HOVER,
    )


def merge_short_runs(labels: np.ndarray, time_s: np.ndarray, min_duration_s: float) -> np.ndarray:
    """Relabel runs shorter than ``min_duration_s`` with the previous run's label.

    The first run is kept as-is (there is nothing before it). Repeats until no
    short run remains, since merging can join neighbours into longer runs.
    """
    labels = labels.copy()
    while True:
        change = np.r_[True, labels[1:] != labels[:-1]]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(labels)]
        # a run lasts from its first reading until the next run starts
        stop_times = np.r_[time_s[starts[1:]], time_s[-1]]
        durations = stop_times - time_s[starts]
        short = [(s, e) for k, (s, e) in enumerate(zip(starts, ends))
                 if k > 0 and durations[k] < min_duration_s]
        if not short:
            return labels
        for s, e in short:
            labels[s:e] = labels[s - 1]


def segment_flight(flight: pd.DataFrame) -> pd.Series:
    """Phase label for every reading of one flight (sorted by time)."""
    cfg = load_config()["phases"]
    z = flight["position_z"].rolling(5, center=True, min_periods=1).median()
    alt_rel = (z - z.iloc[:5].median()).to_numpy()
    gs = np.hypot(flight["velocity_x"], flight["velocity_y"]).to_numpy()
    labels = raw_phase(alt_rel, flight["velocity_z"].to_numpy(), gs, flight["battery_current"].to_numpy())
    labels = merge_short_runs(labels, flight["time"].to_numpy(), cfg["min_duration_s"])
    return pd.Series(labels, index=flight.index, name="phase")


def phase_runs(labels: pd.Series) -> list[str]:
    """Compressed sequence of phases, e.g. ['ground', 'climb', 'cruise', 'descent', 'ground']."""
    vals = labels.to_numpy()
    keep = np.r_[True, vals[1:] != vals[:-1]]
    return vals[keep].tolist()


def check_phase_order(runs: list[str]) -> list[str]:
    """Problems with a cruise flight's phase sequence (empty list = looks right).

    Expected shape: starts and ends on the ground, climbs first, descends last
    and includes cruise. Hover may appear anywhere in the air.
    """
    problems = []
    if runs[0] != GROUND:
        problems.append("does not start on ground")
    if runs[-1] != GROUND:
        problems.append("does not end on ground")
    air = [r for r in runs if r not in (GROUND, HOVER)]
    if not air:
        return problems + ["never airborne"]
    if air[0] != CLIMB:
        problems.append(f"first airborne phase is {air[0]}")
    if air[-1] != DESCENT:
        problems.append(f"last airborne phase is {air[-1]}")
    if CRUISE not in air:
        problems.append("no cruise")
    if runs.count(GROUND) > 2:
        problems.append("touches ground mid-flight")
    return problems
