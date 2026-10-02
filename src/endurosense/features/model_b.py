"""Model B data: targets and planning-time features for *energy required*.

Plain idea: a mission's energy is the sum of its parts - climb to altitude,
fly each leg, pause between legs, descend, plus a little on the ground. Each
part is learned from the recorded flights and the mission total is assembled
from them (``endurosense.mission``), so missions of any shape and length can be
estimated even though nearly every recorded flight uses the same route.

Data facts this design rests on (Phase 1/2 profiling):
- Every route is a closed loop: R1 has 3 straight cruise legs with short
  hovers between them, ends ~2 m from take-off and goes ~140 m from home.
- Cruise power depends mostly on payload, barely on speed (4-10 m/s).
- Measured leg time exceeds distance / commanded speed because the drone
  accelerates and brakes on every leg, so leg *time* is a learned target.

Only planning-time inputs are features: commanded speed, payload, cruise
altitude, ambient wind, and the leg distance. Measured airspeed or
acceleration are never used: they are not known before the mission flies.

Excluded flights: the truncated landing (flight 250) and, by config, the two
flights that change altitude during cruise (their climb is not a single phase).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from endurosense.config import load_config
from endurosense.data.phases import PHASES

PLAN_FEATURES = ["speed", "payload", "alt_cruise_m", "ambient_wind"]
LEG_FEATURES = PLAN_FEATURES + ["distance_m"]


def eligible_flights(flights: pd.DataFrame) -> pd.DataFrame:
    """Cruise-route flights usable for Model B targets."""
    ok = flights["route"].str.startswith("R") & flights["phase_order_ok"]
    if load_config()["model_b"]["exclude_varying_altitude"]:
        ok &= ~flights["varying_altitude"]
    return flights[ok]


def _runs(phase: np.ndarray) -> list[tuple[int, int]]:
    starts = np.r_[0, np.flatnonzero(phase[1:] != phase[:-1]) + 1]
    return list(zip(starts, np.r_[starts[1:], len(phase)]))


def leg_table(samples: pd.DataFrame, flights: pd.DataFrame) -> pd.DataFrame:
    """One row per cruise leg (a cruise run of at least ``min_leg_s``)."""
    min_s = load_config()["model_b"]["min_leg_s"]
    fl = eligible_flights(flights).set_index("flight")
    rows = []
    for fid, f in samples[samples["flight"].isin(fl.index)].groupby("flight", sort=True):
        f = f.sort_values("time")
        ph, t = f["phase"].astype(str).to_numpy(), f["time"].to_numpy()
        e, d = f["energy_wh"].to_numpy(), f["step_dist_m"].to_numpy()
        dt = np.diff(t, prepend=t[0])
        leg = 0
        for a, b in _runs(ph):
            if ph[a] != "cruise":
                continue
            # same convention as the Phase 1 phase totals: each reading owns the
            # interval ending at it, so legs + fragments add up to cruise exactly
            dur = dt[a:b].sum()
            if dur < min_s:
                continue
            rows.append({"flight": fid, "leg": leg, "duration_s": dur,
                         "distance_m": d[a:b].sum(), "energy_wh": e[a:b].sum()})
            leg += 1
    legs = pd.DataFrame(rows)
    legs["n_legs"] = legs.groupby("flight")["leg"].transform("size")
    legs = legs.join(fl[PLAN_FEATURES + ["battery_chain"]], on="flight")
    legs["power_w"] = legs["energy_wh"] * 3600 / legs["duration_s"]
    legs["ground_speed"] = legs["distance_m"] / legs["duration_s"]
    legs["overhead_s"] = legs["duration_s"] - legs["distance_m"] / legs["speed"]
    # a few legs (e.g. flight 2's first) reposition faster than the commanded speed;
    # they are kept for energy accounting but flagged so leg-time models skip them
    legs["at_commanded_speed"] = legs["ground_speed"] <= load_config()["model_b"]["max_speed_ratio"] * legs["speed"]
    return legs


def flight_targets(flights: pd.DataFrame, legs: pd.DataFrame) -> pd.DataFrame:
    """Per-flight phase targets plus an exact energy breakdown.

    ``other_wh`` is cruise energy not inside a leg (sub-``min_leg_s`` fragments),
    so that climb + legs + hover + other + descent + ground == total energy.
    """
    fl = eligible_flights(flights)
    leg_sum = legs.groupby("flight").agg(legs_wh=("energy_wh", "sum"), legs_s=("duration_s", "sum"),
                                         legs_m=("distance_m", "sum"), n_legs=("leg", "size"))
    t = fl[["flight", "battery_chain", *PLAN_FEATURES]].join(leg_sum, on="flight")
    for p in PHASES:
        t[f"{p}_wh"] = fl[f"{p}_energy_wh"].to_numpy()
        t[f"{p}_s"] = fl[f"{p}_time_s"].to_numpy()
    t["other_wh"] = t["cruise_wh"] - t["legs_wh"]
    t["total_wh"] = fl["energy_wh"].to_numpy()
    return t.reset_index(drop=True)
