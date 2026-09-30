"""Per-flight summary table used for profiling and chain reconstruction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from endurosense.config import load_config
from endurosense.data.chains import assign_battery_chains
from endurosense.data.energy import energy_wh
from endurosense.data.wind import ambient_wind, ground_speed, height_above_takeoff


def summarise_flight(flight: pd.DataFrame) -> dict:
    """Key statistics for one flight (readings sorted by time)."""
    cfg = load_config()
    t = flight["time"].to_numpy()
    v = flight["battery_voltage"].to_numpy()
    i = flight["battery_current"].to_numpy()
    e_wh = energy_wh(t, v, i)
    gs = ground_speed(flight)
    alt_rel = height_above_takeoff(flight)
    airborne = alt_rel > cfg["wind"]["airborne_min_alt_m"]
    return {
        "ambient_wind": ambient_wind(flight),
        "duration_s": t[-1] - t[0],
        "median_dt_s": np.median(np.diff(t)),
        "v_start": v[:10].mean(),
        "v_end": v[-10:].mean(),
        "v_min": v.min(),
        "i_mean_air": i[airborne].mean() if airborne.any() else np.nan,
        "i_max": i.max(),
        "energy_wh": e_wh,
        "pct_pack": 100 * e_wh / cfg["battery"]["pack_wh"],
        "airspeed_mean": flight.loc[airborne, "wind_speed"].mean() if airborne.any() else np.nan,
        "ground_speed_max": gs.max(),
        "alt_max_rel": alt_rel.max(),
    }


def build_flight_summary(flights: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    """One row per flight: planned parameters, summary statistics and battery chain id."""
    rows = [{"flight": fid, **summarise_flight(f.sort_values("time"))}
            for fid, f in flights.groupby("flight", sort=True)]
    summary = params.merge(pd.DataFrame(rows), on="flight")
    summary["battery_chain"] = assign_battery_chains(summary)
    return summary
