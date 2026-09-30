"""Phase 1 pipeline: raw readings -> cleaned per-reading, per-flight and per-chain tables.

    raw CSV -> clean (types, gaps, glitches) -> flight phases -> energy & distance
            -> per-flight table (+ rest voltages, wind) -> battery chains

Outputs (written by ``scripts/02_prepare_data.py``):
    data/processed/samples.parquet  one row per reading
    data/processed/flights.parquet  one row per flight
    data/processed/chains.parquet   one row per battery chain
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from endurosense.config import load_config
from endurosense.data import chains as ch
from endurosense.data.clean import clean_flights
from endurosense.data.energy import add_energy_columns
from endurosense.data.phases import PHASES, check_phase_order, phase_runs, segment_flight
from endurosense.data.wind import ambient_wind_stats

FLIGHT_PARAM_COLS = ["route", "speed", "payload", "altitude_label", "alt_cruise_m",
                     "varying_altitude", "date", "local_time"]


def add_phases(df: pd.DataFrame) -> pd.DataFrame:
    """Per-reading ``phase`` and height above take-off ``alt_rel_m``."""
    out = df.copy()
    out["phase"] = pd.concat([segment_flight(f) for _, f in out.groupby("flight", sort=False)])
    z0 = out.groupby("flight")["position_z"].transform(lambda s: s.iloc[:5].median())
    out["alt_rel_m"] = out["position_z"] - z0
    out["phase"] = pd.Categorical(out["phase"], categories=PHASES)
    return out


def summarise(flight: pd.DataFrame) -> dict:
    """Per-flight statistics from the cleaned, phased, energy-annotated readings."""
    is_cruise_route = str(flight["route"].iloc[0]).startswith("R")
    runs = phase_runs(flight["phase"].astype(str))
    problems = check_phase_order(runs) if is_cruise_route else []
    dt = flight["time"].diff().fillna(0.0)
    row = {
        "duration_s": flight["time"].iloc[-1] - flight["time"].iloc[0],
        "energy_wh": flight["energy_wh"].sum(),
        "distance_m": flight["cum_dist_m"].iloc[-1],
        "alt_max_m": flight["alt_rel_m"].max(),
        "phase_sequence": " > ".join(runs),
        "phase_order_ok": not problems,
        "phase_problems": "; ".join(problems),
        "truncated_landing": is_cruise_route and runs[-1] != "ground",
        "glitches_fixed": int(flight["v_glitch"].sum() + flight["i_glitch"].sum()),
        **ambient_wind_stats(flight),
        **ch.rest_voltages(flight),
    }
    for p in PHASES:
        m = (flight["phase"] == p).to_numpy()
        row[f"{p}_energy_wh"] = flight["energy_wh"].to_numpy()[m].sum()
        row[f"{p}_time_s"] = dt.to_numpy()[m].sum()
    cruise = flight["phase"] == "cruise"
    row["cruise_distance_m"] = flight.loc[cruise, "step_dist_m"].sum()
    row["cruise_power_w"] = (flight.loc[cruise, "energy_wh"].sum() * 3600 / row["cruise_time_s"]
                             if row["cruise_time_s"] > 0 else np.nan)
    row["cruise_ground_speed"] = (row["cruise_distance_m"] / row["cruise_time_s"]
                                  if row["cruise_time_s"] > 0 else np.nan)
    return row


def build_tables(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Run the whole Phase 1 pipeline. Returns ``samples, flights, chains, report``."""
    samples, report = clean_flights(raw)
    samples = add_energy_columns(add_phases(samples))

    rows = []
    for fid, f in samples.groupby("flight", sort=True):
        rows.append({"flight": fid, **f[FLIGHT_PARAM_COLS].iloc[0].to_dict(), **summarise(f)})
    flights = pd.DataFrame(rows)
    flights["route"] = flights["route"].astype(str)

    cfg = load_config()["chains"]
    coef = ch.fit_rest_end_model(flights)
    flights = ch.fill_rest_voltages(flights, coef)
    overrides = ch.load_overrides()
    flights["battery_chain"] = ch.assign_battery_chains(
        flights, start_col="v_rest_start", end_col="v_rest_end", overrides=overrides,
        window=tuple(cfg["link_window_v"]))
    flights["link_jump_v"] = ch.link_jumps(flights)
    first_in_chain = ~flights.sort_values(["date", "local_time"])["battery_chain"].duplicated().reindex(flights.index)
    flights["link_confident"] = first_in_chain | flights["link_jump_v"].between(*cfg["confident_window_v"])
    flights = ch.add_rest_after(flights)
    chains = ch.chain_table(flights)

    # energy drawn from the battery since the start of its chain, per reading
    prior = (flights.sort_values(["date", "local_time"])
             .assign(prior_chain_wh=lambda d: d.groupby("battery_chain")["energy_wh"].cumsum() - d["energy_wh"])
             .set_index("flight")[["battery_chain", "prior_chain_wh"]])
    samples = samples.join(prior, on="flight")
    samples["cum_chain_energy_wh"] = samples["prior_chain_wh"] + samples["cum_energy_wh"]
    samples = samples.drop(columns="prior_chain_wh")

    report.update({
        "rest_end_model": {"a": coef[0], "b": coef[1], "c": coef[2]},
        "median_recovery_v": float((flights["v_rest_after"] - flights["v_rest_end"])[
            flights["rest_after_source"] == "next_flight"].median()),
        "rest_start_estimated": flights.loc[flights["v_rest_start_estimated"], "flight"].tolist(),
        "rest_end_estimated": flights.loc[flights["v_rest_end_estimated"], "flight"].tolist(),
        "phase_order_problems": flights.loc[~flights["phase_order_ok"], "flight"].tolist(),
        "truncated_landing": flights.loc[flights["truncated_landing"], "flight"].tolist(),
        "overrides": overrides,
        "uncertain_links": flights.loc[~flights["link_confident"], ["flight", "link_jump_v"]].round(3).values.tolist(),
        "sensitivity": ch.chain_sensitivity(flights, cfg["sensitivity_windows_v"]),
    })
    return samples, flights, chains, report
