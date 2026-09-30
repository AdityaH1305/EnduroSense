"""Battery chains: consecutive flights flown on the same battery.

The dataset does not record which battery was used. When a flight starts at
almost the voltage the previous flight ended at, the same battery was flown
again; a larger jump means a freshly charged battery was swapped in. Linked
flights trace a battery's discharge from full towards the reserve, which gives
Model A measured "energy remaining" labels.

Rest voltages (Phase 1): voltage sags under load, so start/end voltages are
read only while the motors are off (current below ``ground_max_current_a``).
A few recordings start or end with the motors running (e.g. flight 81 ends at
~20 A), so their rest voltage is estimated from the other end of the flight
and the energy used, with a linear model fitted on flights where both ends are
measured (residual std ~0.13 V).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from endurosense.config import ROOT, load_config


def rest_voltages(flight: pd.DataFrame) -> dict:
    """Motors-off voltage just before take-off and after landing (NaN if never motors-off).

    Uses the last 10 motors-off readings of the leading run and of the trailing
    run, i.e. the readings closest to the flight itself.
    """
    off = (flight["battery_current"] < load_config()["phases"]["ground_max_current_a"]).to_numpy()
    v = flight["battery_voltage"].to_numpy()
    lead = np.cumprod(off).astype(bool)
    trail = np.cumprod(off[::-1])[::-1].astype(bool)
    return {"v_rest_start_meas": v[lead][-10:].mean() if lead.any() else np.nan,
            "v_rest_end_meas": v[trail][-10:].mean() if trail.any() else np.nan}


def fit_rest_end_model(summary: pd.DataFrame, min_energy_wh: float = 5.0) -> np.ndarray:
    """Coefficients ``(a, b, c)`` of ``v_rest_end = a + b * v_rest_start + c * energy_wh``.

    Fitted on flights with both rest voltages measured and meaningful energy use.
    """
    ok = summary.dropna(subset=["v_rest_start_meas", "v_rest_end_meas"])
    ok = ok[ok["energy_wh"] > min_energy_wh]
    X = np.c_[np.ones(len(ok)), ok["v_rest_start_meas"], ok["energy_wh"]]
    coef, *_ = np.linalg.lstsq(X, ok["v_rest_end_meas"], rcond=None)
    return coef


def fill_rest_voltages(summary: pd.DataFrame, coef: np.ndarray) -> pd.DataFrame:
    """Add ``v_rest_start`` / ``v_rest_end`` (measured, else estimated) and source flags."""
    a, b, c = coef
    out = summary.copy()
    est_end = a + b * out["v_rest_start_meas"] + c * out["energy_wh"]
    est_start = (out["v_rest_end_meas"] - a - c * out["energy_wh"]) / b
    out["v_rest_start"] = out["v_rest_start_meas"].fillna(est_start)
    out["v_rest_end"] = out["v_rest_end_meas"].fillna(est_end)
    out["v_rest_start_estimated"] = out["v_rest_start_meas"].isna()
    out["v_rest_end_estimated"] = out["v_rest_end_meas"].isna()
    return out


def load_overrides(path: str | Path | None = None) -> dict:
    """Manual chain decisions from review (see docs/chain_review.md)."""
    path = ROOT / (path or load_config()["chains"]["overrides"])
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def link_jumps(summary: pd.DataFrame, start_col: str = "v_rest_start",
               end_col: str = "v_rest_end") -> pd.Series:
    """Voltage jump from the previous same-day flight's end to this flight's start (NaN for a day's first flight)."""
    s = summary.sort_values(["date", "local_time"])
    jump = s[start_col] - s.groupby("date")[end_col].shift()
    return jump.reindex(summary.index)


def assign_battery_chains(summary: pd.DataFrame, tol_v: float | None = None,
                          start_col: str = "v_start", end_col: str = "v_end",
                          overrides: dict | None = None,
                          window: tuple[float, float] | None = None) -> pd.Series:
    """Chain id for each flight in a per-flight summary.

    ``summary`` needs ``flight``, ``date``, ``local_time`` and the two voltage
    columns. A flight continues the previous (same-day) flight's chain when the
    voltage jump ``start - previous end`` lies inside ``window = (lo, hi)``, or,
    if no window is given, when ``|jump| <= tol_v``. ``overrides`` may force a
    new chain (``break_before``) or a link (``join_to_previous``) at given
    flight ids.
    """
    overrides = overrides or {}
    s = summary.sort_values(["date", "local_time"])
    jump = s[start_col] - s[end_col].shift()
    if window is not None:
        linked = jump.between(*window)
    else:
        tol_v = load_config()["chains"]["tol_v"] if tol_v is None else tol_v
        linked = jump.abs() <= tol_v
    new_chain = (s["date"] != s["date"].shift()) | ~linked
    new_chain |= s["flight"].isin(overrides.get("break_before", []))
    new_chain &= ~s["flight"].isin(overrides.get("join_to_previous", []))
    new_chain.iloc[0] = True
    return new_chain.cumsum().reindex(summary.index)


def add_rest_after(summary: pd.DataFrame) -> pd.DataFrame:
    """``v_rest_after``: best estimate of the battery's rest voltage after each flight.

    The next flight in the same chain gives the recovered rest voltage directly.
    For the last flight of a chain, the post-landing reading is raised by the
    median recovery seen between linked flights. ``rest_after_source`` records
    which was used.
    """
    s = summary.sort_values(["date", "local_time"]).copy()
    nxt = s.groupby("battery_chain")["v_rest_start"].shift(-1)
    recovery = float((nxt - s["v_rest_end"]).median())
    s["v_rest_after"] = nxt.fillna(s["v_rest_end"] + recovery)
    s["rest_after_source"] = np.where(nxt.notna(), "next_flight", "end_plus_recovery")
    return s.reindex(summary.index)


def chain_table(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per battery chain: flights, voltages, energy and usability for Model A."""
    cfg = load_config()
    reserve = cfg["battery"]["reserve_v"]
    margin = cfg["chains"]["near_reserve_margin_v"]
    s = summary.sort_values(["date", "local_time"])
    agg = dict(
        date=("date", "first"),
        flights=("flight", list),
        n_flights=("flight", "size"),
        v_rest_first=("v_rest_start", "first"),
        v_rest_last=("v_rest_after", "last"),
        energy_wh=("energy_wh", "sum"),
    )
    if "link_confident" in s:
        agg["uncertain_links"] = ("link_confident", lambda x: int((~x.iloc[1:]).sum()))
    t = s.groupby("battery_chain").agg(**agg)
    t["reaches_reserve"] = t["v_rest_last"] <= reserve
    t["near_reserve"] = t["v_rest_last"] <= reserve + margin
    t["starts_full"] = t["v_rest_first"] >= 25.0
    return t.reset_index()


def chain_sensitivity(summary: pd.DataFrame, windows: list[list[float]]) -> pd.DataFrame:
    """How chain counts change with the linking window."""
    rows = []
    for lo, hi in windows:
        ids = assign_battery_chains(summary, start_col="v_rest_start", end_col="v_rest_end",
                                    overrides=load_overrides(), window=(lo, hi))
        s = summary.assign(battery_chain=ids)
        s = add_rest_after(s)
        t = chain_table(s)
        rows.append({"window_v": f"[{lo:+.2f}, {hi:+.2f}]", "chains": len(t),
                     "multi_flight_chains": int((t["n_flights"] > 1).sum()),
                     "near_reserve_chains": int(t["near_reserve"].sum()),
                     "linked_flights": int(t.loc[t["n_flights"] > 1, "n_flights"].sum())})
    return pd.DataFrame(rows)
