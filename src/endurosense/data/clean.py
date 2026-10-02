"""Cleaning and validating the raw sensor readings.

What profiling showed (Phase 1 investigation):
- No missing values and no out-of-order timestamps; 10 gaps of 1-5 s.
- Large voltage steps are real load changes: they mirror current steps
  (correlation -0.93), so they are kept.
- A handful of readings (~76 of 258k) are glitches that a load change does not
  explain; these are replaced by the local median and flagged.
- ``altitude`` mixes numbers with the label ``"25-50-100-25"`` for flights that
  change altitude en route; it is parsed into numeric columns.
- ``local_time`` is text like ``"9:22"``; sorting it as text puts 9:22 after
  10:05, so chronological order must use ``start_time()``. Four start times are
  mis-logged in the raw data and corrected from ``config/time_corrections.yaml``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from endurosense.config import ROOT, load_config


def load_time_corrections() -> dict:
    """flight id -> correction entry (``date``, ``from``, ``to``, ``reason``)."""
    path = ROOT / load_config()["cleaning"]["time_corrections"]
    with open(path, encoding="utf-8") as fh:
        return {int(k): v for k, v in (yaml.safe_load(fh) or {}).get("corrections", {}).items()}


def correct_start_times(df: pd.DataFrame, time_col: str = "local_time") -> pd.DataFrame:
    """Apply the documented start-time corrections to any frame with ``flight`` and a time column.

    Each correction checks that the raw value still equals its ``from`` value,
    so a changed dataset cannot be silently mis-corrected.
    """
    out = df.copy()
    for fid, c in load_time_corrections().items():
        rows = out["flight"] == fid
        if not rows.any():
            continue
        found = set(out.loc[rows, time_col].astype(str))
        if found != {c["from"]} or set(out.loc[rows, "date"]) != {c["date"]}:
            raise ValueError(f"flight {fid}: expected {c['date']} {c['from']}, found {found}")
        out.loc[rows, time_col] = c["to"]
    return out


def start_time(df: pd.DataFrame, time_col: str = "local_time") -> pd.Series:
    """Flight start as a real timestamp (``date`` + ``local_time``), for correct ordering."""
    return pd.to_datetime(df["date"].astype(str) + " " + df[time_col].astype(str), format="%Y-%m-%d %H:%M")


def chronological(df: pd.DataFrame, time_col: str = "local_time") -> pd.DataFrame:
    """Rows in true time order (ties broken by flight id). Never sort ``local_time`` as text."""
    key = start_time(df, time_col)
    return df.assign(_t=key).sort_values(["_t", "flight"], kind="stable").drop(columns="_t")


def parse_altitude(label: pd.Series) -> pd.DataFrame:
    """Planned altitude label -> ``alt_cruise_m`` (highest planned level) and ``varying_altitude``."""
    label = label.astype(str)
    levels = label.str.split("-")
    return pd.DataFrame({
        "altitude_label": label,
        "alt_cruise_m": levels.map(lambda xs: max(float(x) for x in xs)),
        "varying_altitude": levels.map(len) > 1,
    }, index=label.index)


def _rolling_median(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df.groupby("flight")[col].transform(
        lambda s: s.rolling(window, center=True, min_periods=1).median())


def fix_glitches(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sensor glitches in voltage and current with the local median.

    A voltage glitch is a jump away from the local median with no matching
    current change (a real load change moves both); a current glitch is a jump
    with no voltage response. Adds boolean flags ``v_glitch`` and ``i_glitch``.
    """
    cfg = load_config()["cleaning"]
    w = cfg["spike_window"]
    v_med = _rolling_median(df, "battery_voltage", w)
    i_med = _rolling_median(df, "battery_current", w)
    dv = (df["battery_voltage"] - v_med).abs()
    di = (df["battery_current"] - i_med).abs()
    out = df.copy()
    out["v_glitch"] = (dv > cfg["voltage_spike_v"]) & (di < cfg["voltage_spike_current_a"])
    out["i_glitch"] = (di > cfg["current_spike_a"]) & (dv < cfg["current_spike_voltage_v"])
    out.loc[out["v_glitch"], "battery_voltage"] = v_med[out["v_glitch"]]
    out.loc[out["i_glitch"], "battery_current"] = i_med[out["i_glitch"]]
    return out


def clean_flights(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sort, type, validate and de-glitch the raw readings.

    Returns the cleaned frame and a report dict of what was found and changed.
    Raises ``ValueError`` if a check that should never fail does (e.g. voltage
    outside physical bounds), because later phases rely on these guarantees.
    """
    cfg = load_config()["cleaning"]
    df = raw.sort_values(["flight", "time"], kind="stable").reset_index(drop=True)
    df = df.rename(columns={"time_day": "local_time"})
    df = correct_start_times(df)
    df = pd.concat([df.drop(columns="altitude"), parse_altitude(df["altitude"])], axis=1)
    df["route"] = df["route"].astype("category")

    dt = df.groupby("flight")["time"].diff()
    if (dt <= 0).any():
        raise ValueError(f"{int((dt <= 0).sum())} non-increasing timestamps")
    df["gap_before_s"] = dt.where(dt > cfg["gap_s"], 0.0).fillna(0.0)

    vlo, vhi = cfg["voltage_bounds_v"]
    ilo, ihi = cfg["current_bounds_a"]
    bad_v = ~df["battery_voltage"].between(vlo, vhi)
    bad_i = ~df["battery_current"].between(ilo, ihi)
    if bad_v.any() or bad_i.any():
        raise ValueError(f"out-of-bounds readings: voltage {int(bad_v.sum())}, current {int(bad_i.sum())}")
    if df.isna().any().any():
        raise ValueError("missing values in raw data")

    df = fix_glitches(df)
    no_gps = df.groupby("flight")["position_z"].transform(lambda s: (s == 0).all())

    report = {
        "rows": len(df),
        "flights": df["flight"].nunique(),
        "gaps_over_1s": int((df["gap_before_s"] > 0).sum()),
        "largest_gap_s": float(df["gap_before_s"].max()),
        "voltage_glitches_fixed": int(df["v_glitch"].sum()),
        "current_glitches_fixed": int(df["i_glitch"].sum()),
        "negative_current_rows": int((df["battery_current"] < 0).sum()),
        "min_current_a": float(df["battery_current"].min()),
        "flights_without_gps": sorted(df.loc[no_gps, "flight"].unique().tolist()),
        "start_times_corrected": {fid: f"{c['from']} -> {c['to']}" for fid, c in load_time_corrections().items()},
    }
    return df, report
