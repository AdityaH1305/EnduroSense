"""Model A data: labels and causal features for *energy available*.

Plain idea: at any moment, how many watt-hours can this battery still deliver
before it reaches the safe reserve? Battery chains (consecutive flights on one
battery, Phase 1) let us measure the answer instead of simulating it.

Label construction (per battery chain)
--------------------------------------
Rest voltages (motors off) are known before every flight and after the last
one, each at a known amount of energy drawn since the chain started:

    (E_0 = 0, V_rest_1), (E_1, V_rest_2), ..., (E_total, V_rest_after_last)

The energy at which the battery reaches the reserve voltage ``E_res`` is
found by:

- **measured**: linear interpolation between the two rest points either side
  of the reserve (chain reaches the reserve), or
- **extrapolated**: for chains ending within ``near_reserve_margin_v`` above
  the reserve, a short extrapolation along the discharge curve.

Both follow a fitted discharge curve (``DischargeCurve``: volts lost per Wh as
a function of voltage), because the curve flattens towards the reserve. The
curve is fitted on **development chains only**, so no test data shapes any
label.

Then at every reading:  ``remaining_wh = E_res - energy drawn since chain start``
(negative once below the reserve). The brief's "remaining flight time" label is
the motors-on time still to come before the reserve is reached,
``remaining_min``.

Features (all causal: computed only from readings at or before time t)
---------------------------------------------------------------------
Instantaneous voltage, current and power; look-back rolling mean/std/trend over
10 s and 30 s; an internal-resistance estimate (voltage drop per amp over the
last 30 s, a battery-health signal); voltage sag below the flight's rest
voltage; energy and time since the flight / chain started; flight index in the
chain; motors on/off. Rest voltages used as features are *causal*: a running
mean of the motors-off readings so far before take-off (frozen at take-off), or
else the previous flight's post-landing value plus the typical recovery. The
Phase 1 estimates that use a flight's own energy or later readings are never
used as features (they would leak the future).

Phase labels are deliberately not features: their smoothing looks up to 1 s
ahead. ``phase`` is kept as a metadata column for error analysis only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from endurosense.config import load_config
from endurosense.data.clean import chronological

FEATURES = [
    "v", "i", "p",
    "v_mean_10s", "v_std_10s", "v_slope_10s", "i_mean_10s", "i_std_10s",
    "v_mean_30s", "v_std_30s", "v_slope_30s", "i_mean_30s", "i_std_30s", "p_mean_30s",
    "r_int_30s", "sag_v",
    "v_rest_flight_start", "v_rest_chain_start",
    "e_flight_wh", "e_chain_wh",
    "t_flight_s", "t_chain_s", "t_motors_on_chain_s",
    "flight_index", "motors_on",
]
SEQ_CHANNELS = ["v", "i", "p", "e_chain_wh", "motors_on"]


# --------------------------------------------------------------------- labels
def rest_points(chain_flights: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Energy since chain start, rest voltage, and estimated-flag at each rest point."""
    fl = chronological(chain_flights)
    e = np.r_[0.0, fl["energy_wh"].cumsum().to_numpy()]
    v = np.r_[fl["v_rest_start"].to_numpy(), fl["v_rest_after"].iloc[-1]]
    # the final point is the last flight's post-landing rest voltage plus the typical
    # recovery; it counts as an estimate only if that post-landing voltage was estimated
    est = np.r_[fl["v_rest_start_estimated"].to_numpy(), bool(fl["v_rest_end_estimated"].iloc[-1])]
    return e, v, est


class DischargeCurve:
    """Rest voltage lost per Wh drawn, as a function of rest voltage.

    The discharge curve flattens towards the reserve (measured on development
    chains: ~0.054 V/Wh near full, ~0.031 V/Wh just above 22.6 V), so labels
    interpolate and extrapolate *along this curve* rather than along a straight
    line. slope(v) = a + b*v, clipped to the measured range.
    Validation (dev chains, leave-one-out): straight-line interpolation was
    biased by +3.3 Wh, the curve by -0.04 Wh (spread 1.5 Wh); extrapolating up to
    0.6 V errs by -0.6 +/- 2.2 Wh.
    """

    def __init__(self, a: float, b: float, clip: tuple[float, float]):
        self.a, self.b, self.clip = a, b, clip

    def slope(self, v):
        return np.clip(self.a + self.b * np.asarray(v, dtype=float), *self.clip)

    def energy_between(self, v_high: float, v_low: float, n: int = 400) -> float:
        """Wh drawn while rest voltage falls from ``v_high`` to ``v_low`` (negative if v_low > v_high)."""
        grid = np.linspace(v_low, v_high, n)
        return float(np.trapz(1.0 / self.slope(grid), grid))


def fit_discharge_curve(flights: pd.DataFrame, dev_flight_ids) -> DischargeCurve:
    """Fit slope(v) on consecutive measured rest points of **development** chains only."""
    cfg = load_config()["model_a"]
    lo, hi = cfg["slope_fit_band_v"]
    dev = flights[flights["flight"].isin(set(dev_flight_ids))]
    X, Y, W = [], [], []
    for _, g in dev.groupby("battery_chain"):
        e, v, est = rest_points(g)
        for k in range(len(e) - 1):
            de = e[k + 1] - e[k]
            if est[k] or est[k + 1] or not np.isfinite(v[k:k + 2]).all() or de < cfg["slope_min_segment_wh"]:
                continue
            vm = (v[k] + v[k + 1]) / 2
            if lo <= vm <= hi:
                X.append(vm); Y.append((v[k] - v[k + 1]) / de); W.append(de)
    b, a = np.polyfit(X, Y, 1, w=np.sqrt(W))
    return DischargeCurve(float(a), float(b), tuple(cfg["slope_clip_v_per_wh"]))


def reserve_energy(e: np.ndarray, v: np.ndarray, est: np.ndarray, reserve_v: float,
                   margin_v: float, curve: DischargeCurve) -> dict:
    """Energy since chain start at which rest voltage reaches the reserve.

    Returns ``{"e_res_wh", "source", "extrapolated_wh", "uses_estimate"}``;
    ``e_res_wh`` is NaN when the chain ends too far above the reserve.
    """
    ok = np.isfinite(v)
    e, v, est = e[ok], v[ok], est[ok]
    below = np.flatnonzero(v <= reserve_v)
    if len(below) and below[0] > 0:
        j = below[0]
        frac = curve.energy_between(v[j - 1], reserve_v) / curve.energy_between(v[j - 1], v[j])
        return {"e_res_wh": float(e[j - 1] + frac * (e[j] - e[j - 1])), "source": "measured",
                "extrapolated_wh": 0.0, "uses_estimate": bool(est[j - 1] or est[j])}
    if len(v) and not len(below) and v[-1] <= reserve_v + margin_v:
        extra = curve.energy_between(v[-1], reserve_v)
        return {"e_res_wh": float(e[-1] + extra), "source": "extrapolated",
                "extrapolated_wh": float(extra), "uses_estimate": bool(est[-1])}
    return {"e_res_wh": np.nan, "source": "none", "extrapolated_wh": np.nan, "uses_estimate": False}


def chain_labels(flights: pd.DataFrame, curve: DischargeCurve) -> pd.DataFrame:
    """Per-chain reserve energy ``e_res_wh`` and how it was obtained."""
    cfg = load_config()
    reserve, margin = cfg["battery"]["reserve_v"], cfg["chains"]["near_reserve_margin_v"]
    rows = []
    for ch, g in flights.groupby("battery_chain"):
        e, v, est = rest_points(g)
        rows.append({"battery_chain": ch, **reserve_energy(e, v, est, reserve, margin, curve)})
    return pd.DataFrame(rows).rename(columns={"source": "label_source"})


def dev_recovery_v(flights: pd.DataFrame, dev_flight_ids) -> float:
    """Typical rest-voltage recovery between linked flights, from development chains only."""
    dev = chronological(flights[flights["flight"].isin(set(dev_flight_ids))])
    nxt = dev.groupby("battery_chain")["v_rest_start_meas"].shift(-1)
    return float((nxt - dev["v_rest_end_meas"]).median())


# ------------------------------------------------------------------- features
def flight_context(flights: pd.DataFrame, recovery_v: float) -> pd.DataFrame:
    """Per flight: position in its chain, and a fallback rest voltage from the past.

    The fallback (previous flight's measured post-landing voltage plus the
    typical recovery) is used only when a recording starts with the motors
    already running, so no motors-off reading exists before take-off.
    """
    s = chronological(flights).copy()
    s["v_rest_fallback"] = s.groupby("battery_chain")["v_rest_end_meas"].shift() + recovery_v
    s["flight_index"] = s.groupby("battery_chain").cumcount()
    return s.set_index("flight")[["v_rest_fallback", "flight_index"]]


def running_rest_voltage(voltage: np.ndarray, current: np.ndarray, motors_on_a: float, n: int = 10) -> np.ndarray:
    """Causal rest voltage before take-off for one flight.

    Mean of the last ``n`` motors-off readings *so far* during the pre-flight
    idle, frozen once the motors start (it then equals the measured
    ``v_rest_start_meas``). NaN if the recording starts with the motors on.
    """
    lead = np.cumprod(current < motors_on_a).astype(bool)
    out = np.full(len(voltage), np.nan)
    if lead.any():
        out[lead] = pd.Series(voltage[lead]).rolling(n, min_periods=1).mean().to_numpy()
        out = pd.Series(out).ffill().to_numpy()
    return out


def _rolling_flight_features(f: pd.DataFrame) -> pd.DataFrame:
    """Look-back rolling features for one flight's readings (sorted by time)."""
    cfg = load_config()["model_a"]
    t = f["time"].to_numpy()
    g = pd.DataFrame({"t": t, "v": f["battery_voltage"].to_numpy(), "i": f["battery_current"].to_numpy()},
                     index=pd.to_timedelta(t, unit="s"))
    g["p"] = g["v"] * g["i"]
    g["tv"], g["tt"], g["vi"], g["ii"] = g["t"] * g["v"], g["t"] ** 2, g["v"] * g["i"], g["i"] ** 2
    out = pd.DataFrame(index=f.index)
    for w in cfg["windows_s"]:
        r = g.rolling(f"{w}s")
        m, sd = r.mean(), r.std()
        var_t = m["tt"] - m["t"] ** 2
        slope = (m["tv"] - m["t"] * m["v"]) / var_t.where(var_t > 1e-6)
        out[f"v_mean_{w}s"], out[f"v_std_{w}s"] = m["v"].to_numpy(), sd["v"].to_numpy()
        out[f"v_slope_{w}s"] = slope.to_numpy()
        out[f"i_mean_{w}s"], out[f"i_std_{w}s"] = m["i"].to_numpy(), sd["i"].to_numpy()
        if w == 30:
            out["p_mean_30s"] = m["p"].to_numpy()
    rw = cfg["r_int_window_s"]
    m = g.rolling(f"{rw}s").mean()
    var_i = m["ii"] - m["i"] ** 2
    r_int = -(m["vi"] - m["v"] * m["i"]) / var_i.where(var_i > cfg["r_int_min_current_var"])
    out[f"r_int_{rw}s"] = r_int.clip(*cfg["r_int_clip_ohm"]).to_numpy()
    return out


def build_model_a(samples: pd.DataFrame, flights: pd.DataFrame, chain_lab: pd.DataFrame,
                  recovery_v: float) -> pd.DataFrame:
    """One row per second of flight for every reading in a labelled chain."""
    cfg = load_config()
    motors_on_a = cfg["phases"]["ground_max_current_a"]
    labelled = chain_lab.loc[chain_lab["e_res_wh"].notna(), "battery_chain"]
    ctx = flight_context(flights, recovery_v)
    starts = flights.set_index("flight")["start_time"]

    parts = []
    for fid, f in samples[samples["battery_chain"].isin(labelled)].groupby("flight", sort=False):
        f = f.sort_values("time")
        feat = _rolling_flight_features(f)
        feat["flight"], feat["battery_chain"] = fid, f["battery_chain"].to_numpy()
        feat["time"], feat["phase"] = f["time"].to_numpy(), f["phase"].astype(str).to_numpy()
        feat["v"], feat["i"] = f["battery_voltage"].to_numpy(), f["battery_current"].to_numpy()
        feat["p"] = feat["v"] * feat["i"]
        feat["motors_on"] = (feat["i"] >= motors_on_a).astype(int)
        feat["v_rest_flight_start"] = running_rest_voltage(feat["v"].to_numpy(), feat["i"].to_numpy(), motors_on_a)
        feat["e_flight_wh"] = f["cum_energy_wh"].to_numpy()
        feat["e_chain_wh"] = f["cum_chain_energy_wh"].to_numpy()
        feat["t_flight_s"] = feat["time"]
        feat["wall_time"] = starts[fid] + pd.to_timedelta(feat["time"], unit="s")
        dt = np.diff(feat["time"].to_numpy(), prepend=feat["time"].iloc[0])
        feat["_on_dt"] = dt * feat["motors_on"].to_numpy()
        parts.append(feat)
    df = pd.concat(parts, ignore_index=True)
    df = df.join(ctx, on="flight")
    df["v_rest_flight_start"] = df["v_rest_flight_start"].fillna(df["v_rest_fallback"])
    df["sag_v"] = df["v_rest_flight_start"] - df["v"]

    # chain-level values, in true time order
    df = df.sort_values(["battery_chain", "wall_time"]).reset_index(drop=True)
    # rest voltage when the chain started: the first flight's running value while
    # it is still on the ground, then its final pre-take-off value
    first = df[df["flight_index"] == 0]
    final_first = first.groupby("battery_chain")["v_rest_flight_start"].last()
    df["v_rest_chain_start"] = np.where(df["flight_index"] == 0, df["v_rest_flight_start"],
                                        df["battery_chain"].map(final_first))
    chain_start = df.groupby("battery_chain")["wall_time"].transform("min")
    df["t_chain_s"] = (df["wall_time"] - chain_start).dt.total_seconds()
    df["t_motors_on_chain_s"] = df.groupby("battery_chain")["_on_dt"].cumsum()

    # labels
    lab = chain_lab.set_index("battery_chain")
    df = df.join(lab[["e_res_wh", "label_source", "extrapolated_wh", "uses_estimate"]], on="battery_chain")
    df["remaining_wh"] = df["e_res_wh"] - df["e_chain_wh"]
    df["remaining_min"] = _remaining_minutes(df)

    # downsample to sample_hz (first reading in each bin, per flight)
    hz = cfg["model_a"]["sample_hz"]
    keep = ~df.assign(_b=np.floor(df["time"] * hz)).duplicated(["flight", "_b"])
    df = df[keep].drop(columns=["_on_dt", "v_rest_fallback"]).reset_index(drop=True)
    return df


def _remaining_minutes(df: pd.DataFrame) -> np.ndarray:
    """Motors-on minutes from each reading until the chain reaches its reserve energy.

    The time at which cumulative chain energy equals ``e_res_wh`` is found by
    interpolation; beyond the chain's last reading it is extended at the
    chain's average motors-on power.
    """
    out = np.full(len(df), np.nan)
    for _, idx in df.groupby("battery_chain").indices.items():
        g = df.iloc[idx]
        e = np.maximum.accumulate(g["e_chain_wh"].to_numpy())
        t_on = g["t_motors_on_chain_s"].to_numpy()
        e_res = g["e_res_wh"].iloc[0]
        if e_res <= e[-1]:
            t_res = np.interp(e_res, e, t_on)
        else:
            avg_w = e[-1] * 3600 / t_on[-1]
            t_res = t_on[-1] + (e_res - e[-1]) * 3600 / avg_w
        out[idx] = (t_res - t_on) / 60.0
    return out


# ------------------------------------------------------------------ sequences
def build_series(samples: pd.DataFrame, chains_used) -> pd.DataFrame:
    """Per-flight sequence channels averaged into ``seq_hz`` bins (for LSTM/GRU)."""
    cfg = load_config()
    hz, on_a = cfg["model_a"]["seq_hz"], cfg["phases"]["ground_max_current_a"]
    s = samples[samples["battery_chain"].isin(set(chains_used))]
    s = s.assign(v=s["battery_voltage"], i=s["battery_current"], p=s["power_w"],
                 e_chain_wh=s["cum_chain_energy_wh"],
                 motors_on=(s["battery_current"] >= on_a).astype(float),
                 bin=np.floor(s["time"] * hz).astype(int))
    return (s.groupby(["flight", "bin"], sort=True)[SEQ_CHANNELS].mean().reset_index())


def sequence_windows(series: pd.DataFrame, rows: pd.DataFrame, seq_len: int | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Input windows for sequence models: for each row (``flight``, ``time``) the
    last ``seq_len`` *completed* bins of that flight before ``time``.

    The bin containing ``time`` is excluded because it averages readings up to
    half a second after ``time``. Bins before the flight started are filled with
    the flight's first bin and marked 0 in the returned mask.
    Returns ``X`` of shape (rows, seq_len, channels) and ``mask`` (rows, seq_len).
    """
    cfg = load_config()["model_a"]
    seq_len = seq_len or cfg["seq_len"]
    hz = cfg["seq_hz"]
    X = np.zeros((len(rows), seq_len, len(SEQ_CHANNELS)), dtype=np.float32)
    mask = np.zeros((len(rows), seq_len), dtype=np.float32)
    by_flight = {f: g for f, g in series.groupby("flight")}
    for n, (fid, t) in enumerate(zip(rows["flight"].to_numpy(), rows["time"].to_numpy())):
        g = by_flight[fid]
        vals, bins = g[SEQ_CHANNELS].to_numpy(np.float32), g["bin"].to_numpy()
        end = np.searchsorted(bins, int(np.floor(t * hz)), side="left")   # completed bins only
        win = vals[max(0, end - seq_len):end]
        k = len(win)
        if k == 0:                       # very first half-second of the flight: no history yet
            continue
        X[n, seq_len - k:] = win
        X[n, :seq_len - k] = win[0]
        mask[n, seq_len - k:] = 1.0
    return X, mask
