"""Everything the dashboard computes, with no Streamlit in it (so it can be tested).

The dashboard never trains anything. It loads:

- the battery model's calibrated ranges for every reading of the unseen (test)
  batteries, exactly as scored in the final evaluation (``results/final``);
- the saved calibrated mission model, which is run live on whatever mission is
  entered;
- the final result tables.

Battery readings and recorded flights shown here are test data: batteries and
flights that no model, setting or calibration ever saw.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from endurosense import whatif as W
from endurosense.config import data_path, load_config
from endurosense.data.split import DEV, select
from endurosense.feasibility import p_success
from endurosense.mission import MissionSpec
from endurosense.uncertainty import mission as UB
from endurosense.uncertainty.quantiles import PredictiveDistribution

FINAL = data_path("results") / "final"
STATE_COLUMNS = ["flight", "battery_chain", "time", "t_chain_s", "flight_index", "v", "i", "p", "p_mean_30s", "motors_on",
                 "phase", "e_chain_wh", "e_res_wh", "remaining_wh", "remaining_min", "label_source"]


# ------------------------------------------------------------------ loading
def levels() -> np.ndarray:
    return np.array(load_config()["uncertainty"]["quantiles"])


def tau_default() -> float:
    return float(load_config()["decision"]["tau"])


def load_battery_states() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Readings of the unseen batteries and their calibrated quantiles (one row per second of recording)."""
    q = pd.read_parquet(FINAL / "model_a_test_quantiles.parquet")
    a = pd.read_parquet(data_path("features") / "model_a.parquet", columns=STATE_COLUMNS).loc[q.index]
    a = a.assign(fold=0)
    a["second"] = a.groupby("battery_chain").cumcount()            # seconds of recording on this battery, flights joined
    return a, q


def load_mission_model():
    with open(data_path("models") / "model_b" / "model_b_calibrated.pkl", "rb") as fh:
        return pickle.load(fh)


def load_recorded_missions(model, lv) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Recorded test flights as missions (single flights first, then real same-battery pairs and triples),
    their predicted quantiles, and the per-leg plan of each single flight."""
    detail = pd.read_csv(FINAL / "model_b_test_flights.csv")
    fb = pd.read_parquet(data_path("features") / "model_b_flights.parquet")
    fb = fb[fb["flight"].isin(detail["flight"])].reset_index(drop=True)
    legs = pd.read_parquet(data_path("features") / "model_b_legs.parquet")
    legs = legs[legs["flight"].isin(fb["flight"])]
    parts = UB.predicted_parts(model.model, fb, legs)
    parts = parts.add_prefix("pred_").assign(battery_chain=fb.set_index("flight").loc[parts.index, "battery_chain"])
    missions, q_b = W.build_missions(parts, fb, lv, calibration=model.tuples)
    info = detail.set_index("flight")

    def describe(flights):
        if len(flights) == 1:
            r = info.loc[flights[0]]
            return f"Flight {flights[0]}: route {r['route']}, {r['speed']:g} m/s, {r['payload']:g} g, {r['alt_cruise_m']:g} m"
        return f"{len(flights)} sorties on one battery: flights " + ", ".join(str(f) for f in flights)

    missions["label"] = [describe(f) for f in missions["flights"]]
    plan = legs.merge(fb[["flight", "ambient_wind"]].rename(columns={"ambient_wind": "wind"}), on="flight")[
        ["flight", "leg", "distance_m", "speed", "payload", "alt_cruise_m", "wind"]]
    return missions, q_b, plan


def typical_flying_power() -> float:
    """Typical power while flying, from development data (used by the minutes-left rule on the ground)."""
    dev = select(pd.read_parquet(data_path("features") / "model_a.parquet", columns=["flight", "motors_on", "p"]), DEV)
    return float(dev.loc[dev["motors_on"] == 1, "p"].median())


def phase_durations() -> dict:
    """Typical climb, descent and hover times from development flights, for estimating a planned mission's duration."""
    fb = select(pd.read_parquet(data_path("features") / "model_b_flights.parquet"), DEV)
    by_alt = fb.groupby("alt_cruise_m")[["climb_s", "descent_s"]].median()
    return {"altitude_m": by_alt.index.to_numpy(float), "climb_s": by_alt["climb_s"].to_numpy(),
            "descent_s": by_alt["descent_s"].to_numpy(), "hover_s_per_leg": float((fb["hover_s"] / fb["n_legs"]).median())}


def development_margins() -> dict:
    """Safety margins for the comparison rules, tuned on development data (never on test)."""
    m = json.loads((data_path("results") / "decisions" / "matched_margins.json").read_text())
    return {name: {float(t): v for t, v in by_tau.items()} for name, by_tau in m.items()}


# ------------------------------------------------------------------ one decision
@dataclass
class Decision:
    p_success: float
    go: bool
    available: PredictiveDistribution
    required: PredictiveDistribution
    minutes_left: float
    mission_minutes: float
    rules: pd.DataFrame            # one row per rule: what it looked at and what it says


def minutes_left(median_wh: float, state: pd.Series, typical_w: float) -> float:
    """The brief's output: energy left divided by the power being drawn now (or a typical flying power on the ground)."""
    flying = state["motors_on"] == 1 and state["p_mean_30s"] >= 100
    return float(median_wh / (state["p_mean_30s"] if flying else typical_w) * 60.0)


def mission_minutes(spec: MissionSpec, model, durations: dict) -> float:
    """Estimated airborne duration of a planned mission: leg times from the mission model, plus typical
    climb, descent and hover times for that altitude."""
    legs = sum(model.model.leg_time_s(d, spec.speed, pay, spec.wind) for d, pay in zip(spec.legs_m, spec.payload_g))
    climb = float(np.interp(spec.altitude_m, durations["altitude_m"], durations["climb_s"]))
    descent = float(np.interp(spec.altitude_m, durations["altitude_m"], durations["descent_s"]))
    return float((legs + climb + descent + durations["hover_s_per_leg"] * spec.n_legs) / 60.0)


def decide(q_available: np.ndarray, q_required: np.ndarray, state: pd.Series, duration_min: float, lv, tau: float,
           typical_w: float, margin_wh: float | None = None) -> Decision:
    """All three rules for one battery state and one mission."""
    a, b = PredictiveDistribution(lv, q_available), PredictiveDistribution(lv, q_required)
    p = p_success(a, b)
    mins = minutes_left(a.median, state, typical_w)
    rows = [{"rule": "Minutes left (the brief)", "looks at": f"{mins:.1f} min of flight left, mission takes {duration_min:.1f} min",
             "says": "GO" if mins >= duration_min else "NO-GO"},
            {"rule": "Energy, best estimates", "looks at": f"{a.median:.1f} Wh available, {b.median:.1f} Wh needed",
             "says": "GO" if a.median >= b.median else "NO-GO"}]
    if margin_wh is not None:
        rows.append({"rule": f"Energy, best estimates + {margin_wh:.0f} Wh margin", "looks at": f"{a.median - b.median:+.1f} Wh to spare",
                     "says": "GO" if a.median - b.median >= margin_wh else "NO-GO"})
    rows.append({"rule": "EnduroSense", "looks at": f"P(success) {p:.1%}, needs {tau:.0%}", "says": "GO" if p >= tau else "NO-GO"})
    return Decision(float(p), bool(p >= tau), a, b, mins, float(duration_min), pd.DataFrame(rows))


def spec_from_plan(plan: pd.DataFrame, flight: int) -> MissionSpec:
    """The plan of a recorded flight: its leg lengths, commanded speed, payload, cruise altitude and wind."""
    g = plan[plan["flight"] == flight].sort_values("leg")
    r = g.iloc[0]
    return MissionSpec.loop(g["distance_m"].tolist(), float(r["payload"]), float(r["speed"]), float(r["alt_cruise_m"]), float(r["wind"]))


# ------------------------------------------------------------------ every battery state x every recorded mission
def score_grid(states: pd.DataFrame, q_a: pd.DataFrame, missions: pd.DataFrame, q_b: np.ndarray, lv, typical_w: float) -> pd.DataFrame:
    """All combinations of the given battery states with the recorded missions, scored by every rule."""
    si, mi = (x.ravel() for x in np.meshgrid(np.arange(len(states)), np.arange(len(missions)), indexing="ij"))
    p = pd.DataFrame({"state": states.index.to_numpy()[si], "mission": mi,
                      "battery_chain": states["battery_chain"].to_numpy()[si],
                      "mission_chain": missions["battery_chain"].to_numpy()[mi],
                      "available_wh": states["remaining_wh"].to_numpy()[si], "required_wh": missions["true_wh"].to_numpy()[mi]})
    p["true_margin_wh"] = p["available_wh"] - p["required_wh"]
    p["feasible"] = p["true_margin_wh"] >= 0
    return p.join(W.policy_scores(p, states, q_a, missions, q_b, lv, pd.Series({0: typical_w})))


def three_missions(grid: pd.DataFrame, tau: float) -> pd.DataFrame | None:
    """One battery before take-off and three recorded missions that the minutes-left rule approves:
    two that EnduroSense approves and that would succeed, one that it refuses and that would fail.
    Chosen by a fixed rule from real cases: the battery with the most such refusals, the refusal with
    the median shortfall, and the smallest and largest of the approved missions."""
    ok1 = grid["P1"] >= 1.0
    trap = grid[ok1 & (grid["P3"] < tau) & ~grid["feasible"]]
    safe = grid[ok1 & (grid["P3"] >= tau) & grid["feasible"]]
    counts = trap.groupby("state").size()
    counts = counts[counts.index.isin(safe.groupby("state").size().loc[lambda s: s >= 2].index)]
    if counts.empty:
        return None
    state = counts.sort_values(ascending=False, kind="stable").index[0]
    t = trap[trap["state"] == state].sort_values("true_margin_wh")
    s = safe[safe["state"] == state].sort_values("required_wh")
    out = pd.concat([s.iloc[[0]], t.iloc[[len(t) // 2]], s.iloc[[-1]]])
    return out.assign(role=["small mission", "the one to refuse", "large mission"]).reset_index(drop=True)


def example_cases(grid_pre: pd.DataFrame, grid_all: pd.DataFrame, tau: float) -> dict:
    """Real cases to start from, each chosen by a fixed rule (the median case of its kind)."""
    def median_case(g, by):
        g = g.sort_values(by)
        return None if g.empty else g.iloc[len(g) // 2]

    go1 = grid_pre["P1"] >= 1.0
    cases = {
        "A clear go": median_case(grid_pre[go1 & (grid_pre["P3"] >= 0.999) & grid_pre["feasible"] & (grid_pre["true_margin_wh"] < 25)], "true_margin_wh"),
        "Minutes-left says go, EnduroSense says no (and it would have failed)":
            median_case(grid_pre[go1 & (grid_pre["P3"] < tau) & ~grid_pre["feasible"]], "true_margin_wh"),
        "A close call EnduroSense refuses (it would have just made it)":
            median_case(grid_pre[(grid_pre["P3"] < tau) & (grid_pre["P3"] > 0.5) & grid_pre["feasible"]], "true_margin_wh"),
        "A case EnduroSense gets wrong (approved, but it would have cut into the reserve)":
            median_case(grid_all[(grid_all["P3"] >= tau) & ~grid_all["feasible"]], "true_margin_wh"),
    }
    return {k: v for k, v in cases.items() if v is not None}


# ------------------------------------------------------------------ tables for the results page
def result_tables() -> dict:
    crit = pd.read_csv(FINAL / "success_criteria.csv")
    a = pd.read_csv(FINAL / "model_a_test_metrics.csv")
    b = pd.read_csv(FINAL / "model_b_test_metrics.csv")
    pts = pd.read_csv(FINAL / "operating_points.csv")
    return {"criteria": crit, "model_a": a, "model_a_ranges": pd.read_csv(FINAL / "model_a_test_ranges.csv"),
            "model_b": b, "model_b_ranges": pd.read_csv(FINAL / "model_b_test_ranges.csv"),
            "operating_points": pts[pts["margins"] == "tuned on development data"],
            "reliability": pd.read_csv(FINAL / "probability_reliability.csv"),
            "fleet": pd.read_csv(FINAL / "fleet_simulation.csv"),
            "latency": pd.read_csv(data_path("results") / "model_a" / "cv_metrics.csv")[["model", "latency_ms", "size_mb"]],
            "summary": json.loads((FINAL / "summary.json").read_text())}
