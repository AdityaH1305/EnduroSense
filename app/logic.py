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
def missing_inputs() -> list[str]:
    """Saved files the dashboard needs that do not exist yet (paths relative to the project)."""
    root = data_path("results").parent
    needed = [FINAL / n for n in ("model_a_test_quantiles.parquet", "model_a_test_metrics.csv", "model_a_test_ranges.csv",
                                  "model_a_test_coverage_by_chain.csv", "model_b_test_flights.csv", "model_b_test_metrics.csv",
                                  "model_b_test_ranges.csv", "success_criteria.csv", "operating_points.csv", "probability_reliability.csv",
                                  "fleet_simulation.csv", "summary.json")]
    needed += [data_path("models") / "model_b" / "model_b_calibrated.pkl", data_path("results") / "decisions" / "matched_margins.json",
               data_path("results") / "model_a" / "cv_metrics.csv"]
    needed += [data_path("features") / n for n in ("model_a.parquet", "model_b_flights.parquet", "model_b_legs.parquet")]
    return [(p.relative_to(root) if p.is_relative_to(root) else p).as_posix() for p in needed if not p.exists()]


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


# ------------------------------------------------------------------ headline facts (one source for dashboard, report and slides)
def criterion_verdicts(criteria: pd.DataFrame) -> list[str]:
    """In words: met, not met, or not met but with the target inside the sampling interval."""
    out = []
    for _, c in criteria.iterrows():
        if bool(c["met"]):
            out.append("Met")
        elif "inside both intervals: yes" in str(c["evidence"]):
            out.append("Not met, within sampling noise")
        else:
            out.append("Not met")
    return out


def one_in(rate: float) -> int:
    """A rate as "1 in N", rounded to two significant figures (12.4% -> 8, 0.61% -> 160)."""
    n = 1.0 / rate
    digits = max(0, int(np.floor(np.log10(n))) - 1)
    return int(round(n, -digits))


def _unseen_all_over_predicted() -> bool:
    d = pd.read_csv(FINAL / "model_b_test_flights.csv")
    d = d[~d["seen_route"].astype(bool)]
    return bool(len(d) and (d["predicted_wh"] > d["total_wh"]).all())


def headline_facts() -> dict:
    """Every result-dependent statement made in the dashboard, the report and the slides, read from
    the result files. If the pipeline is rerun (for example with a different reserve or threshold),
    the three deliverables follow without any text being edited."""
    cfg, t = load_config(), result_tables()
    tau = float(cfg["decision"]["tau"])
    crit, a, ar = t["criteria"], t["model_a"].set_index("model"), t["model_a_ranges"].iloc[0]
    main = "GRU ensemble, calibrated (main)"
    bm = t["model_b"][t["model_b"]["model"] == "Physics-first"].set_index("flights_group")
    br = t["model_b_ranges"].set_index("model").loc["Physics-first, calibrated: all test flights"]
    pts = t["operating_points"].assign(policy=lambda d: d["policy"].str.strip())
    allp, pre = (pts[pts["pairs"] == s].set_index("policy") for s in ("all pairs", "pre-flight states"))
    p1, p2, p3 = "P1 minutes left, as in the brief (ratio >= 1)", "P2 energy point estimates (margin >= 0 Wh)", f"P3 EnduroSense, tau = {tau}"
    with_margin = lambda table, rule: table.loc[[p for p in table.index if p.startswith(f"{rule} with margin") and f"tau = {tau} " in p][0]]
    fleet = t["fleet"].set_index("policy")
    f1, f3 = fleet.loc["P1 minutes left (brief)"], fleet.loc[f"P3 EnduroSense (tau = {tau})"]
    by_chain = pd.read_csv(FINAL / "model_a_test_coverage_by_chain.csv")
    band = t["reliability"][t["reliability"]["band"] == "(0.95, 0.99]"].iloc[0]
    verdicts = criterion_verdicts(crit)
    log = (data_path("results").parent / "docs" / "verification_log.md")
    passes = len([ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.startswith("## Pass ")]) if log.exists() else 0
    words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    return {
        "tau": tau, "reserve_v": float(cfg["battery"]["reserve_v"]), "criteria": crit, "verdicts": verdicts,
        "criteria_met": int(crit["met"].sum()), "criteria_total": int(len(crit)), "number_word": lambda n: words[n] if n < len(words) else str(n),
        "verification_passes": passes,
        "test_flights": int(t["summary"]["model_b"]["flights"]), "labelled_batteries": int(ar["chains"]), "test_readings": int(t["summary"]["model_a"]["readings"]),
        "a_main": float(a.loc[main, "test_mae_supported"]), "a_main_cv": float(a.loc[main, "cv_mae_supported"]),
        "a_lookup": float(a.loc["Voltage lookup", "test_mae_supported"]), "a_lookup_cv": float(a.loc["Voltage lookup", "cv_mae_supported"]),
        "a_counting": float(a.loc["Energy counting (BMS)", "test_mae_supported"]),
        "a_gru": float(a.loc["GRU", "test_mae_supported"]), "a_gru_cv": float(a.loc["GRU", "cv_mae_supported"]), "a_lstm_cv": float(a.loc["LSTM", "cv_mae_supported"]),
        "a_tabular_cv": (float(a.loc[["Random Forest", "Linear Regression", "XGBoost", "XGBoost + physics"], "cv_mae_supported"].min()),
                         float(a.loc[["Random Forest", "Linear Regression", "XGBoost", "XGBoost + physics"], "cv_mae_supported"].max())),
        "a_cov90": float(ar["cov90"]), "a_cov90_ci": (float(ar["cov90_ci_low"]), float(ar["cov90_ci_high"])), "a_width90": float(ar["width90"]),
        "a_truth_below": float(ar["truth_below_90_range"]),
        "a_batteries_covered": int((by_chain["coverage"] >= 0.99).sum()), "a_batteries_over_estimated": int((by_chain["truth_below_range"] > 0.05).sum()),
        "a_min_coverage_of_covered": float(by_chain.loc[by_chain["coverage"] >= 0.99, "coverage"].min()),
        "b_all": float(bm.loc["all test flights", "mape_pct"]), "b_seen": float(bm.loc["route seen in development", "mape_pct"]),
        "b_unseen": float(bm.loc["unseen routes", "mape_pct"]), "b_unseen_flights": int(bm.loc["unseen routes", "flights"]),
        "b_unseen_bias": float(bm.loc["unseen routes", "bias_wh"]), "b_unseen_mae": float(bm.loc["unseen routes", "mae_wh"]),
        "b_unseen_all_over_predicted": bool(_unseen_all_over_predicted()), "b_cov90": float(br["cov90"]),
        "unsafe_p1": float(allp.loc[p1, "unsafe_approval_rate"]), "unsafe_p2": float(allp.loc[p2, "unsafe_approval_rate"]),
        "unsafe_p3": float(allp.loc[p3, "unsafe_approval_rate"]), "unsafe_p3_takeoff": float(pre.loc[p3, "unsafe_approval_rate"]),
        "wasted_p3": float(allp.loc[p3, "wasted_refusal_rate"]), "wasted_p3_takeoff": float(pre.loc[p3, "wasted_refusal_rate"]),
        "wasted_p1_margin": float(with_margin(allp, "P1")["wasted_refusal_rate"]), "unsafe_p2_margin": float(with_margin(allp, "P2")["unsafe_approval_rate"]),
        "wasted_p2_margin": float(with_margin(allp, "P2")["wasted_refusal_rate"]), "wasted_p2_margin_takeoff": float(with_margin(pre, "P2")["wasted_refusal_rate"]),
        "one_in_p1": one_in(float(allp.loc[p1, "unsafe_approval_rate"])), "one_in_p3": one_in(float(allp.loc[p3, "unsafe_approval_rate"])),
        "fleet_p1": float(f1["unsafe_per_100_missions"]), "fleet_p3": float(f3["unsafe_per_100_missions"]),
        "fleet_ratio": int(f1["unsafe_per_100_missions"] / f3["unsafe_per_100_missions"]) if f3["unsafe_per_100_missions"] > 0 else None,
        "fleet_extra_swaps": float(f3["swaps_per_day"] / f1["swaps_per_day"] - 1), "fleet_batteries": int(f3["batteries"]),
        "fleet_p1_margin_completed": float(fleet.loc["P1 + margin tuned on development data", "completed_per_day"]),
        "band_predicted": float(band["mean_predicted"]), "band_succeeded": float(band["observed_success"]),
        "latency_ms": int(np.ceil(t["latency"].loc[t["latency"]["model"] != "Fixed capacity (reference)", "latency_ms"].max())),
    }
