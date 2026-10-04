"""The dashboard: its logic module, its charts and every page (run headlessly)."""
import sys

import numpy as np
import pandas as pd
import pytest

from endurosense import whatif as W
from endurosense.config import ROOT, data_path
from endurosense.mission import MissionSpec

sys.path.insert(0, str(ROOT / "app"))
FINAL = data_path("results") / "final"
pytestmark = [pytest.mark.data, pytest.mark.skipif(not (FINAL / "model_a_test_quantiles.parquet").exists(),
                                                   reason="run scripts/run_all.py --final first")]


@pytest.fixture(scope="module")
def world():
    import logic as L
    lv = L.levels()
    a, q = L.load_battery_states()
    model = L.load_mission_model()
    missions, q_b, plan = L.load_recorded_missions(model, lv)
    tw = L.typical_flying_power()
    pre = W.preflight_states(a)
    return {"L": L, "lv": lv, "a": a, "q": q, "model": model, "missions": missions, "q_b": q_b, "plan": plan, "tw": tw,
            "pre": pre, "grid": L.score_grid(pre, q, missions, q_b, lv, tw), "durations": L.phase_durations()}


def test_dashboard_scores_equal_the_final_evaluation(world):
    saved = pd.read_parquet(FINAL / "whatif_pairs_preflight.parquet").drop_duplicates(["state", "mission"])
    m = world["grid"].merge(saved, on=["state", "mission"], suffixes=("", "_saved"))
    assert len(m) > 2000
    for col in ("P1", "P2", "P3", "required_wh", "available_wh"):
        assert np.allclose(m[col], m[col + "_saved"]), col
    assert len(world["grid"]) == len(world["pre"]) * len(world["missions"])          # every battery x every mission, once


def test_three_missions_are_what_the_page_says_they_are(world):
    L, g = world["L"], world["grid"]
    tm = L.three_missions(g, 0.95)
    assert tm is not None and tm["role"].tolist() == ["small mission", "the one to refuse", "large mission"]
    assert tm["state"].nunique() == 1 and tm["mission"].nunique() == 3 and (tm["P1"] >= 1).all()   # one battery; minutes-left approves all
    assert (tm["P3"] >= 0.95).tolist() == [True, False, True] and tm["feasible"].tolist() == [True, False, True]
    assert tm["required_wh"].iloc[0] < tm["required_wh"].iloc[2]
    assert L.three_missions(g, 0.95).equals(tm)                                         # a fixed rule, not a random pick
    assert L.three_missions(g[g["P3"] > 2], 0.95) is None                              # nothing to show: says so instead of failing


def test_example_cases_match_their_descriptions(world):
    L = world["L"]
    cases = L.example_cases(world["grid"], world["grid"], 0.95)
    go = cases["A clear go"]
    assert go["P3"] >= 0.999 and go["feasible"] and go["P1"] >= 1
    trap = cases["Minutes-left says go, EnduroSense says no (and it would have failed)"]
    assert trap["P1"] >= 1 and trap["P3"] < 0.95 and not trap["feasible"]
    close = cases["A close call EnduroSense refuses (it would have just made it)"]
    assert close["P3"] < 0.95 and close["feasible"]


def test_one_decision_reports_every_rule_consistently(world):
    L, lv, model = world["L"], world["lv"], world["model"]
    state = world["pre"].iloc[0]
    spec = MissionSpec.delivery(300, 500, 8, 50)
    minutes = L.mission_minutes(spec, model, world["durations"])
    d = L.decide(world["q"].loc[state.name].to_numpy(), model.distribution(spec).values, state, minutes, lv, 0.95, world["tw"], margin_wh=7.0)
    says = d.rules.set_index("rule")["says"]
    assert says["Minutes left (the brief)"] == ("GO" if d.minutes_left >= minutes else "NO-GO")
    assert says["Energy, best estimates"] == ("GO" if d.available.median >= d.required.median else "NO-GO")
    assert says["Energy, best estimates + 7 Wh margin"] == ("GO" if d.available.median - d.required.median >= 7 else "NO-GO")
    assert says["EnduroSense"] == ("GO" if d.go else "NO-GO") and d.go == (d.p_success >= 0.95)
    assert L.decide(world["q"].loc[state.name].to_numpy(), model.distribution(spec).values, state, minutes, lv, 1.01, world["tw"]).go is False
    assert d.minutes_left == pytest.approx(d.available.median / world["tw"] * 60)      # on the ground: typical flying power


def test_harder_missions_are_less_likely_to_succeed(world):
    L, lv, model = world["L"], world["lv"], world["model"]
    q_row = world["q"].loc[world["pre"].index[0]].to_numpy()
    state = world["pre"].iloc[0]
    p = lambda spec: L.decide(q_row, model.distribution(spec).values, state, 3.0, lv, 0.95, world["tw"]).p_success
    by_distance = [p(MissionSpec.delivery(d, 500, 8, 50)) for d in (100, 300, 600, 1200, 2400)]
    by_payload = [p(MissionSpec.delivery(900, w, 8, 50)) for w in (0, 250, 500, 750)]
    assert all(x >= y - 1e-9 for x, y in zip(by_distance, by_distance[1:])) and by_distance[0] > 0.9 and by_distance[-1] < 0.1
    assert all(x >= y - 1e-9 for x, y in zip(by_payload, by_payload[1:]))
    mins = [L.mission_minutes(MissionSpec.delivery(d, 500, 8, 50), model, world["durations"]) for d in (100, 300, 600)]
    assert mins[0] < mins[1] < mins[2] and 0.5 < mins[0] < 5


def test_recorded_flight_plans_reproduce_the_saved_predictions(world):
    L, lv = world["L"], world["lv"]
    singles = world["missions"][world["missions"]["sorties"] == 1]
    for i in list(singles.index[:5]) + [singles.index[-1]]:
        spec = L.spec_from_plan(world["plan"], singles.loc[i, "flights"][0])
        assert world["model"].distribution(spec).median == pytest.approx(float(np.interp(0.5, lv, world["q_b"][i])), abs=0.05)
    assert world["missions"]["label"].str.len().min() > 10


def test_charts_build_valid_specifications(world):
    import charts as C
    from endurosense.uncertainty.quantiles import PredictiveDistribution
    L, lv, a, q = world["L"], world["lv"], world["a"], world["q"]
    d1, d2 = PredictiveDistribution(lv, q.iloc[0].to_numpy()), PredictiveDistribution(lv, world["q_b"][0])
    g = a[a["battery_chain"] == a["battery_chain"].iloc[0]]
    qs = q.loc[g.index].to_numpy()
    chain = g[["second", "remaining_wh"]].assign(q05=qs[:, 2], q50=qs[:, 11], q95=qs[:, -3])
    t = L.result_tables()
    for chart in (C.range_chart(d1, d2), C.range_chart(d1, d2, {"available": 30.0, "required": None}, domain=(0, 80)),
                  C.battery_timeline(chain, 100, True), C.battery_timeline(chain, 100, False),
                  C.policy_bars(t["fleet"], "unsafe_per_100_missions", "unsafe"), C.reliability_chart(t["reliability"]), C.cv_vs_test(t["model_a"])):
        spec = chart.to_dict()                                                           # raises if the specification is invalid
        assert "layer" in spec or "mark" in spec


def test_every_page_runs_and_responds(world):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=300).run()
    assert not at.exception and at.title[0].value == "Three missions, one battery"
    at.sidebar.toggle[0].set_value(True).run()                                          # reveal what was measured
    assert not at.exception and any("into the reserve" in m.value for m in at.markdown)
    at.sidebar.slider[0].set_value(0.99).run()                                          # a stricter threshold still renders
    assert not at.exception
    at.sidebar.slider[0].set_value(0.95).run()

    at.sidebar.radio[0].set_value("Mission check").run()
    assert not at.exception and at.metric[-1].label == "P(success)"
    start = at.selectbox[0]
    for option in start.options[1:]:                                                    # every prepared example
        at.selectbox[0].set_value(option).run()
        assert not at.exception, option
        go, no_go = [v.value for v in at.success], [v.value for v in at.error]
        assert all(v.startswith("GO: ") for v in go) and all(v.startswith("NO-GO: ") for v in no_go) and len(go) + len(no_go) == 1
        if option.startswith("A clear go") or "gets wrong" in option:
            assert go, option                                                           # EnduroSense approved
        else:
            assert no_go, option                                                        # EnduroSense refused
    at.selectbox[0].set_value(start.options[0]).run()
    # choosing another take-off shows that battery state, not the first one
    pre = world["pre"]
    takeoff = next(s for s in at.selectbox if s.label == "Which take-off")
    chain = next(s for s in at.selectbox if s.label == "Unseen battery (test set)").value
    flights = pre.loc[pre["battery_chain"] == chain, "flight"].tolist()
    assert len(flights) >= 2
    takeoff.set_value(flights[-1]).run()
    volts = next(m for m in at.metric if m.label == "Battery voltage now").value
    assert volts == f"{pre.loc[pre['flight'] == flights[-1], 'v'].iloc[0]:.2f} V" != f"{pre.loc[pre['flight'] == flights[0], 'v'].iloc[0]:.2f} V"
    radio = lambda label: next(r for r in at.radio if r.label == label)
    radio("Moment").set_value("Any moment of the recording").run()
    assert not at.exception
    radio("Mission").set_value("A recorded flight").run()
    assert not at.exception and any("What really happened" in m.value for m in list(at.info) + list(at.warning))

    at.sidebar.radio[0].set_value("Fleet").run()
    assert not at.exception
    at.slider[-1].set_value(10).run()                                                   # 10 simulated days keeps the test quick
    at.button[0].click().run()
    assert not at.exception and len(at.dataframe) >= 2

    for page in ("Results", "How it works"):
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page


def test_minutes_left_uses_current_power_in_the_air_and_typical_power_on_the_ground(world):
    L = world["L"]
    ground = pd.Series({"motors_on": 0, "p_mean_30s": 5.0})
    flying = pd.Series({"motors_on": 1, "p_mean_30s": 300.0})
    idle = pd.Series({"motors_on": 1, "p_mean_30s": 40.0})                               # motors on but not yet flying
    assert L.minutes_left(30.0, ground, 600.0) == pytest.approx(3.0)                     # 30 Wh at 600 W
    assert L.minutes_left(30.0, flying, 600.0) == pytest.approx(6.0)                     # 30 Wh at 300 W
    assert L.minutes_left(30.0, idle, 600.0) == pytest.approx(3.0)
    assert 300 < world["tw"] < 800                                                       # a flying power, not a ground one


def test_planned_mission_duration_adds_legs_climb_descent_and_hover(world):
    L, model, dur = world["L"], world["model"], world["durations"]
    spec = MissionSpec.loop([200, 200, 200], 250, 8, 50)
    legs = sum(model.model.leg_time_s(200, 8, 250, 3.0) for _ in range(3))
    climb, descent = (float(np.interp(50, dur["altitude_m"], dur[k])) for k in ("climb_s", "descent_s"))
    assert L.mission_minutes(spec, model, dur) == pytest.approx((legs + climb + descent + 3 * dur["hover_s_per_leg"]) / 60)
    assert 0.5 < dur["hover_s_per_leg"] < 6 and (np.diff(dur["climb_s"]) > 0).all()      # per leg, and higher takes longer
    est = np.array([L.mission_minutes(L.spec_from_plan(world["plan"], f[0]), model, dur) for f in world["missions"]["flights"][:40]])
    true = world["missions"]["duration_min"].to_numpy()[:40]
    assert np.mean(np.abs(est - true)) < 0.15                                            # close to the real airborne time of recorded flights


def test_battery_seconds_run_through_the_whole_battery_and_states_carry_the_truth(world):
    a = world["a"]
    for _, g in a.groupby("battery_chain"):
        assert g["second"].tolist() == list(range(len(g)))                               # one battery, flights joined end to end
    assert a["flight"].nunique() > a["battery_chain"].nunique()
    g = world["grid"]
    assert np.allclose(g["available_wh"], a.loc[g["state"], "remaining_wh"]) and (g["feasible"] == (g["available_wh"] >= g["required_wh"])).all()
    assert 0.2 < g["feasible"].mean() < 0.9


def test_the_refused_mission_is_the_median_case_not_the_most_dramatic(world):
    L, g = world["L"], world["grid"]
    tm = L.three_missions(g, 0.95)
    state = tm["state"].iloc[0]
    trap = g[(g["state"] == state) & (g["P1"] >= 1) & (g["P3"] < 0.95) & ~g["feasible"]].sort_values("true_margin_wh")
    assert tm["true_margin_wh"].iloc[1] == trap["true_margin_wh"].iloc[len(trap) // 2]
    counts = g[(g["P1"] >= 1) & (g["P3"] < 0.95) & ~g["feasible"]].groupby("state").size()
    assert len(trap) == counts.max()                                                     # the battery with the most such refusals


def test_headline_facts_are_read_from_the_result_files(world):
    L = world["L"]
    f = L.headline_facts()
    crit = pd.read_csv(FINAL / "success_criteria.csv")
    assert f["criteria_met"] == int(crit["met"].sum()) and f["criteria_total"] == len(crit) and len(f["verdicts"]) == len(crit)
    a = pd.read_csv(FINAL / "model_a_test_metrics.csv").set_index("model")
    assert f["a_main"] == a.loc["GRU ensemble, calibrated (main)", "test_mae_supported"] and f["a_gru_cv"] == a.loc["GRU", "cv_mae_supported"]
    by_chain = pd.read_csv(FINAL / "model_a_test_coverage_by_chain.csv")
    assert f["a_batteries_covered"] + f["a_batteries_over_estimated"] == len(by_chain) == f["labelled_batteries"]
    assert f["a_batteries_covered"] == int((by_chain["coverage"] >= 0.99).sum()) < len(by_chain)
    assert f["one_in_p1"] == L.one_in(f["unsafe_p1"]) and f["unsafe_p3"] < f["unsafe_p2"] < f["unsafe_p1"]
    log = (ROOT / "docs" / "verification_log.md").read_text(encoding="utf-8")
    assert f["verification_passes"] == log.count("\n## Pass ") >= 5
    assert f["number_word"](2) == "two" and f["number_word"](11) == "11"


def test_verdict_words_and_one_in():
    import logic as L
    crit = pd.DataFrame({"met": [True, False, False], "evidence": ["x", "A 80%; 90% inside both intervals: yes", "B; 90% inside both intervals: no"]})
    assert L.criterion_verdicts(crit) == ["Met", "Not met, within sampling noise", "Not met"]
    assert [L.one_in(r) for r in (0.1236, 0.0061, 0.5, 0.00102, 0.031)] == [8, 160, 2, 980, 32]


def test_missing_inputs_are_named(monkeypatch, tmp_path):
    import logic as L
    assert L.missing_inputs() == []                                                      # everything is built on this machine
    monkeypatch.setattr(L, "FINAL", tmp_path)                                            # as on a fresh clone before run_all --final
    missing = L.missing_inputs()
    assert len(missing) == 12 and all(isinstance(m, str) for m in missing)


def test_margin_rule_and_close_call_selection():
    import logic as L
    from scipy.stats import norm
    lv = L.levels()
    q = lambda mu, sd: mu + sd * norm.ppf(lv)
    state = pd.Series({"motors_on": 0, "p_mean_30s": 5.0})
    d = L.decide(q(30.0, 3.0), q(26.0, 1.0), state, 3.0, lv, 0.95, 500.0, margin_wh=7.0)     # 4 Wh to spare: less than the margin
    says = d.rules.set_index("rule")["says"]
    assert says["Energy, best estimates"] == "GO" and says["Energy, best estimates + 7 Wh margin"] == "NO-GO"
    assert L.decide(q(30.0, 3.0), q(22.0, 1.0), state, 3.0, lv, 0.95, 500.0, margin_wh=7.0).rules.set_index("rule")["says"]["Energy, best estimates + 7 Wh margin"] == "GO"
    # the "close call it refuses" example must be a mission that would have succeeded, even when most refused ones would not
    grid = pd.DataFrame({"state": range(7), "mission": range(7), "P1": 2.0, "P3": [0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.99],
                         "feasible": [False, False, False, False, True, False, True], "true_margin_wh": [-5.0, -4, -3, -2, 1.0, -1, 9]})
    case = L.example_cases(grid, grid, 0.95)["A close call EnduroSense refuses (it would have just made it)"]
    assert case["feasible"] and case["true_margin_wh"] == 1.0
    assert not L.example_cases(grid, grid, 0.95)["Minutes-left says go, EnduroSense says no (and it would have failed)"]["feasible"]


def test_range_chart_puts_each_range_on_its_own_row(world):
    import charts as C
    from endurosense.uncertainty.quantiles import PredictiveDistribution
    lv = world["lv"]
    battery, mission = PredictiveDistribution(lv, world["q"].iloc[0].to_numpy()), PredictiveDistribution(lv, world["q_b"][0])
    data = C.range_chart(battery, mission).layer[0].data
    row = lambda what: data[(data["what"] == what) & (data["range"] == "90% range")].iloc[0]
    assert row("Battery can give")["from"] == pytest.approx(battery.quantile(0.05)) and row("Battery can give")["to"] == pytest.approx(battery.quantile(0.95))
    assert row("Mission needs")["from"] == pytest.approx(mission.quantile(0.05)) and row("Mission needs")["to"] == pytest.approx(mission.quantile(0.95))


def test_fleet_facts_are_ratios_of_the_fleet_table(world):
    f = world["L"].headline_facts()
    fleet = pd.read_csv(FINAL / "fleet_simulation.csv").set_index("policy")
    p1, p3 = fleet.loc["P1 minutes left (brief)"], fleet.loc[f"P3 EnduroSense (tau = {f['tau']})"]
    assert f["fleet_extra_swaps"] == pytest.approx(p3["swaps_per_day"] / p1["swaps_per_day"] - 1) and f["fleet_extra_swaps"] > 0
    assert f["fleet_ratio"] == int(p1["unsafe_per_100_missions"] / p3["unsafe_per_100_missions"])



def test_policy_charts_label_every_row_and_leave_room_for_the_values(world):
    import charts as C
    names = ["P1 minutes left (brief)", "P2 energy, point estimates", "P1 + margin tuned on development data", "P2 + 7 Wh margin",
             "   P2 with margin tuned on development data for tau = 0.95 (threshold 6.95)", "P2 energy point estimates (margin >= 0 Wh)",
             "P3 EnduroSense (tau = 0.95)", "P4 oracle"]
    assert [C.short_policy(n) for n in names] == ["Minutes left (the brief)", "Energy, best estimates", "Minutes left + margin", "Best estimates + margin",
                                                  "Best estimates + margin", "Energy, best estimates", "EnduroSense", "Oracle (knows the truth)"]
    fleet = world["L"].result_tables()["fleet"]
    chart = C.policy_bars(fleet, "swaps_per_day", "battery swaps per day")
    spec, data = chart.to_dict(), chart.data                                                # both layers share one table
    y = spec["layer"][0]["encoding"]["y"]
    assert y["field"] == "rule" and y["axis"]["labelOverlap"] is False                     # no row label is dropped
    assert data["rule"].is_unique and len(data) == len(fleet)                               # one distinct label per policy
    assert spec["layer"][0]["encoding"]["x"]["scale"]["domain"][1] > fleet["swaps_per_day"].max() * 1.1   # room for the value labels
    assert spec["height"] >= 32 * len(fleet) + 80                                            # tall enough for every row plus legend and axis
    assert (data.loc[data["kind"] == "EnduroSense", "rule"] == "EnduroSense").all() and (data["kind"] == "EnduroSense").sum() == 1
