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
        assert any(v.value.startswith(("GO", "NO-GO")) for v in list(at.success) + list(at.error))
    at.selectbox[0].set_value(start.options[0]).run()
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
