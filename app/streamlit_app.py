"""EnduroSense dashboard.

Run:  streamlit run app/streamlit_app.py

Nothing is trained here. The dashboard loads the saved models and the final
results, and shows decisions for batteries and flights the models never saw.
"""
import numpy as np
import pandas as pd
import streamlit as st

import charts as C
import logic as L
from endurosense import whatif as W
from endurosense.data.load import load_processed
from endurosense.mission import MissionSpec
from endurosense.scheduler import FleetSimulator, batteries_from, compare_policies, task_pool

st.set_page_config(page_title="EnduroSense", layout="wide")
PAGES = ["Three missions, one battery", "Mission check", "Fleet", "Results", "How it works"]
GO, NOGO = ":material/check_circle:", ":material/block:"


# ------------------------------------------------------------------ cached data
@st.cache_resource(show_spinner="Loading saved models and final results...")
def world() -> dict:
    lv = L.levels()
    a, q = L.load_battery_states()
    model = L.load_mission_model()
    missions, q_b, plan = L.load_recorded_missions(model, lv)
    tw = L.typical_flying_power()
    pre = W.preflight_states(a)
    return {"lv": lv, "a": a, "q": q, "model": model, "missions": missions, "q_b": q_b, "plan": plan, "tw": tw,
            "durations": L.phase_durations(), "margins": L.development_margins(), "pre": pre,
            "grid_pre": L.score_grid(pre, q, missions, q_b, lv, tw), "tables": L.result_tables()}


@st.cache_resource(show_spinner="Scoring every battery state against every recorded mission...")
def grid_all() -> pd.DataFrame:
    w = world()
    return L.score_grid(W.decision_states(w["a"], 10), w["q"], w["missions"], w["q_b"], w["lv"], w["tw"])


@st.cache_data(show_spinner="Simulating the fleet...")
def fleet_run(tau: float, drones: int, tasks: int, days: int) -> pd.DataFrame:
    w = world()
    full = load_processed("chains").set_index("battery_chain")["starts_full"]
    bats = batteries_from(w["a"][w["a"]["battery_chain"].map(full).to_numpy()])
    sim = FleetSimulator(bats, w["a"], w["q"], w["missions"], w["q_b"], w["lv"], pd.Series({0: w["tw"]}), drones)
    m = w["margins"]["all pairs"][0.95]
    rules = {"P1 minutes left (the brief)": ("P1", 1.0), "P2 energy, best estimates": ("P2", 0.0),
             f"P2 + {m['P2']:.0f} Wh margin": ("P2", m["P2"]), f"P3 EnduroSense (tau = {tau:.2f})": ("P3", tau), "P4 oracle (knows the truth)": ("P4", 0.0)}
    return compare_policies(sim, rules, days, tasks, task_pool(w["missions"], 0.3, 42), 42)


# ------------------------------------------------------------------ small pieces
def verdict(go: bool, text: str) -> None:
    (st.success if go else st.error)(("GO: " if go else "NO-GO: ") + text, icon=GO if go else NOGO)


def battery_card(state: pd.Series, q_row: np.ndarray, lv, reveal: bool) -> None:
    lo, mid, hi = (float(np.interp(x, lv, q_row)) for x in (0.05, 0.5, 0.95))
    c = st.columns(4)
    c[0].metric("Battery voltage now", f"{state['v']:.2f} V")
    c[1].metric("Energy drawn so far", f"{state['e_chain_wh']:.1f} Wh")
    c[2].metric("Energy left above reserve", f"{mid:.1f} Wh", help="The battery model's best estimate.")
    c[3].metric("90% range", f"{lo:.0f} to {hi:.0f} Wh")
    if reveal:
        st.caption(f"Measured afterwards: {state['remaining_wh']:.1f} Wh were really left above the reserve.")


def decision_block(d: L.Decision, tau: float, truth: dict | None, reveal: bool) -> None:
    left, right = st.columns([1, 2])
    with left:
        st.metric("P(success)", f"{d.p_success:.1%}", help="Chance that the battery's energy above the reserve covers the mission.")
        verdict(d.go, f"needs {tau:.0%} to approve")
    with right:
        st.altair_chart(C.range_chart(d.available, d.required, truth if reveal else None), width="stretch")
        st.caption("Darker = more likely. The black tick is the best estimate"
                   + ("; the white diamond is what was measured." if reveal and truth else "."))
    st.dataframe(d.rules.rename(columns={"rule": "Rule", "looks at": "What it looks at", "says": "Decision"}), hide_index=True, width="stretch")
    if reveal and truth and truth.get("required") is not None:
        margin = truth["available"] - truth["required"]
        if margin >= 0:
            st.info(f"What really happened: the battery had {truth['available']:.1f} Wh and the mission took {truth['required']:.1f} Wh. "
                    f"It would have **succeeded** with {margin:.1f} Wh to spare.", icon=":material/task_alt:")
        else:
            st.warning(f"What really happened: the battery had {truth['available']:.1f} Wh and the mission took {truth['required']:.1f} Wh. "
                       f"It would have **cut {-margin:.1f} Wh into the reserve**.", icon=":material/warning:")


# ------------------------------------------------------------------ pages
def page_three(tau: float, reveal: bool) -> None:
    w = world()
    st.title("Three missions, one battery")
    st.write("One battery from the unseen test set, just before take-off, and three missions that were really flown. "
             "The rule in the brief, *minutes of flight left ≥ mission duration*, approves all three. EnduroSense asks a different "
             "question: **what is the chance this battery has the energy this mission needs?**")
    tm = L.three_missions(w["grid_pre"], tau)
    if tm is None:
        st.info("At this threshold there is no battery with two approved missions and one refused one. Move τ back towards 0.95.")
        return
    state = w["a"].loc[tm["state"].iloc[0]]
    st.subheader(f"The battery: chain {int(state['battery_chain'])}, before flight {int(state['flight'])}")
    battery_card(state, w["q"].loc[state.name].to_numpy(), w["lv"], reveal)
    q_state = w["q"].loc[state.name].to_numpy()
    wh = np.r_[q_state, w["q_b"][tm["mission"].to_numpy(int)].ravel()]
    axis = (max(0.0, 5 * np.floor(wh.min() / 5)), 5 * np.ceil(wh.max() / 5))          # one Wh axis for all three charts
    cols = st.columns(3)
    for col, (_, r) in zip(cols, tm.iterrows()):
        m = w["missions"].iloc[int(r["mission"])]
        d = L.decide(w["q"].loc[state.name].to_numpy(), w["q_b"][int(r["mission"])], state, m["duration_min"], w["lv"], tau, w["tw"])
        with col:
            st.markdown(f"**{r['role'].capitalize()}**")
            st.caption(m["label"])
            st.metric("P(success)", f"{d.p_success:.0%}")
            st.write(f"Minutes-left rule: **GO** ({d.minutes_left:.1f} min left, mission {d.mission_minutes:.1f} min)")
            verdict(d.go, "EnduroSense")
            st.altair_chart(C.range_chart(d.available, d.required,
                                          {"available": state["remaining_wh"], "required": m["true_wh"]} if reveal else None,
                                          height=150, domain=axis),
                            width="stretch")
            if reveal:
                margin = state["remaining_wh"] - m["true_wh"]
                st.write(f"Really needed {m['true_wh']:.1f} Wh of the {state['remaining_wh']:.1f} Wh available: "
                         + (f"**fine**, {margin:.1f} Wh to spare." if margin >= 0 else f"**{-margin:.1f} Wh into the reserve.**"))
    if not reveal:
        st.caption("Turn on *Show what was measured* in the sidebar to see how each mission really turned out.")
    st.divider()
    st.caption("These cases are picked by a fixed rule from real test data: the battery with the most missions that minutes-left approves and "
               "EnduroSense refuses, the refused mission with the median shortfall, and the smallest and largest approved missions.")


def page_check(tau: float, reveal: bool) -> None:
    w = world()
    a, q, lv, missions = w["a"], w["q"], w["lv"], w["missions"]
    st.title("Mission check")
    st.write("Pick a battery and a mission. The battery model says how much energy is left above the reserve; the mission model says "
             "how much the mission needs; the two ranges give the chance of success.")
    cases = L.example_cases(w["grid_pre"], grid_all(), tau)
    start = st.selectbox("Start from", ["My own battery and mission"] + list(cases))
    truth = None
    if start != "My own battery and mission":
        case = cases[start]
        state, mi = a.loc[int(case["state"])], int(case["mission"])
        mission = missions.iloc[mi]
        st.subheader(f"Battery: chain {int(state['battery_chain'])}, flight {int(state['flight'])}, {state['time']:.0f} s into the recording")
        battery_card(state, q.loc[state.name].to_numpy(), lv, reveal)
        st.subheader("Mission: " + mission["label"])
        q_req, duration, truth = w["q_b"][mi], mission["duration_min"], {"available": state["remaining_wh"], "required": mission["true_wh"]}
        warnings = []
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("Battery")
            chain = st.selectbox("Unseen battery (test set)", sorted(a["battery_chain"].unique()), format_func=lambda c: f"Battery chain {c}")
            g = a[a["battery_chain"] == chain]
            when = st.radio("Moment", ["Before a take-off", "Any moment of the recording"], horizontal=True)
            if when == "Before a take-off":
                pre = w["pre"][w["pre"]["battery_chain"] == chain]
                flight = st.selectbox("Which take-off", pre["flight"].tolist(), format_func=lambda f: f"Flight {f}")
                state = pre[pre["flight"] == flight].iloc[0]
            else:
                sec = st.slider("Seconds of recording on this battery", 0, int(g["second"].max()), int(g["second"].max() // 2), step=5)
                state = g[g["second"] == sec].iloc[0]
        with right:
            st.subheader("Mission")
            kind = st.radio("Mission", ["Plan my own", "A recorded flight"], horizontal=True)
            if kind == "Plan my own":
                shape = st.selectbox("Type", ["Delivery: out with the payload, back empty", "Loop: same payload all the way"])
                c1, c2 = st.columns(2)
                dist = c1.slider("Distance per leg (m)", 50, 600, 300, step=10)
                n_legs = c2.slider("Legs", 2, 5, 3) if shape.startswith("Loop") else 2
                payload = c1.slider("Payload (g)", 0, 1000, 500, step=50)
                speed = c2.slider("Cruise speed (m/s)", 2, 15, 8)
                altitude = c1.slider("Cruise altitude (m)", 10, 150, 50, step=5)
                wind = c2.slider("Expected wind (m/s)", 0.0, 10.0, 3.0, step=0.5)
                spec = (MissionSpec.delivery(dist, payload, speed, altitude, wind) if shape.startswith("Delivery")
                        else MissionSpec.loop([dist] * n_legs, payload, speed, altitude, wind))
                q_req, duration, warnings = w["model"].distribution(spec).values, L.mission_minutes(spec, w["model"], w["durations"]), w["model"].warnings(spec)
            else:
                mi = st.selectbox("Recorded mission (test set)", range(len(missions)), format_func=lambda i: missions["label"].iloc[i])
                q_req, duration, warnings = w["q_b"][mi], missions["duration_min"].iloc[mi], []
                truth = {"available": state["remaining_wh"], "required": missions["true_wh"].iloc[mi]}
        battery_card(state, q.loc[state.name].to_numpy(), lv, reveal)
        if truth is None:
            truth = {"available": state["remaining_wh"], "required": None}
    for msg in warnings:
        st.warning("Outside what the mission model was tested on: " + msg, icon=":material/warning:")

    st.subheader("Decision")
    margin = w["margins"]["pre-flight states" if state["motors_on"] == 0 else "all pairs"][0.95]["P2"]
    d = L.decide(q.loc[state.name].to_numpy(), q_req, state, duration, lv, tau, w["tw"], margin)
    decision_block(d, tau, truth, reveal)

    with st.expander("This battery over its whole recording"):
        g = a[a["battery_chain"] == state["battery_chain"]]
        qs = q.loc[g.index].to_numpy()
        chain_df = g[["second", "remaining_wh"]].assign(q05=np.array([np.interp(0.05, lv, r) for r in qs]),
                                                        q50=np.array([np.interp(0.5, lv, r) for r in qs]),
                                                        q95=np.array([np.interp(0.95, lv, r) for r in qs]))
        st.altair_chart(C.battery_timeline(chain_df, int(state["second"]), reveal), width="stretch")
        st.caption("Blue band: the battery model's 90% range at every second. Zero is the reserve (22.6 V at rest), not an empty battery.")


def page_fleet(tau: float, reveal: bool) -> None:
    w = world()
    st.title("Fleet")
    st.write("A small fleet works through a day's queue of missions. Each drone carries a real unseen battery; each task is a real recorded "
             "flight or a pair of sorties. A rule decides which drone may take a task and when a battery must be swapped. "
             "An **unsafe mission** is one that drew more than the battery really had above its reserve.")
    t = w["tables"]["fleet"]
    st.subheader(f"Final evaluation: {int(t['batteries'].iloc[0])} unseen batteries, 4 drones, 40 tasks a day, 200 days")
    c1, c2 = st.columns(2)
    c1.altair_chart(C.policy_bars(t, "unsafe_per_100_missions", "unsafe missions per 100 flown", fmt=".2f"), width="stretch")
    c2.altair_chart(C.policy_bars(t, "swaps_per_day", "battery swaps per day"), width="stretch")
    st.caption("The margins for P1 and P2 were tuned on development data and applied here unchanged. "
               "With its margin, the minutes-left rule is safe only because it refuses almost everything (11 of 40 tasks a day completed).")
    with st.expander("Table"):
        st.dataframe(t[["policy", "unsafe_per_100_missions", "completed_per_day", "dropped_per_day", "swaps_per_day", "energy_left_at_swap_wh"]]
                     .round(2), hide_index=True, width="stretch")

    st.subheader("Try other settings")
    c = st.columns(3)
    drones = c[0].slider("Drones", 2, 6, 4)
    tasks = c[1].slider("Tasks per day", 20, 60, 40, step=5)
    days = c[2].slider("Simulated days", 10, 60, 30, step=10)
    if st.button("Run the simulation", type="primary"):
        r = fleet_run(float(tau), drones, tasks, days)
        c1, c2 = st.columns(2)
        c1.altair_chart(C.policy_bars(r, "unsafe_per_100_missions", "unsafe missions per 100 flown", fmt=".2f", height=200), width="stretch")
        c2.altair_chart(C.policy_bars(r, "completed_per_day", "missions completed per day", height=200), width="stretch")
        st.dataframe(r[["policy", "unsafe_per_100_missions", "completed_per_day", "dropped_per_day", "swaps_per_day"]].round(2),
                     hide_index=True, width="stretch")
        st.caption(f"EnduroSense uses the threshold in the sidebar (τ = {tau:.2f}). Fewer days means noisier numbers.")


def page_results(tau: float, reveal: bool) -> None:
    t = world()["tables"]
    st.title("Results on the locked test set")
    s = t["summary"]
    st.write(f"{s['model_b']['flights']} flights and {s['model_a']['chains']} labelled batteries that no model, setting or calibration ever saw. "
             "What would count as success was written down before the test set was opened. Two of the four criteria were met.")
    crit = t["criteria"].assign(result=np.where(t["criteria"]["met"], "✔ Met", "✘ Not met"))
    st.dataframe(crit[["criterion", "result", "evidence"]].rename(columns=str.capitalize), hide_index=True, width="stretch")

    a = t["model_a"].set_index("model")
    pts = t["operating_points"]
    p = pts[pts["pairs"] == "all pairs"].set_index("policy")["unsafe_approval_rate"]
    b = t["model_b"][t["model_b"]["model"] == "Physics-first"].set_index("flights_group")["mape_pct"]
    c = st.columns(4)
    c[0].metric("Energy available: error", f"{a.loc['GRU ensemble, calibrated (main)', 'test_mae_supported']:.2f} Wh",
                f"{a.loc['GRU ensemble, calibrated (main)', 'vs_lookup_diff']:.2f} Wh vs best non-ML", delta_color="inverse")
    c[1].metric("Energy required: error", f"{b['all test flights']:.1f}%", f"{b['unseen routes']:.1f}% on unseen routes", delta_color="off")
    c[2].metric("Unsafe approvals, minutes-left", f"{p['P1 minutes left, as in the brief (ratio >= 1)']:.1%}")
    c[3].metric("Unsafe approvals, EnduroSense", f"{p['P3 EnduroSense, tau = 0.95']:.2%}")

    st.subheader("Energy available (Model A)")
    st.altair_chart(C.cv_vs_test(t["model_a"]), width="stretch")
    r = t["model_a_ranges"].iloc[0]
    st.write(f"The main model's 90% range covered **{r['cov90']:.1%}** of test readings (target 88–92%; with {int(r['chains'])} batteries "
             f"the plausible band is {r['cov90_ci_low']:.0%}–{r['cov90_ci_high']:.0%}) and is {r['width90']:.1f} Wh wide on average. "
             "A single GRU, the best model in cross-validation, was no better than the baseline on test; the five-network ensemble was.")

    st.subheader("Energy required (Model B)")
    mb = t["model_b"][t["model_b"]["model"] == "Physics-first"][["flights_group", "flights", "mape_pct", "mae_wh", "bias_wh"]]
    st.dataframe(mb.rename(columns={"flights_group": "Flights", "flights": "n", "mape_pct": "Error (%)", "mae_wh": "Error (Wh)", "bias_wh": "Bias (Wh)"}).round(2),
                 hide_index=True, width="stretch")
    rb = t["model_b_ranges"].iloc[0]
    st.write(f"Its 90% range covered only **{rb['cov90']:.0%}** of test flights: too narrow on new data. On unseen, longer routes the model "
             "over-predicts (the safe direction) and misses the 5% target.")

    st.subheader("Decisions")
    c1, c2 = st.columns(2)
    op = pts[(pts["pairs"] == "all pairs") & pts["policy"].str.contains("brief|margin >= 0|tau = 0.95|oracle")].copy()
    op["policy"] = op["policy"].str.strip()
    c1.altair_chart(C.policy_bars(op.assign(pct=op["unsafe_approval_rate"] * 100), "pct", "unsafe approvals (% of missions that would fail)",
                                  fmt=".2f", height=220), width="stretch")
    c2.altair_chart(C.reliability_chart(t["reliability"]), width="stretch")
    c2.caption("Is P(success) honest? On the dashed line, predicted and actual agree. Missions rated 95–99% succeeded 94% of the time: slightly over-confident.")

    st.subheader("What did not hold up")
    st.markdown("- **Mission model on unseen routes:** 5.7% error against a 5% target (over-predicted).\n"
                "- **Mission model's ranges:** 80% coverage on test against a 90% target.\n"
                "- **P(success) at the top:** slightly over-confident, because two batteries were over-estimated.\n"
                "- **Probability vs a tuned margin:** best estimates plus a well-chosen fixed margin did as well over all cases. "
                "The probability refused fewer feasible missions at take-off (25% against 32%) and needed no tuning.")
    st.subheader("Speed and size")
    st.dataframe(t["latency"].rename(columns={"model": "Model", "latency_ms": "One prediction (ms, laptop CPU, one thread)", "size_mb": "Size (MB)"}).round(2),
                 hide_index=True, width="stretch")
    st.caption("Full details: docs/results_summary.md. Every number here is read from results/final.")


def page_how(tau: float, reveal: bool) -> None:
    st.title("How it works")
    st.markdown("""
**The brief** asks for the remaining flight time of a UAV battery. Minutes are a weak basis for a go / no-go decision: the same battery
lasts very different times depending on what it is asked to do next.

**EnduroSense** predicts two energies and compares them.

1. **Energy available (Model A).** From the battery's telemetry (voltage, current, energy drawn, the last minute of readings), five GRU
   networks estimate how many watt-hours are left above a safe reserve. Their spread, calibrated on batteries they never saw, gives a range.
2. **Energy required (Model B).** A mission is built from its parts: climb, each cruise leg, hover between legs, descent, ground.
   Each part is predicted by a physics-shaped model corrected by XGBoost. The model's past errors on unseen flights give a range.
3. **P(success)** is the chance that the first energy covers the second. A mission is approved only if P(success) ≥ τ.

**Data.** 209 flights of a DJI Matrice 100 (Carnegie Mellon University, 2021). Consecutive flights on one battery were linked into
90 "battery chains". 16 chains were set aside before any modelling and used once, for the final results shown here.

**What it can and cannot claim.**
- Deciding on energy with a calibrated margin cut unsafe approvals from about 12% to under 1% on unseen batteries and flights.
- One drone type, one battery type, 11 labelled test batteries. Ranges are wide for that reason.
- The risk is per battery: when the model over-estimates a battery, it does so for that battery's whole life.
""")
    st.caption("Reserve: 22.6 V at rest (about 3.77 V per cell). Threshold τ = 0.95 by default. Both are project defaults awaiting the guide's confirmation.")


# ------------------------------------------------------------------ layout
def main() -> None:
    st.sidebar.title("EnduroSense")
    st.sidebar.caption("Can this battery fly this mission?")
    page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")
    tau = st.sidebar.slider("Approve only if P(success) is at least (τ)", 0.80, 0.99, L.tau_default(), step=0.01)
    reveal = st.sidebar.toggle("Show what was measured", value=False, help="Reveal the true energy afterwards measured for each battery and flight.")
    st.sidebar.divider()
    st.sidebar.caption("Batteries and flights shown here are from the locked test set: unseen by every model.")
    {PAGES[0]: page_three, PAGES[1]: page_check, PAGES[2]: page_fleet, PAGES[3]: page_results, PAGES[4]: page_how}[page](tau, reveal)


main()
