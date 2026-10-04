"""Feasibility probability, what-if evaluation and the fleet simulator."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from endurosense import whatif as W
from endurosense.feasibility import decide, p_success, p_success_batch
from endurosense.scheduler import Battery, FleetSimulator, batteries_from, choose_drone
from endurosense.uncertainty.quantiles import PredictiveDistribution

LEVELS = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
Z = norm.ppf(LEVELS)


def normal_q(mu, sd):
    return np.asarray(mu, float)[:, None] + np.asarray(sd, float)[:, None] * Z[None, :]


def test_p_success_matches_the_closed_form_for_normal_distributions():
    mu_a, sd_a = np.array([50.0, 30.0, 20.0, 20.0, 5.0]), np.array([5.0, 3.0, 4.0, 1.0, 2.0])
    mu_b, sd_b = np.array([20.0, 28.0, 20.0, 25.0, 30.0]), np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    p = p_success_batch(normal_q(mu_a, sd_a), normal_q(mu_b, sd_b), LEVELS)
    exact = norm.cdf((mu_a - mu_b) / np.hypot(sd_a, sd_b))
    assert p == pytest.approx(exact, abs=0.01)
    assert p[2] == pytest.approx(0.5, abs=0.005)                      # same centre: a coin flip
    assert ((p >= 0) & (p <= 1)).all()


def test_p_success_batch_agrees_with_the_single_pair_version():
    qa, qb = normal_q([40.0, 22.0], [4.0, 3.0]), normal_q([30.0, 21.0], [2.0, 1.5])
    batch = p_success_batch(qa, qb, LEVELS)
    for i in range(2):
        single = p_success(PredictiveDistribution(LEVELS, qa[i]), PredictiveDistribution(LEVELS, qb[i]))
        assert single == pytest.approx(batch[i], abs=1e-6)


def test_p_success_falls_as_the_mission_gets_harder_and_rises_with_more_energy():
    need = np.linspace(5, 60, 30)
    p = p_success_batch(normal_q(np.full(30, 30.0), np.full(30, 4.0)), normal_q(need, 0.05 * need), LEVELS)
    assert (np.diff(p) <= 1e-12).all() and p[0] > 0.999 and p[-1] < 0.001
    have = np.linspace(5, 60, 30)
    p = p_success_batch(normal_q(have, np.full(30, 4.0)), normal_q(np.full(30, 30.0), np.full(30, 1.5)), LEVELS)
    assert (np.diff(p) >= -1e-12).all()


def test_wider_uncertainty_pulls_the_probability_towards_a_half():
    sure = p_success_batch(normal_q([30.0], [1.0]), normal_q([25.0], [1.0]), LEVELS)[0]
    unsure = p_success_batch(normal_q([30.0], [6.0]), normal_q([25.0], [1.0]), LEVELS)[0]
    assert 0.5 < unsure < sure


def test_decision_rule():
    assert decide(0.96, 0.95) and not decide(0.94, 0.95)
    assert decide(0.95, 0.95)                                           # the threshold itself is a GO
    assert not decide(None, 0.5)                                        # battery model abstained
    assert decide(0.96) and not decide(0.5)                             # default tau from the config (0.95)


def test_rates_count_the_two_kinds_of_mistake():
    feasible = np.array([1, 1, 1, 1, 0, 0], bool)
    approve = np.array([1, 1, 1, 0, 1, 0], bool)
    r = W.rates(approve, feasible)
    assert r["unsafe_approval_rate"] == 0.5 and r["wasted_refusal_rate"] == 0.25
    assert r["approved_share"] == pytest.approx(4 / 6) and r["unsafe_share_of_approvals"] == 0.25


def test_a_perfect_score_approves_everything_feasible_with_no_unsafe_approvals():
    rng = np.random.default_rng(0)
    margin = rng.normal(0, 10, 500)
    share, thr = W.approved_at_unsafe(margin, margin >= 0, 0.0)
    assert share == 1.0 and thr >= 0
    noisy = margin + rng.normal(0, 8, 500)
    assert W.approved_at_unsafe(noisy, margin >= 0, 0.01)[0] < 1.0
    cur = W.tradeoff_curve(noisy, margin >= 0)
    assert cur["unsafe_approval_rate"].is_monotonic_increasing and cur["wasted_refusal_rate"].is_monotonic_decreasing


def toy_world(n_chains=6, seed=0):
    """Batteries of 70 Wh above the reserve read every 1 Wh; the battery model is right
    on average with a 2 Wh spread. Missions of 5-25 Wh known to within 0.5 Wh."""
    rng = np.random.default_rng(seed)
    rows = []
    for ch in range(n_chains):
        for i in range(81):
            rows.append({"battery_chain": ch, "flight": ch * 10 + i // 27, "time": float(i % 27) * 10, "t_chain_s": float(i * 10),
                         "motors_on": int(i % 27 > 0), "p_mean_30s": 400.0, "p": 400.0, "fold": ch % 2,
                         "e_chain_wh": float(i), "e_res_wh": 70.0, "remaining_wh": 70.0 - i})
    a = pd.DataFrame(rows)
    q_a = pd.DataFrame(normal_q(a["remaining_wh"], np.full(len(a), 2.0)), index=a.index)
    true = rng.uniform(5, 25, 40)
    missions = pd.DataFrame({"flights": [(i,) for i in range(40)], "sorties": 1, "fold": np.arange(40) % 2,
                             "battery_chain": np.arange(40) % n_chains, "true_wh": true, "duration_min": true / 400 * 60})
    return a, q_a, missions, normal_q(true, np.full(40, 0.5))


def test_states_and_pairs():
    a, q_a, missions, q_b = toy_world()
    assert (W.decision_states(a, 20)["time"] % 20 == 0).all()
    pre = W.preflight_states(a)
    assert len(pre) == a["flight"].nunique() and (pre["motors_on"] == 0).all() and (pre["time"] == 0).all()
    pairs = W.make_pairs(a, missions, 3000, np.random.default_rng(1))
    assert (pairs["true_margin_wh"] == pairs["available_wh"] - pairs["required_wh"]).all()
    assert (pairs["feasible"] == (pairs["true_margin_wh"] >= 0)).all()
    assert (a.loc[pairs["state"], "remaining_wh"].to_numpy() == pairs["available_wh"].to_numpy()).all()
    assert 0.2 < pairs["feasible"].mean() < 0.95                       # both outcomes present
    assert (missions["battery_chain"].to_numpy()[pairs["mission"]] == pairs["mission_chain"].to_numpy()).all()


def test_multi_sortie_missions_are_real_same_battery_flights_with_one_shared_error():
    # 3 batteries: flights (1, 2, 3), (4, 5) and (6,); two folds
    idx = pd.Index([1, 2, 3, 4, 5, 6], name="flight")
    chain, fold = [10, 10, 10, 20, 20, 30], [0, 0, 0, 1, 1, 1]
    parts = pd.DataFrame({"pred_climb": 2.0, "pred_descent": 1.0, "pred_hover": 0.5, "pred_legs": [10.0, 12, 14, 16, 18, 20],
                          "pred_ground": 0.5, "err_climb": [0.1, -0.1, 0.2, 0.0, 0.3, -0.2], "err_descent": 0.0, "err_hover": 0.0,
                          "err_ground": 0.0, "err_legs_rel": [0.05, -0.05, 0.02, 0.1, -0.1, 0.0], "err_total_rel": 0.0,
                          "err_battery_chain": chain, "fold": fold}, index=idx)
    fb = pd.DataFrame({"flight": idx, "total_wh": parts.filter(like="pred_").sum(axis=1).to_numpy() + 0.3,
                       "climb_s": 20.0, "cruise_s": 100.0, "descent_s": 20.0, "hover_s": 10.0})
    missions, q = W.build_missions(parts, fb, LEVELS)
    assert missions["flights"].iloc[:6].tolist() == [(i,) for i in idx]                 # single flights first, in order
    multi = missions[missions.sorties > 1]
    assert sorted(multi["flights"]) == [(1, 2), (1, 2, 3), (1, 3), (2, 3), (4, 5)]      # only flights that shared a battery
    assert q.shape == (len(missions), len(LEVELS)) and (np.diff(q, axis=1) >= 0).all()
    m = missions[missions["flights"] == (4, 5)].iloc[0]
    assert m["true_wh"] == pytest.approx(fb.set_index("flight").loc[[4, 5], "total_wh"].sum())
    assert m["duration_min"] == pytest.approx(2 * 150 / 60) and m["battery_chain"] == 20 and m["fold"] == 1
    # one shared error set: the two-sortie range is as wide as the two single ranges added, not narrower
    width = lambda i: q[i, -1] - q[i, 0]
    pos = {f: i for i, f in enumerate(missions["flights"])}
    assert width(pos[(4, 5)]) == pytest.approx(width(pos[(4,)]) + width(pos[(5,)]), rel=1e-6)


def test_preflight_state_is_the_last_reading_before_take_off_and_typical_power_comes_from_other_folds():
    a = pd.DataFrame({"flight": 1, "battery_chain": 1, "time": [0.0, 1, 2, 3, 4, 5], "motors_on": [0, 0, 0, 1, 1, 0],
                      "p": [10.0, 10, 10, 100, 100, 10], "fold": 0})
    b = a.assign(flight=2, battery_chain=2, fold=1, p=[10.0, 10, 10, 300, 300, 10])
    both = pd.concat([a, b], ignore_index=True)
    pre = W.preflight_states(both)
    assert pre["time"].tolist() == [2.0, 2.0]                          # not the first reading, not the one after landing
    assert W.typical_power_by_fold(both).to_dict() == {0: 300.0, 1: 100.0}


def test_bootstrap_reweights_the_battery_side_and_the_mission_side():
    chains = np.arange(5)
    pairs = pd.DataFrame([(a, b) for a in chains for b in chains], columns=["battery_chain", "mission_chain"])
    pairs["feasible"] = np.arange(len(pairs)) % 2 == 0
    n = 0
    for keep, w in W._chain_resamples(pairs, 30, seed=0):
        full = pd.Series(0.0, index=pairs.index); full[keep] = w
        times = {c: np.sqrt(full[(pairs.battery_chain == c) & (pairs.mission_chain == c)].iloc[0]) for c in chains}   # (c, c) pair
        expected = pairs["battery_chain"].map(times) * pairs["mission_chain"].map(times)
        assert np.allclose(full, expected) and sum(times.values()) == len(chains)
        n += 1
    assert n > 20


def test_policies_on_a_world_where_the_models_are_right():
    a, q_a, missions, q_b = toy_world()
    pairs = W.make_pairs(a, missions, 4000, np.random.default_rng(2))
    sc = W.policy_scores(pairs, a, q_a, missions, q_b, LEVELS, W.typical_power_by_fold(a))
    f = pairs["feasible"].to_numpy()
    # the models' medians are the truth here, so P2's margin is the true margin and P3 agrees on the ranking
    assert sc["P2"].to_numpy() == pytest.approx(pairs["true_margin_wh"].to_numpy(), abs=1e-9)
    assert W.rates(sc["P4"] >= 0, f)["unsafe_approval_rate"] == 0 and W.rates(sc["P4"] >= 0, f)["wasted_refusal_rate"] == 0
    assert W.rates(sc["P3"] >= 0.95, f)["unsafe_approval_rate"] == 0
    assert ((sc["P3"] >= 0.5) == (sc["P2"] >= 0)).mean() > 0.995
    # constant 400 W draw: minutes left / duration is exactly energy available / energy required
    assert sc["P1"].to_numpy() == pytest.approx((pairs["available_wh"] / pairs["required_wh"]).to_numpy(), rel=1e-9)
    t = W.summary_table(sc, pairs, n_boot=20)
    assert t["auc"].min() > 0.999 and set(t["policy"]) == set(W.POLICIES.values())
    assert (t["auc_ci_low"] <= t["auc"]).all() and (t["auc"] <= t["auc_ci_high"]).all()
    g = W.paired_gain(sc, pairs, "P4", "P1", n_boot=20).set_index("metric")
    assert (g["gain"] >= 0).all() and (g["ci_low"] <= g["gain"]).all() and (g["gain"] <= g["ci_high"]).all()
    assert (W.paired_gain(sc, pairs, "P2", "P2", n_boot=5)["gain"] == 0).all()        # a policy against itself


def test_probability_reliability_recovers_an_honest_probability():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 40000)
    ok = rng.uniform(0, 1, 40000) < p
    rel = W.probability_reliability(p, ok, rng.integers(0, 30, 40000))
    assert (rel["mean_predicted"] - rel["observed_success"]).abs().max() < 0.03
    assert rel["pairs"].sum() == 40000


def test_batteries_follow_their_chain():
    a, *_ = toy_world(2)
    bats = batteries_from(a)
    assert [b.chain for b in bats] == [0, 1] and bats[0].e_res == 70.0 and bats[0].e_max == 80.0
    b = bats[1]
    assert a.loc[b.row_at(0.0), "e_chain_wh"] == 0 and a.loc[b.row_at(12.4), "e_chain_wh"] == 13   # first reading at or past it
    assert a.loc[b.row_at(500.0), "e_chain_wh"] == 80                                               # clipped to the last reading
    assert isinstance(b, Battery) and (a.loc[b.rows, "battery_chain"] == 1).all()
    # a battery starts at its pre-flight reading: the last motors-off reading before the first take-off
    late = a[a["battery_chain"] == 0].assign(motors_on=lambda d: (d["e_chain_wh"] >= 4).astype(int))
    start = batteries_from(late)[0].row_at(0.0)
    assert late.loc[start, "e_chain_wh"] == 3 and late.loc[start, "motors_on"] == 0
    assert batteries_from(late)[0].e_res == 67.0 and batteries_from(late)[0].e[0] == 0     # measured from that reading


def test_best_fit_assignment():
    score, energy = np.array([0.99, 0.2, 0.97, 0.96]), np.array([60.0, 5.0, 30.0, 45.0])
    assert choose_drone(score, energy, 0.95) == 2                       # the emptiest of the approved drones
    assert choose_drone(score, energy, 0.98) == 0
    assert choose_drone(score, energy, 0.999) is None


def test_fleet_bookkeeping_on_a_case_worked_out_by_hand():
    # one drone; every battery has exactly 70 Wh above the reserve and 80 Wh of recording; tasks cost exactly 20 Wh
    a, q_a, _, _ = toy_world()
    missions = pd.DataFrame({"flights": [(0,), (1,)], "sorties": 1, "fold": 0, "battery_chain": 0,
                             "true_wh": [20.0, 100.0], "duration_min": [3.0, 15.0]})
    sim = FleetSimulator(batteries_from(a), a, q_a, missions, normal_q(missions["true_wh"], [0.5, 0.5]), LEVELS,
                         W.typical_power_by_fold(a), n_drones=1)
    day = lambda policy, thr, task: sim.run(policy, thr, days=1, tasks_per_day=30, task_pool=np.array([task]), seed=0).iloc[0]
    oracle = day("P4", 0.0, 0)                 # 3 tasks per battery (60 Wh), the 4th would need 80 > 70: swap first
    assert (oracle["completed"], oracle["unsafe"], oracle["dropped"], oracle["swaps"]) == (30, 0, 0, 9)
    assert oracle["leftover_wh"] == pytest.approx(10.0)
    reckless = day("P2", -1e9, 0)              # flies the 4th task into the reserve, then the recording runs out
    assert (reckless["completed"], reckless["unsafe"], reckless["dropped"], reckless["swaps"]) == (30, 7, 0, 7)
    assert reckless["leftover_wh"] == 0.0
    too_big = day("P4", 0.0, 1)                # 100 Wh fits no battery: swap, ask again, drop
    assert (too_big["completed"], too_big["unsafe"], too_big["dropped"], too_big["swaps"]) == (0, 0, 30, 30)


def test_fleet_simulation_safety_and_bookkeeping():
    a, q_a, missions, q_b = toy_world()
    sim = FleetSimulator(batteries_from(a), a, q_a, missions, q_b, LEVELS, W.typical_power_by_fold(a), n_drones=3)
    pool = np.arange(len(missions))
    run = lambda policy, thr: sim.run(policy, thr, days=15, tasks_per_day=30, task_pool=pool, seed=7)
    oracle, enduro, reckless = run("P4", 0.0), run("P3", 0.95), run("P2", -1e9)
    for d in (oracle, enduro, reckless):
        assert ((d["completed"] + d["dropped"]) == 30).all()            # every task is flown or dropped
    assert oracle["unsafe"].sum() == 0 and oracle["dropped"].sum() == 0
    assert enduro["unsafe"].sum() == 0                                  # models are right on average here
    assert reckless["unsafe"].sum() > 0 and reckless["dropped"].sum() == 0
    assert oracle["swaps"].sum() <= enduro["swaps"].sum()               # caution costs battery swaps
    assert oracle["leftover_wh"].mean() < enduro["leftover_wh"].mean()
    assert run("P3", 0.95).equals(enduro)                               # same seed, same days


def test_missions_for_unseen_flights_use_the_given_error_sets():
    idx = pd.Index([1, 2, 3], name="flight")
    parts = pd.DataFrame({"pred_climb": 2.0, "pred_descent": 1.0, "pred_hover": 0.5, "pred_legs": [10.0, 12.0, 20.0],
                          "pred_ground": 0.5, "battery_chain": [7, 7, 8]}, index=idx)          # no errors, no folds: unseen flights
    fb = pd.DataFrame({"flight": idx, "total_wh": [14.0, 16.5, 24.0], "climb_s": 20.0, "cruise_s": 100.0, "descent_s": 20.0, "hover_s": 10.0})
    cal = pd.DataFrame({"climb": [-1.0, 0.0, 1.0], "descent": 0.0, "hover": 0.0, "ground": 0.0, "legs_rel": [-0.1, 0.0, 0.1],
                        "total_rel": 0.0, "battery_chain": [1, 2, 3]})
    missions, q = W.build_missions(parts, fb, np.array([0.3, 0.6, 0.9]), calibration=cal)
    assert missions["flights"].tolist() == [(1,), (2,), (3,), (1, 2)] and (missions["fold"] == -1).all()
    assert q[0].tolist() == pytest.approx([14.0 - 1 - 1, 14.0, 14.0 + 1 + 1])                  # the three error sets replayed on flight 1
    assert q[3].tolist() == pytest.approx(q[0] + q[1])                                         # one shared error set for both sorties


def test_operating_points_with_fixed_margins_and_the_mistake_summary():
    a, q_a, missions, q_b = toy_world()
    pairs = W.make_pairs(a, missions, 4000, np.random.default_rng(5))
    sc = W.policy_scores(pairs, a, q_a, missions, q_b, LEVELS, W.typical_power_by_fold(a))
    f = pairs["feasible"].to_numpy()
    hind = W.matched_margins(sc, pairs, [0.95])
    assert hind[0.95]["P2"] > 0 and hind[0.95]["P1"] > 1                                        # a margin, not a discount
    pts = W.operating_points(sc, pairs, [0.95], margins={0.95: {"P1": 1.5, "P2": 4.0}}).set_index("policy")
    row = pts.loc["   P2 with margin tuned on development data for tau = 0.95 (threshold 4.00)"]
    assert row["unsafe_approval_rate"] == W.rates(sc["P2"] >= 4.0, f)["unsafe_approval_rate"]
    assert row["wasted_refusal_rate"] == W.rates(sc["P2"] >= 4.0, f)["wasted_refusal_rate"]
    assert pts.loc["P4 oracle", "unsafe_approval_rate"] == 0 and len(pts) == 6
    m = W.mistakes_at(sc, pairs, 0.95)
    approve = (sc["P3"] >= 0.95).to_numpy()
    assert m["wasted_refusals"] == int((~approve & f).sum()) and m["unsafe_approvals"] == int((approve & ~f).sum())
    assert m["refused_margin_wh_median"] > 0 and m["unsafe_shortfall_wh_worst"] <= 0


def test_real_pair_check_reports_an_honest_margin_when_the_models_are_right():
    rng = np.random.default_rng(0)
    n = 400
    pre = pd.DataFrame({"flight": np.arange(n), "remaining_wh": rng.uniform(30, 70, n)}, index=np.arange(1000, 1000 + n))
    need = rng.uniform(10, 25, n)
    q_a = pd.DataFrame(normal_q(pre["remaining_wh"] + rng.normal(0, 3, n), np.full(n, 3.0)), index=pre.index)   # 3 Wh error, 3 Wh spread
    missions = pd.DataFrame({"true_wh": need})
    q_b = normal_q(need + rng.normal(0, 0.5, n), np.full(n, 0.5))
    r = W.real_pair_check(pre, np.arange(n), q_a, missions, q_b, LEVELS, rng)
    assert r["flights"] == n and abs(r["correlation"]) < 0.15 and r["ci_low"] < r["correlation"] < r["ci_high"]
    assert r["margin_90_range_covers"] == pytest.approx(0.90, abs=0.05)
    assert r["margin_error_sd_actual_wh"] == pytest.approx(r["margin_error_sd_if_independent_wh"], rel=0.1)


def test_task_pool_and_policy_comparison():
    from endurosense.scheduler import compare_policies, task_pool
    a, q_a, missions, q_b = toy_world()
    missions = missions.assign(sorties=np.where(np.arange(len(missions)) < 28, 1, 2))
    pool = task_pool(missions, 0.3, seed=1)
    assert (missions["sorties"].to_numpy()[pool] == 2).mean() == pytest.approx(0.3, abs=0.01) and len(pool) == 40
    assert len(task_pool(missions.assign(sorties=1), 0.3, seed=1)) == len(missions)            # no two-sortie missions available
    sim = FleetSimulator(batteries_from(a), a, q_a, missions, q_b, LEVELS, W.typical_power_by_fold(a), n_drones=2)
    t = compare_policies(sim, {"oracle": ("P4", 0.0), "reckless": ("P2", -1e9)}, days=5, tasks_per_day=20, task_pool=pool, seed=3)
    assert t["policy"].tolist() == ["oracle", "reckless"] and t.loc[0, "unsafe_per_100_missions"] == 0
    assert t.loc[1, "unsafe_per_100_missions"] > 0 and (t["completed_per_day"] + t["dropped_per_day"] == 20).all()

