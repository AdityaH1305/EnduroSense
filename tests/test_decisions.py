"""Feasibility probability, what-if evaluation and the fleet simulator."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from endurosense import whatif as W
from endurosense.feasibility import decide, p_success, p_success_batch
from endurosense.scheduler import Battery, FleetSimulator, batteries_from
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
                             "true_wh": true, "duration_min": true / 400 * 60})
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
