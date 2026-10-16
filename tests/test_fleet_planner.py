"""The energy-aware fleet planner and the computing choice (Phase 9 extensions)."""
import copy

import numpy as np
import pandas as pd
import pytest

from endurosense import whatif as W
from endurosense.config import load_config
from endurosense.edge import Computing
from endurosense.fleet_planner import POLICIES, FleetPlanner, Station, compare, make_day
from endurosense.scheduler import batteries_from
from test_decisions import LEVELS, toy_world

CFG = {"day_min": 240, "last_arrival_before_end_min": 30, "tasks_per_day": 40, "days": 3, "two_sortie_share": 0.0,
       "priorities": {"high": {"share": 0.3, "weight": 3, "deadline_min": 20}, "normal": {"share": 0.4, "weight": 2, "deadline_min": 60},
                      "low": {"share": 0.3, "weight": 1, "deadline_min": 180}},
       "drone_max_payload_g": [500, 250], "turnaround_min": 3, "swap_min": 2, "batteries_per_drone": 2, "chargers": 1, "charger_w": 100}


def world(cfg=CFG, computing=None, bias_wh=0.0, **changes):
    a, q_a, missions, q_b = toy_world()
    q_a = q_a + bias_wh                                          # a battery model that over-estimates by this much
    missions = missions.assign(payload_g=np.where(np.arange(len(missions)) % 4 == 0, 750.0, 250.0), distance_m=460.0)
    cfg = {**copy.deepcopy(cfg), **changes}
    return FleetPlanner(batteries_from(a), a, q_a, missions, q_b, LEVELS, W.typical_power_by_fold(a), cfg, 0.95, computing), missions


def day(planner, missions, policy, seed=0):
    tasks = make_day(missions, np.arange(len(missions)), planner.cfg, np.random.default_rng(seed))
    return tasks, planner.run_day(policy, tasks, np.random.default_rng(seed + 1))


def test_tasks_of_a_day():
    planner, missions = world()
    tasks, _ = day(planner, missions, "S4")
    pr = CFG["priorities"]
    assert len(tasks) == CFG["tasks_per_day"] and tasks["arrival"].is_monotonic_increasing
    assert (tasks["arrival"] <= CFG["day_min"] - CFG["last_arrival_before_end_min"]).all()
    assert np.allclose(tasks["deadline"] - tasks["arrival"], [pr[p]["deadline_min"] for p in tasks["priority"]])
    assert (tasks["weight"] == [pr[p]["weight"] for p in tasks["priority"]]).all()
    assert (tasks["duration"].to_numpy() == missions["duration_min"].to_numpy()[tasks["mission"]]).all()


def test_charging_station():
    s = Station(ready=0, chargers=[0.0])
    s.charge(10.0, 50.0, 100.0)                     # 50 Wh at 100 W: 30 minutes
    s.charge(10.0, 25.0, 100.0)                     # waits for the only charger, then 15 minutes
    s.tick(39.0)
    assert s.ready == 0
    s.tick(40.0)
    assert s.ready == 1
    s.tick(55.0)
    assert s.ready == 2 and s.charging == []


def test_every_policy_keeps_its_promises():
    planner, missions = world()
    for policy in POLICIES:
        tasks, r = day(planner, missions, policy)
        assert r["flown"] + r["missed"] == r["tasks"] == len(tasks) and r["on_time"] + r["late"] == r["flown"]
        assert 0 <= r["weighted_on_time"] <= 1
        if policy in ("S3", "S4", "S5"):
            assert r["late"] == 0                   # the planner never starts a task it cannot finish on time
    assert day(planner, missions, "S5")[1]["unsafe"] == 0       # the oracle never cuts into the reserve


def test_payload_limit_is_respected():
    planner, missions = world(drone_max_payload_g=[250, 250])  # no drone can carry the 750 g tasks
    for policy in POLICIES:
        tasks, r = day(planner, missions, policy)
        assert r["missed"] >= int((tasks["payload_g"] > 250).sum())   # heavy tasks are never flown
        assert r["flown"] > 0                                          # and they do not block the queue


def test_priorities_and_deadlines_matter_to_the_planner_only():
    planner, missions = world(tasks_per_day=60)     # more work than two drones can do
    res = compare(planner, ["S2", "S4"], np.arange(len(missions)), days=4, seed=0).set_index("code")
    assert res.loc["S4", "high_priority_on_time_pct"] > res.loc["S2", "high_priority_on_time_pct"]
    assert res.loc["S4", "weighted_on_time_pct"] > res.loc["S2", "weighted_on_time_pct"]
    tight, _ = world(tasks_per_day=60, priorities={k: {**v, "deadline_min": v["deadline_min"] / 3} for k, v in CFG["priorities"].items()})
    res_tight = compare(tight, ["S4"], np.arange(len(missions)), days=4, seed=0).set_index("code")
    assert res_tight.loc["S4", "weighted_on_time_pct"] < res.loc["S4", "weighted_on_time_pct"]


def test_spare_packs_and_chargers():
    plenty, missions = world(batteries_per_drone=50, chargers=50)
    assert compare(plenty, ["S4"], np.arange(len(missions)), days=2, seed=0)["pack_shortage_min_per_day"].iloc[0] == 0
    scarce, _ = world(batteries_per_drone=1, chargers=1, charger_w=20)   # no spares and slow charging: drones stand idle
    r = compare(scarce, ["S4"], np.arange(len(missions)), days=2, seed=0).iloc[0]
    assert r["pack_shortage_min_per_day"] > 0 and r["flown_per_day"] < compare(plenty, ["S4"], np.arange(len(missions)), days=2, seed=0)["flown_per_day"].iloc[0]


# ------------------------------------------------------------------ computing
CC = load_config()["extensions"]["computing"]
JOB = pd.Series({"data_mb": 80.0, "local_s": 40.0, "on_site": True, "deadline_s": 60.0, "fade": 1.0, "blocked": False, "edge_wait_s": 0.0})
TASK = pd.Series({"distance_m": 460.0})


def test_link_rate():
    c = Computing(CC, 540.0, "energy-aware")
    assert c.rate_mbps(100) > c.rate_mbps(300) > c.rate_mbps(600) > 0          # weaker further away
    assert c.rate_mbps(200, blocked=True) == pytest.approx(CC["blocked_rate_factor"] * c.rate_mbps(200))
    assert c.rate_mbps(200, fade=0.1) < c.rate_mbps(200)
    assert c.distance_m(JOB, TASK) == 230.0 and c.distance_m({**JOB, "on_site": False}, TASK) == 115.0


def test_options_and_the_energy_aware_choice():
    c = Computing(CC, 540.0, "energy-aware")
    opts = c.options(JOB, TASK, 1.0, False, 0.0, helper=True)
    t_local, e_local, _ = opts["local"]
    assert t_local == 40.0 and e_local == pytest.approx((540.0 + CC["cpu_w"]) * 40.0 / 3600.0)   # hovering dominates
    busy = c.options(JOB, TASK, 1.0, False, 45.0, helper=True)
    assert busy["offload"][0] == pytest.approx(opts["offload"][0] + 45.0) and busy["offload"][1] > opts["offload"][1]
    assert opts["hand over"][2] == pytest.approx(CC["cpu_w"] * 40.0 / 3600.0)   # the helper drone pays for the computing
    k = c.choose(JOB, opts)
    on_time = {n for n, (s, _, _) in opts.items() if s <= JOB["deadline_s"]}
    assert k in on_time and opts[k][1] == min(opts[n][1] for n in on_time)
    late_everywhere = {n: (s + 1000, e, h) for n, (s, e, h) in opts.items()}
    assert c.choose(JOB, late_everywhere) == min(opts, key=lambda n: opts[n][0])   # nothing on time: the fastest
    assert Computing(CC, 540.0, "local").choose(JOB, opts) == "local"
    off_site = c.options({**JOB, "on_site": False}, TASK, 1.0, False, 0.0, helper=False)
    assert off_site["local"][1] == pytest.approx(CC["cpu_w"] * 40.0 / 3600.0)   # processed on the way back: no hovering


def test_planned_energy_allows_for_a_bad_link():
    for rule in ("local", "offload", "energy-aware"):
        c = Computing(CC, 540.0, rule)
        good = c.carry_out(JOB, TASK, [])
        assert c.planned_wh(JOB, TASK) >= good["wh"] - 1e-9
    assert Computing(CC, 540.0, "energy-aware").planned_wh(JOB, TASK) <= Computing(CC, 540.0, "local").planned_wh(JOB, TASK)


def test_weak_links_push_the_choice_on_board():
    jobs = Computing(CC, 540.0, "energy-aware").jobs(pd.DataFrame(index=range(400)), np.random.default_rng(0))
    share = {}
    for snr in (5, 30):
        c = Computing({**CC, "snr_db_at_100m": snr}, 540.0, "energy-aware")
        share[snr] = np.mean([c.carry_out(j, TASK, [])["option"] == "local" for _, j in jobs.iterrows()])
    assert share[5] > share[30]


def test_computing_energy_enters_the_decision():
    heavy = {**CC, "on_site_share": 1.0, "local_s": [600, 600], "job_deadline_s": [900, 900], "snr_db_at_100m": -20}
    light, missions = world()
    slow, _ = world(computing=Computing(heavy, 540.0, "local"))    # ten minutes of hovering per task: about 90 Wh
    assert day(slow, missions, "S4")[1]["flown"] < day(light, missions, "S4")[1]["flown"]
    assert day(slow, missions, "S4")[1]["flown"] == 0                # no battery can cover that


# ------------------------------------------------------------------ added after mutation testing
def test_under_pressure_the_planner_is_never_late_and_reckless_rules_are_caught():
    planner, missions = world(tasks_per_day=90)                  # far more work than two drones can do
    for seed in range(3):
        assert day(planner, missions, "S4", seed)[1]["late"] == 0
    optimistic, _ = world(tasks_per_day=90, bias_wh=6.0)
    res = compare(optimistic, ["S3", "S4", "S5"], np.arange(len(missions)), days=6, seed=0).set_index("code")
    assert res.loc["S3", "unsafe_per_day"] > 0                    # an over-optimistic battery model does cut into the reserve
    assert res.loc["S5", "unsafe_per_day"] == 0 and res.loc["S4", "unsafe_per_day"] <= res.loc["S3", "unsafe_per_day"]


def test_priority_weights_steer_the_planner():
    same = {k: {**v, "deadline_min": 120} for k, v in CFG["priorities"].items()}             # deadlines equal: only the weight differs
    weighted, missions = world(tasks_per_day=90, priorities=same)
    flat, _ = world(tasks_per_day=90, priorities={k: {**v, "weight": 1} for k, v in same.items()})
    hi = lambda p: compare(p, ["S4"], np.arange(len(missions)), days=6, seed=0)["high_priority_on_time_pct"].iloc[0]
    assert hi(weighted) > hi(flat)


def test_a_task_no_drone_can_carry_does_not_block_the_queue():
    planner, missions = world(drone_max_payload_g=[250, 250])
    for policy in ("S1", "S2"):
        tasks, r = day(planner, missions, policy)
        light = int((tasks["payload_g"] <= 250).sum())
        assert r["flown"] >= 0.6 * light                          # light tasks keep flowing past the heavy ones


def test_the_choice_never_trades_the_deadline_for_energy():
    c = Computing(CC, 540.0, "energy-aware")
    opts = {"local": (40.0, 5.0, 0.0), "offload": (100.0, 1.0, 0.0)}   # the cheaper option would be late
    assert c.choose(JOB, opts) == "local"
