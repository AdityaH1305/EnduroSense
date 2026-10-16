"""Energy-aware fleet planner: who flies which task, and when (extension 1).

Plain idea: a small fleet works through a day of delivery tasks. Each task has
a priority, a deadline, a payload and a route length; batteries have to be
swapped and recharged. The planner gives each free drone a task only if the
drone's battery very likely covers it (P(success) >= tau) and the task can
still be finished before its deadline, putting urgent, high-priority tasks
first and keeping fuller batteries for bigger tasks.

What is real and what is simulated
----------------------------------
Real: the batteries (recorded chains with their held-out Model A ranges), and
the tasks' missions (recorded flights or same-battery sortie pairs, with their
measured energy, duration, payload and distance, and Model B's held-out range).
Simulated: when tasks arrive, their priorities and deadlines, the time on the
ground between flights, the charging station (chargers, spare packs, charging
power) and each drone's payload limit. None of these are in the dataset; their
settings are in ``config.yaml`` under ``extensions.planner``.

Policies (all see the same simulated days)
------------------------------------------
- ``S1`` first come first served, approved by minutes left (the brief's rule);
- ``S2`` first come first served, approved by P(success) >= tau, best fit (the
  Phase 6 fleet simulation's rule);
- ``S3`` the planner with best estimates (approve if estimate >= need);
- ``S4`` the planner with P(success) >= tau (EnduroSense);
- ``S5`` the planner with the true energies (oracle).

The planner assigns all waiting tasks to all free drones at once
(``scipy.optimize.linear_sum_assignment``): priority first, then the least time
to spare before the deadline, then the least spare energy. A free drone that
cannot safely take any waiting task swaps its battery if a charged one is
waiting. First-come rules ignore deadlines and priorities, as a plain queue
would.

Simplifications: the day runs in one-minute steps; a mission's flight time is
taken as known from the plan (energy, not time, is the uncertain part); a
removed battery is recharged to full and comes back as a fresh recorded chain,
as in ``endurosense.scheduler``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from endurosense.feasibility import p_success_batch
from endurosense.scheduler import Battery, choose_drone

POLICIES = {"S1": "First come, minutes left (the brief)", "S2": "First come, EnduroSense check",
            "S3": "Planner, best estimates", "S4": "Planner, EnduroSense", "S5": "Planner, oracle"}
PLANNERS = ("S3", "S4", "S5")
NEVER = 1e9


def make_day(missions: pd.DataFrame, pool: np.ndarray, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """One day's tasks: a recorded mission each, with a simulated arrival time, priority and deadline."""
    n, pr = cfg["tasks_per_day"], cfg["priorities"]
    names = list(pr)
    prio = rng.choice(names, n, p=[pr[k]["share"] for k in names])
    arrival = np.sort(rng.uniform(0, cfg["day_min"] - cfg["last_arrival_before_end_min"], n))
    mission = rng.choice(pool, n)
    m = missions.iloc[mission]
    return pd.DataFrame({"mission": mission, "priority": prio, "weight": [pr[k]["weight"] for k in prio],
                         "arrival": arrival, "deadline": arrival + np.array([pr[k]["deadline_min"] for k in prio]),
                         "payload_g": m["payload_g"].to_numpy(), "distance_m": m["distance_m"].to_numpy(),
                         "duration": m["duration_min"].to_numpy()})


@dataclass
class Drone:
    battery: Battery
    max_payload_g: float
    used: float = 0.0                  # energy drawn from the current battery (Wh)
    free_at: float = 0.0               # minute the drone is next back at the depot and ready
    flying_min: float = 0.0
    needs_swap: bool = False


@dataclass
class Station:
    """Spare packs and chargers at the depot."""
    ready: int
    chargers: list = field(default_factory=list)      # minute each charger is free
    charging: list = field(default_factory=list)      # minute each charging pack is full

    def tick(self, t: float) -> None:
        done = [c for c in self.charging if c <= t]
        self.charging = [c for c in self.charging if c > t]
        self.ready += len(done)

    def charge(self, t: float, used_wh: float, charger_w: float) -> None:
        k = int(np.argmin(self.chargers))
        start = max(t, self.chargers[k])
        self.chargers[k] = start + used_wh / charger_w * 60.0
        self.charging.append(self.chargers[k])


class FleetPlanner:
    def __init__(self, batteries, states: pd.DataFrame, q_a: pd.DataFrame, missions: pd.DataFrame, q_b: np.ndarray,
                 levels, typical_w: pd.Series, cfg: dict, tau: float, computing=None):
        self.batteries, self.states, self.missions, self.q_b = batteries, states, missions, q_b
        self.q_a, self.levels, self.typical_w, self.cfg, self.tau = q_a, np.asarray(levels, float), typical_w, cfg, tau
        self.computing = computing                     # an ``endurosense.edge.Computing`` or None
        self.mid = int(np.argmin(np.abs(self.levels - 0.5)))
        self.median_a = dict(zip(q_a.index, q_a.to_numpy()[:, self.mid]))
        self._cache: dict = {}

    # ------------------------------------------------------------------ safety scores
    def scores(self, policy: str, drones: list, tasks: pd.DataFrame, extra_wh: np.ndarray) -> np.ndarray:
        """(drones, tasks) array: the policy's safety score; approved if >= the policy's threshold.
        ``extra_wh`` (drones, tasks): planned extra energy on top of the mission (computing), or zeros."""
        out = np.empty((len(drones), len(tasks)))
        need = []
        for i, d in enumerate(drones):
            row = d.battery.row_at(d.used)
            for j, (m, x) in enumerate(zip(tasks["mission"].to_numpy(), extra_wh[i])):
                key = (policy, row, int(m), round(float(x), 3), d.used if policy == "S5" else 0.0, d.battery.chain)
                if key in self._cache:
                    out[i, j] = self._cache[key]
                else:
                    need.append((i, j, key, row, int(m), float(x), d))
        if need:
            rows = np.array([n[3] for n in need])
            mis = np.array([n[4] for n in need])
            extra = np.array([n[5] for n in need])
            qa = self.q_a.loc[rows].to_numpy()
            qb = self.q_b[mis] + extra[:, None]
            if policy in ("S2", "S4"):
                val = p_success_batch(qa, qb, self.levels)
            elif policy == "S3":
                val = qa[:, self.mid] - qb[:, self.mid]
            elif policy == "S5":
                val = np.array([n[6].battery.e_res - n[6].used for n in need]) - self.missions["true_wh"].to_numpy()[mis] - extra
            else:                                                       # S1: minutes left / mission minutes
                s = self.states.loc[rows]
                flying = (s["motors_on"].to_numpy() == 1) & (s["p_mean_30s"].to_numpy() >= 100)
                power = np.where(flying, s["p_mean_30s"].to_numpy(), s["fold"].map(self.typical_w).to_numpy())
                val = qa[:, self.mid] / power * 60.0 / self.missions["duration_min"].to_numpy()[mis]
            for (i, j, key, *_), v in zip(need, val):
                self._cache[key] = out[i, j] = float(v)
        return out

    def threshold(self, policy: str) -> float:
        return {"S1": 1.0, "S2": self.tau, "S3": 0.0, "S4": self.tau, "S5": 0.0}[policy]

    def predicted_wh(self, drones: list) -> np.ndarray:
        return np.array([self.median_a[d.battery.row_at(d.used)] for d in drones])

    # ------------------------------------------------------------------ one day
    def run_day(self, policy: str, tasks: pd.DataFrame, rng: np.random.Generator) -> dict:
        cfg = self.cfg
        new_battery = lambda: self.batteries[rng.integers(len(self.batteries))]
        drones = [Drone(new_battery(), p) for p in cfg["drone_max_payload_g"]]
        station = Station(ready=len(drones) * (cfg["batteries_per_drone"] - 1), chargers=[0.0] * cfg["chargers"])
        jobs = self.computing.jobs(tasks, rng) if self.computing else None
        planned = (np.array([self.computing.planned_wh(jobs.loc[k], tasks.loc[k]) for k in tasks.index]) if jobs is not None
                   else np.zeros(len(tasks)))           # computing energy allowed for before take-off
        state = np.zeros(len(tasks), int)              # 0 waiting/not arrived, 1 flown, 2 dropped
        swapped_for = np.zeros(len(tasks), bool)
        short = set()                                   # minutes in which a swap was wanted but no charged pack was ready
        out = {"flown": 0, "on_time": 0, "late": 0, "unsafe": 0, "swaps": 0,
               "computing_wh": 0.0, "jobs_on_time": 0, "jobs": 0, "local": 0, "offload": 0, "hand over": 0}
        on_time = np.zeros(len(tasks), bool)

        def swap(d: Drone, t: float) -> bool:
            if station.ready == 0:
                short.add(t)
                return False
            station.ready -= 1
            station.charge(t, max(d.used, 0.0), cfg["charger_w"])
            d.battery, d.used, d.needs_swap = new_battery(), 0.0, False
            d.free_at = t + cfg["swap_min"]
            out["swaps"] += 1
            return True

        def fly(d: Drone, k: int, t: float, others: list) -> None:
            m = int(tasks["mission"].iloc[k])
            cost, extra_min = float(self.missions["true_wh"].iloc[m]), 0.0
            if jobs is not None:
                res = self.computing.carry_out(jobs.iloc[k], tasks.iloc[k], others, rng)
                cost += res["wh"]
                extra_min = res["hover_s"] / 60.0
                out["computing_wh"] += res["wh"]
                out["jobs"] += 1
                out["jobs_on_time"] += int(res["on_time"])
                out[res["option"]] += 1
                if res["helper"] is not None:
                    res["helper"].used += res["helper_wh"]
            out["unsafe"] += int(cost > d.battery.e_res - d.used)
            d.used += cost
            dur = float(tasks["duration"].iloc[k]) + extra_min
            d.free_at = t + dur + cfg["turnaround_min"]
            d.flying_min += dur
            d.needs_swap = d.used >= d.battery.e_max
            state[k] = 1
            out["flown"] += 1
            if t + dur <= tasks["deadline"].iloc[k]:
                on_time[k] = True
                out["on_time"] += 1
            else:
                out["late"] += 1

        for t in np.arange(0, cfg["day_min"] + 1, 1.0):
            station.tick(t)
            for d in drones:
                if d.needs_swap and d.free_at <= t:
                    swap(d, t)
            waiting = np.flatnonzero((state == 0) & (tasks["arrival"].to_numpy() <= t))
            if policy in PLANNERS:                      # deadline-aware: drop what can no longer be on time
                late = waiting[t + tasks["duration"].to_numpy()[waiting] > tasks["deadline"].to_numpy()[waiting]]
                state[late] = 2
                waiting = np.setdiff1d(waiting, late)
            free = [d for d in drones if d.free_at <= t and not d.needs_swap]
            if not len(waiting) or not free:
                continue
            if policy in PLANNERS:
                self._plan(policy, free, drones, waiting, tasks, planned, t, swap, fly)
            else:
                self._first_come(policy, free, drones, waiting, tasks, planned, t, swap, fly, swapped_for, state)
        w = tasks["weight"].to_numpy()
        prio = tasks["priority"].to_numpy()
        out.update({"tasks": len(tasks), "missed": int((state != 1).sum()), "pack_shortage_min": float(len(short)),
                    "weighted_on_time": float((w * on_time).sum() / w.sum()),
                    "high_on_time": float(on_time[prio == "high"].mean()) if (prio == "high").any() else np.nan,
                    "utilisation": float(sum(d.flying_min for d in drones) / (len(drones) * cfg["day_min"]))})
        return out

    def _plan(self, policy, free, drones, waiting, tasks, planned, t, swap, fly) -> None:
        tk = tasks.iloc[waiting]
        safe = self.scores(policy, free, tk, np.tile(planned[waiting], (len(free), 1))) >= self.threshold(policy)
        pay = np.array([[p <= d.max_payload_g for p in tk["payload_g"]] for d in free])
        ok = safe & pay
        spare = self.predicted_wh(free)[:, None] - np.array([self.q_b[m, self.mid] for m in tk["mission"]])[None, :]
        slack = (tk["deadline"].to_numpy() - t - tk["duration"].to_numpy())[None, :]
        value = tk["weight"].to_numpy()[None, :] * 10_000 - slack * 10 - np.clip(spare, 0, None)
        cost = np.where(ok, -value, NEVER)
        rows, cols = linear_sum_assignment(cost)
        taken, served = set(), set()
        for i, j in zip(rows, cols):
            if ok[i, j]:
                fly(free[i], int(waiting[j]), t, [d for d in drones if d is not free[i] and d.free_at <= t])
                taken.add(i)
                served.add(j)
        left = np.array([j not in served for j in range(len(tk))])
        for i, d in enumerate(free):                   # idle, and too little energy for a waiting task it could carry
            if i not in taken and d.used > 0 and (pay[i] & ~safe[i] & left).any():
                swap(d, t)

    def _first_come(self, policy, free, drones, waiting, tasks, planned, t, swap, fly, swapped_for, state) -> None:
        for k in waiting:
            free = [d for d in free if d.free_at <= t]
            if tasks["payload_g"].iloc[k] > max(d.max_payload_g for d in drones):
                state[k] = 2                            # no drone in the fleet can carry it
                continue
            capable = [d for d in free if tasks["payload_g"].iloc[k] <= d.max_payload_g]
            if not capable:
                return                                  # the queue waits for a drone that can carry the head task
            tk = tasks.iloc[[k]]
            sc = self.scores(policy, capable, tk, np.full((len(capable), 1), planned[k]))[:, 0]
            i = choose_drone(sc, self.predicted_wh(capable), self.threshold(policy))
            if i is None:
                d = capable[int(np.argmin(self.predicted_wh(capable)))]
                if swapped_for[k] or d.used == 0:
                    state[k] = 2                        # refused even with a fresh battery
                    continue
                if swap(d, t):
                    swapped_for[k] = True
                return                                  # the head task waits for the swap or a charged pack
            fly(capable[i], int(k), t, [d for d in drones if d is not capable[i] and d.free_at <= t])

    # ------------------------------------------------------------------ many days
    def run(self, policy: str, missions_pool: np.ndarray, days: int, seed: int) -> pd.DataFrame:
        """Simulate ``days`` days. The same seed gives every policy the same tasks, computing jobs and
        starting batteries; later fresh batteries come from the same random stream."""
        rows = []
        for day in range(days):
            tasks = make_day(self.missions, missions_pool, self.cfg, np.random.default_rng(seed + day))
            rows.append({"day": day, **self.run_day(policy, tasks, np.random.default_rng(seed + 10_000 + day))})
        return pd.DataFrame(rows)


def compare(planner: FleetPlanner, policies, missions_pool: np.ndarray, days: int, seed: int) -> pd.DataFrame:
    """One row per policy: day averages with 95% intervals over simulated days."""
    rows = []
    for p in policies:
        d = planner.run(p, missions_pool, days, seed)
        ci = lambda c: 1.96 * d[c].std() / np.sqrt(len(d))
        rows.append({"policy": POLICIES[p], "code": p,
                     "weighted_on_time_pct": 100 * d["weighted_on_time"].mean(), "weighted_on_time_ci": 100 * ci("weighted_on_time"),
                     "high_priority_on_time_pct": 100 * d["high_on_time"].mean(),
                     "on_time_per_day": d["on_time"].mean(), "late_per_day": d["late"].mean(), "missed_per_day": d["missed"].mean(),
                     "flown_per_day": d["flown"].mean(), "unsafe_per_day": d["unsafe"].mean(), "unsafe_ci": ci("unsafe"),
                     "unsafe_per_100_flown": 100 * d["unsafe"].sum() / max(d["flown"].sum(), 1),
                     "swaps_per_day": d["swaps"].mean(), "pack_shortage_min_per_day": d["pack_shortage_min"].mean(),
                     "utilisation_pct": 100 * d["utilisation"].mean(),
                     "computing_wh_per_task": d["computing_wh"].sum() / max(d["flown"].sum(), 1),
                     "jobs_on_time_pct": 100 * d["jobs_on_time"].sum() / d["jobs"].sum() if d["jobs"].sum() else np.nan,
                     **{f"{k}_pct": 100 * d[k].sum() / d["jobs"].sum() if d["jobs"].sum() else np.nan for k in ("local", "offload", "hand over")}})
    return pd.DataFrame(rows)
