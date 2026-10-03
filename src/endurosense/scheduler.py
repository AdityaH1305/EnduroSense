"""Fleet simulation on real battery chains: what the decision rule changes in practice.

Plain idea: a small fleet works through a day's queue of missions. Each drone
carries a battery whose behaviour is a *real* recorded chain: its true energy
above the reserve at any point, and the telemetry the battery model sees, both
come from that chain. A mission's true cost is the measured energy of the real
flight(s) it is made of. Nothing about the batteries or the missions is
synthetic; only the pairing of batteries with missions is simulated.

One day:
1. Each drone starts with a freshly swapped-in battery (a random full chain),
   at that chain's pre-flight reading: the last motors-off reading before its
   first take-off, which is when a real go / no-go decision is made.
2. For each task, every drone's policy score is computed from that battery's
   held-out prediction at its current state. Among drones whose score clears
   the threshold, the one with the *least* predicted energy takes the task
   (best fit: fuller batteries are kept for bigger tasks).
3. If no drone is approved, the drone with the least predicted energy swaps
   its battery (a "recharge") and is asked once more; if it still is not
   approved the task is dropped.
4. Flying a task draws its true energy. If that is more than the battery truly
   had above the reserve, the mission cut into the reserve: an **unsafe
   mission** (reserve violation).
5. A battery with no recorded telemetry left is swapped.

Simplification: after a mission the battery's telemetry is taken from the
chain reading with the same energy drawn, which is usually a mid-flight
reading. The battery model is equally well calibrated in flight and on the
ground (Phase 5), so this does not favour any policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from endurosense.feasibility import p_success_batch


@dataclass
class Battery:
    """One real chain: energy drawn at each reading, in order, and the reading ids."""
    chain: int
    e: np.ndarray            # energy drawn since the chain started (non-decreasing)
    rows: np.ndarray         # reading ids (index into the Model A tables)
    e_res: float             # energy drawn at which the reserve is reached

    def row_at(self, energy: float) -> int:
        return int(self.rows[min(np.searchsorted(self.e, energy), len(self.e) - 1)])

    @property
    def e_max(self) -> float:
        return float(self.e[-1])


def batteries_from(a: pd.DataFrame) -> list[Battery]:
    """Full, labelled, supported chains as simulated batteries. ``a``: Model A rows
    (index = reading id) with ``battery_chain``, ``e_chain_wh``, ``e_res_wh``, ``t_chain_s``,
    ``motors_on``. Each battery starts at its pre-flight reading; the first seconds
    of a recording, when the battery model has seen only a reading or two, are dropped."""
    out = []
    for ch, g in a.sort_values("t_chain_s").groupby("battery_chain"):
        on = np.flatnonzero(g["motors_on"].to_numpy() == 1)
        g = g.iloc[max(on[0] - 1, 0):] if len(on) else g
        e = np.maximum.accumulate(g["e_chain_wh"].to_numpy())
        out.append(Battery(int(ch), e, g.index.to_numpy(), float(g["e_res_wh"].iloc[0])))
    return out


class FleetSimulator:
    def __init__(self, batteries, states: pd.DataFrame, q_a: pd.DataFrame, missions: pd.DataFrame,
                 q_b: np.ndarray, levels, typical_w: pd.Series, n_drones: int):
        self.batteries, self.states, self.missions = batteries, states, missions
        self.q_a, self.q_b, self.levels = q_a, q_b, np.asarray(levels, float)
        self.typical_w, self.n_drones = typical_w, n_drones
        self.mid = int(np.argmin(np.abs(self.levels - 0.5)))

    def _scores(self, policy: str, rows: np.ndarray, energies: np.ndarray, bats, task: int) -> tuple[np.ndarray, np.ndarray]:
        """Policy score and predicted energy available for each drone."""
        qa = self.q_a.loc[rows].to_numpy()
        med_a = qa[:, self.mid]
        qb = self.q_b[[task] * len(rows)]
        if policy == "P4":
            true_left = np.array([b.e_res - e for b, e in zip(bats, energies)])
            return true_left - self.missions["true_wh"].iloc[task], med_a
        if policy == "P3":
            return p_success_batch(qa, qb, self.levels), med_a
        if policy == "P2":
            return med_a - qb[:, self.mid], med_a
        s = self.states.loc[rows]
        flying = (s["motors_on"].to_numpy() == 1) & (s["p_mean_30s"].to_numpy() >= 100)
        power = np.where(flying, s["p_mean_30s"].to_numpy(), s["fold"].map(self.typical_w).to_numpy())
        return med_a / power * 60.0 / self.missions["duration_min"].iloc[task], med_a

    def run_day(self, policy: str, threshold: float, tasks: np.ndarray, rng: np.random.Generator) -> dict:
        new_battery = lambda: self.batteries[rng.integers(len(self.batteries))]
        bats = [new_battery() for _ in range(self.n_drones)]
        used = np.zeros(self.n_drones)
        out = {"completed": 0, "unsafe": 0, "dropped": 0, "swaps": 0, "leftover_wh": []}

        def swap(d):
            out["swaps"] += 1
            out["leftover_wh"].append(max(bats[d].e_res - used[d], 0.0))
            bats[d], used[d] = new_battery(), 0.0

        for task in tasks:
            rows = np.array([b.row_at(e) for b, e in zip(bats, used)])
            score, med = self._scores(policy, rows, used, bats, task)
            ok = np.flatnonzero(score >= threshold)
            if len(ok) == 0:
                d = int(np.argmin(med))
                swap(d)
                rows[d] = bats[d].row_at(0.0)
                score, med = self._scores(policy, rows, used, bats, task)
                if score[d] < threshold:
                    out["dropped"] += 1
                    continue
                ok = np.array([d])
            d = int(ok[np.argmin(med[ok])])
            cost = float(self.missions["true_wh"].iloc[task])
            out["unsafe"] += int(cost > bats[d].e_res - used[d])
            out["completed"] += 1
            used[d] += cost
            if used[d] >= bats[d].e_max:                  # no recorded telemetry beyond this point
                swap(d)
        out["leftover_wh"] = float(np.mean(out["leftover_wh"])) if out["leftover_wh"] else np.nan
        return out

    def run(self, policy: str, threshold: float, days: int, tasks_per_day: int, task_pool: np.ndarray, seed: int) -> pd.DataFrame:
        """Simulate ``days`` days. The same seed gives every policy the same tasks and batteries order."""
        rows = []
        for day in range(days):
            rng = np.random.default_rng(seed + day)
            tasks = rng.choice(task_pool, tasks_per_day)
            rows.append({"day": day, **self.run_day(policy, threshold, tasks, np.random.default_rng(seed + 10_000 + day))})
        return pd.DataFrame(rows)
