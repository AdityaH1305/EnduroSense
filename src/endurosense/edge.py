"""Where to process a task's data: on board, on an edge server, or on another drone (extension 2).

Plain idea: many delivery tasks come with a computing job, for example checking
images of the drop-off site. The drone can

- **process it locally** on its onboard computer;
- **offload it** over the radio to an edge server at the depot, which is faster
  but may be busy, and the link may be weak or blocked;
- **hand it over** to another drone waiting at the depot, which computes it on
  the ground with its own battery.

If the result is needed before the drone leaves the site, it hovers while it
waits, and hovering costs about 540 W (measured on development flights). The
radio (about 1 W) and the computer (about 15 W) cost little by comparison, so
the energy of a choice is mostly the *hover time* it adds. Jobs not needed on
site are processed on the way back, without hovering.

The **energy-aware** rule measures the link and asks the server's queue at the
site, then picks the option with the least energy for this drone that still
meets the job's deadline (the fastest one if none does). Before take-off the
planner does not know the link yet, so for P(success) it adds the energy the
job would take in a bad case (weak, blocked link and a busy server).

Everything here is simulated: the dataset has no radio, computer or server
data. Settings are in ``config.yaml`` under ``extensions.computing`` and are
typical published values, not measurements; only the hover power is measured.

Link model: rate = bandwidth x log2(1 + SNR), SNR falling with distance to the
depot (path-loss exponent), multiplied by Rayleigh fading (an exponential
factor), and cut by a fixed factor when the link is blocked. Mobility enters
through the distance: on-site jobs are sent from the delivery site (the far
point of the route, about half its length from the depot), in-flight jobs from
half that distance on average.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULES = ("local", "offload", "energy-aware")


class Computing:
    def __init__(self, cfg: dict, hover_w: float, rule: str):
        if rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}")
        self.cfg, self.hover_w, self.rule = cfg, float(hover_w), rule

    # ------------------------------------------------------------------ jobs and the link
    def jobs(self, tasks: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """One computing job per task, plus the conditions it will meet (drawn now so every rule sees the same)."""
        c, n = self.cfg, len(tasks)
        busy = rng.random(n) < c["edge_busy_prob"]
        return pd.DataFrame({"data_mb": rng.uniform(*c["data_mb"], n), "local_s": rng.uniform(*c["local_s"], n),
                             "on_site": rng.random(n) < c["on_site_share"], "deadline_s": rng.uniform(*c["job_deadline_s"], n),
                             "fade": rng.exponential(1.0, n), "blocked": rng.random(n) < c["blocked_prob"],
                             "edge_wait_s": np.where(busy, rng.exponential(c["edge_wait_s_mean"], n), 0.0)}, index=tasks.index)

    def distance_m(self, job, task) -> float:
        site = max(float(task["distance_m"]) / 2.0, 1.0)
        return site if job["on_site"] else site / 2.0

    def rate_mbps(self, distance_m: float, fade: float = 1.0, blocked: bool = False) -> float:
        c = self.cfg
        snr_db = c["snr_db_at_100m"] - 10 * c["path_loss_exponent"] * np.log10(max(distance_m, 1.0) / 100.0)
        rate = c["bandwidth_mhz"] * np.log2(1 + 10 ** (snr_db / 10) * fade)
        return float(rate * (c["blocked_rate_factor"] if blocked else 1.0))

    # ------------------------------------------------------------------ the options
    def options(self, job, task, fade: float, blocked: bool, edge_wait_s: float, helper: bool) -> dict:
        """option -> (seconds until the result is ready, Wh from this drone, Wh from the helper drone)."""
        c = self.cfg
        hover = lambda s: self.hover_w * s / 3600.0 if job["on_site"] else 0.0
        t_tx = job["data_mb"] * 8.0 / max(self.rate_mbps(self.distance_m(job, task), fade, blocked), 1e-6)
        out = {"local": (job["local_s"], c["cpu_w"] * job["local_s"] / 3600.0 + hover(job["local_s"]), 0.0)}
        t_off = t_tx + edge_wait_s + job["local_s"] / c["edge_speedup"]
        out["offload"] = (t_off, c["tx_w"] * t_tx / 3600.0 + hover(t_off), 0.0)
        if helper:
            t_help = t_tx + job["local_s"]
            out["hand over"] = (t_help, c["tx_w"] * t_tx / 3600.0 + hover(t_help), c["cpu_w"] * job["local_s"] / 3600.0)
        return out

    def choose(self, job, opts: dict) -> str:
        if self.rule != "energy-aware":
            return self.rule
        on_time = [k for k, (s, _, _) in opts.items() if s <= job["deadline_s"]]
        if on_time:
            return min(on_time, key=lambda k: opts[k][1])
        return min(opts, key=lambda k: opts[k][0])

    # ------------------------------------------------------------------ planning and doing
    def planned_wh(self, job, task) -> float:
        """Energy to allow for before take-off: the rule's option in a bad case (link and server unknown yet)."""
        bad = self.cfg["plan_for"]
        opts = self.options(job, task, bad["fade"], bad["blocked"], bad["edge_wait_s"], helper=False)
        if self.rule == "energy-aware":
            return float(min(e for _, e, _ in opts.values()))
        return float(opts[self.rule][1])

    def carry_out(self, job, task, others: list, rng=None) -> dict:
        """What happens at the site: the option taken, its energy, hover time and whether the job met its deadline."""
        helper = others[0] if others else None
        opts = self.options(job, task, job["fade"], bool(job["blocked"]), job["edge_wait_s"], helper is not None)
        k = self.choose(job, opts)
        seconds, wh, helper_wh = opts[k]
        return {"option": k, "wh": float(wh), "hover_s": float(seconds) if job["on_site"] else 0.0,
                "on_time": bool(seconds <= job["deadline_s"]), "helper": helper if k == "hand over" else None,
                "helper_wh": float(helper_wh)}
