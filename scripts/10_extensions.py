"""Phase 9: the guide's extensions, as simulations on top of the finished models.

Usage:
    python scripts/10_extensions.py                          development batteries and missions -> results/extensions/dev/
    python scripts/10_extensions.py --final --reason "..."   the unseen batteries and missions   -> results/extensions/final/

1. Energy-aware fleet planner (``endurosense.fleet_planner``): five scheduling
   policies on the same simulated working days, with priorities, deadlines,
   payload limits, battery swaps and a charging station.
2. Where to process a task's data (``endurosense.edge``): the EnduroSense planner
   with computing jobs, processed always on board, always on the edge server, or
   by the energy-aware choice; plus a sweep over link quality and server load.

Batteries and missions are real (held-out ranges, measured energies); the
schedule, the tasks' priorities and deadlines, the charging station and all
computing settings are simulated (``config.yaml``, ``extensions``). Nothing is
trained here and no model or earlier result changes.

The criteria for the final run are fixed in ``docs/extensions_protocol.md``.
``--final`` opens the locked test set and is logged in
``results/final/test_access_log.json`` like the final evaluation.
"""
import argparse
import copy
import json
import pickle
import sys

import numpy as np
import pandas as pd

from endurosense import whatif as W
from endurosense.access import ReasonRequired, code_fingerprint, register_opening
from endurosense.config import data_path, load_config, set_seed
from endurosense.data.load import load_processed
from endurosense.data.split import DEV, TEST, select
from endurosense.edge import RULES, Computing
from endurosense.fleet_planner import POLICIES, FleetPlanner, compare
from endurosense.plots import INK_MUTED, SERIES, apply_style, plt, save
from endurosense.scheduler import batteries_from, task_pool
from endurosense.uncertainty import mission as UB

RES = data_path("results")


def with_plan_details(missions: pd.DataFrame) -> pd.DataFrame:
    """Add each mission's payload (heaviest sortie) and flown distance (all sorties) from the recorded flights."""
    f = load_processed("flights").set_index("flight")
    return missions.assign(payload_g=[float(f.loc[list(m), "payload"].max()) for m in missions["flights"]],
                           distance_m=[float(f.loc[list(m), "distance_m"].sum()) for m in missions["flights"]])


def hover_power_w() -> float:
    """Typical power while hovering, measured on development flights."""
    f = select(load_processed("flights"), DEV)
    f = f[f["hover_time_s"] > 3]
    return float((f["hover_energy_wh"] * 3600 / f["hover_time_s"]).median())


def development_inputs(levels):
    unc = RES / "uncertainty"
    oof = pd.read_parquet(unc / "model_a_oof_distribution.parquet")
    q_a = pd.read_parquet(unc / "model_a_oof_quantiles.parquet")
    a = pd.read_parquet(data_path("features") / "model_a.parquet").loc[oof.index].assign(fold=oof["fold"].astype(int))
    parts = pd.read_parquet(unc / "model_b_oof_parts.parquet")
    fb = select(pd.read_parquet(data_path("features") / "model_b_flights.parquet"), DEV)
    missions, q_b = W.build_missions(parts, fb, levels)
    return a, q_a, W.typical_power_by_fold(a), missions, q_b


def test_inputs(levels):
    """The unseen batteries (their saved held-out ranges) and missions, built as scripts/09_final_test.py builds them."""
    q_a = pd.read_parquet(RES / "final" / "model_a_test_quantiles.parquet")
    df = pd.read_parquet(data_path("features") / "model_a.parquet")
    a = select(df, TEST, final=True).loc[q_a.index].assign(fold=0)
    dev = select(df, DEV)
    air_w = float(dev.loc[dev["motors_on"] == 1, "p"].median())
    meta = load_processed("flights").set_index("flight")[["date", "route"]]
    fb = select(pd.read_parquet(data_path("features") / "model_b_flights.parquet").join(meta, on="flight"), TEST, final=True).reset_index(drop=True)
    legs = pd.read_parquet(data_path("features") / "model_b_legs.parquet")
    legs = legs[legs["flight"].isin(fb["flight"])]
    with open(data_path("models") / "model_b" / "model_b_calibrated.pkl", "rb") as fh:
        calb = pickle.load(fh)
    parts = UB.predicted_parts(calb.model, fb, legs)
    parts_ev = parts.add_prefix("pred_").assign(battery_chain=fb.set_index("flight").loc[parts.index, "battery_chain"])
    missions, q_b = W.build_missions(parts_ev, fb, levels, calibration=calb.tuples)
    return a, q_a, pd.Series({0: air_w}), missions, q_b


def criteria(planner: pd.DataFrame, computing: pd.DataFrame) -> pd.DataFrame:
    """The criteria fixed in docs/extensions_protocol.md before the run on unseen batteries."""
    p, c = planner.set_index("code"), computing.set_index("rule")
    s1, s2, s3, s4 = (p.loc[k] for k in ("S1", "S2", "S3", "S4"))
    best = c.loc[["local", "offload"], "jobs_on_time_pct"].idxmax()
    ea, fixed = c.loc["energy-aware"], c.loc[best]
    rows = [
        ("1. The planner gets more priority-weighted work done on time than first come, first served (both with the EnduroSense check)",
         s4["weighted_on_time_pct"] - s4["weighted_on_time_ci"] > s2["weighted_on_time_pct"] + s2["weighted_on_time_ci"],
         f"{s4['weighted_on_time_pct']:.1f}% ± {s4['weighted_on_time_ci']:.1f} against {s2['weighted_on_time_pct']:.1f}% ± {s2['weighted_on_time_ci']:.1f}; "
         f"high priority {s4['high_priority_on_time_pct']:.0f}% against {s2['high_priority_on_time_pct']:.0f}%"),
        ("2. The planner stays safe: at most a tenth of the unsafe missions of the brief's rule, and fewer than the planner with best estimates",
         s4["unsafe_per_100_flown"] <= s1["unsafe_per_100_flown"] / 10 and s4["unsafe_per_100_flown"] < s3["unsafe_per_100_flown"],
         f"unsafe per 100 flown: {s4['unsafe_per_100_flown']:.2f} against {s1['unsafe_per_100_flown']:.2f} (minutes left) and "
         f"{s3['unsafe_per_100_flown']:.2f} (best estimates)"),
        ("3. The energy-aware computing choice meets at least as many job deadlines as the better fixed rule, for no more energy",
         ea["jobs_on_time_pct"] >= fixed["jobs_on_time_pct"] and ea["computing_wh_per_task"] <= fixed["computing_wh_per_task"],
         f"jobs on time {ea['jobs_on_time_pct']:.1f}% against {fixed['jobs_on_time_pct']:.1f}% (always {best}); computing energy "
         f"{ea['computing_wh_per_task']:.2f} against {fixed['computing_wh_per_task']:.2f} Wh per task")]
    return pd.DataFrame(rows, columns=["criterion", "met", "evidence"])


def figure(planner: pd.DataFrame, computing: pd.DataFrame, path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    labels = [POLICIES[c].replace(", ", ",\n") for c in planner["code"]]
    colours = [SERIES[0] if c == "S4" else "#9a9992" for c in planner["code"]]
    for ax, col, title, fmt in ((axes[0], "weighted_on_time_pct", "priority-weighted tasks on time (%)", "{:.0f}"),
                                (axes[1], "unsafe_per_100_flown", "unsafe missions per 100 flown", "{:.2f}")):
        y = np.arange(len(planner))
        ax.barh(y, planner[col], color=colours)
        for yi, v in zip(y, planner[col]):
            ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=8.5)
        ax.set_yticks(y, labels if ax is axes[0] else [""] * len(y)); ax.invert_yaxis(); ax.set_title(title); ax.grid(axis="y", visible=False)
    ax = axes[2]
    x = np.arange(len(computing))
    ax.bar(x - 0.2, computing["computing_wh_per_task"], 0.4, color=SERIES[1], label="computing energy per task (Wh)")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, computing["jobs_on_time_pct"], 0.4, color=SERIES[2], label="jobs on time (%)")
    ax.set_xticks(x, computing["rule"]); ax.set_title("where to process the data (EnduroSense planner)")
    ax.set_ylabel("Wh per task", color=INK_MUTED); ax2.set_ylabel("jobs on time (%)", color=INK_MUTED); ax2.set_ylim(0, 105)
    ax2.spines["right"].set_visible(True); ax.grid(axis="x", visible=False)
    fig.legend(loc="lower right", bbox_to_anchor=(0.99, -0.06), ncol=2, fontsize=8.5)
    fig.tight_layout()
    save(fig, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", action="store_true", help="run on the unseen batteries and missions (opens the locked test set)")
    ap.add_argument("--reason", help="why the test set is being opened again (required: the code changed since the last opening)")
    ap.add_argument("--days", type=int, help="override the number of simulated days (quick checks only)")
    args = ap.parse_args()

    set_seed()
    cfg = load_config()
    levels, tau, seed = np.array(cfg["uncertainty"]["quantiles"]), cfg["decision"]["tau"], cfg["seed"]
    pcfg, ccfg = dict(cfg["extensions"]["planner"]), dict(cfg["extensions"]["computing"])
    days_p, days_c = args.days or pcfg["days"], args.days or ccfg["days"]
    out = RES / "extensions" / ("final" if args.final else "dev")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    try:
        fp = register_opening(RES / "final" / "test_access_log.json", args.reason) if args.final else code_fingerprint()
    except ReasonRequired as e:
        sys.exit(str(e))
    print(f"extensions on {'unseen (test)' if args.final else 'development'} data; code fingerprint {fp}", flush=True)

    a, q_a, typical_w, missions, q_b = (test_inputs if args.final else development_inputs)(levels)
    missions = with_plan_details(missions)
    full = load_processed("chains").set_index("battery_chain")["starts_full"]
    bats = batteries_from(a[a["battery_chain"].map(full).to_numpy()])
    pool = task_pool(missions, pcfg["two_sortie_share"], seed)
    hover_w = hover_power_w()
    print(f"{len(bats)} batteries, {len(missions)} missions ({(missions['sorties'] == 1).sum()} single flights); hover power {hover_w:.0f} W", flush=True)

    # 1. the planner ---------------------------------------------------------------------------
    planner = compare(FleetPlanner(bats, a, q_a, missions, q_b, levels, typical_w, pcfg, tau), list(POLICIES), pool, days_p, seed)
    planner.to_csv(out / "planner.csv", index=False)
    print("\nfleet planner:\n" + planner.round(2).to_string(index=False), flush=True)

    # 2. where to process the data -------------------------------------------------------------
    rows = []
    for rule in RULES:
        fp_ = FleetPlanner(bats, a, q_a, missions, q_b, levels, typical_w, pcfg, tau, computing=Computing(ccfg, hover_w, rule))
        rows.append(compare(fp_, ["S4"], pool, days_c, seed).assign(rule=rule))
    computing = pd.concat(rows, ignore_index=True)
    computing.to_csv(out / "computing.csv", index=False)
    print("\nwhere to process the data:\n" + computing[["rule", "computing_wh_per_task", "jobs_on_time_pct", "local_pct", "offload_pct",
                                                         "hand over_pct", "weighted_on_time_pct", "unsafe_per_100_flown"]].round(2).to_string(index=False), flush=True)

    sweep = []
    for snr in (10, 20, 30):
        for busy in (0.1, 0.3, 0.6):
            c = copy.deepcopy(ccfg)
            c.update(snr_db_at_100m=snr, edge_busy_prob=busy)
            for rule in RULES:
                fp_ = FleetPlanner(bats, a, q_a, missions, q_b, levels, typical_w, pcfg, tau, computing=Computing(c, hover_w, rule))
                r = compare(fp_, ["S4"], pool, max(days_c // 4, 1), seed).iloc[0]
                sweep.append({"snr_db_at_100m": snr, "edge_busy_prob": busy, "rule": rule, "computing_wh_per_task": r["computing_wh_per_task"],
                              "jobs_on_time_pct": r["jobs_on_time_pct"], "local_pct": r["local_pct"], "offload_pct": r["offload_pct"],
                              "hand over_pct": r["hand over_pct"]})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(out / "computing_sensitivity.csv", index=False)
    print("\nsensitivity:\n" + sweep.round(2).to_string(index=False), flush=True)

    crit = criteria(planner, computing)
    crit.to_csv(out / "criteria.csv", index=False)
    print("\ncriteria:\n" + crit.to_string(index=False), flush=True)
    figure(planner, computing, out / "figures" / "extensions.png")
    (out / "summary.json").write_text(json.dumps({
        "data": "unseen (test) batteries and missions" if args.final else "development batteries and missions (held-out predictions)",
        "code_fingerprint": fp, "batteries": len(bats), "missions": int(len(missions)), "hover_power_w": hover_w,
        "days": {"planner": days_p, "computing": days_c}, "tau": tau, "settings": cfg["extensions"],
        "simulated": "task arrivals, priorities, deadlines, payload limits, charging station and every computing setting; "
                     "batteries and missions are recorded"}, indent=1))


if __name__ == "__main__":
    main()
