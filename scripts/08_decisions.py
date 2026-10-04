"""Phase 6: mission feasibility decisions - what-if evaluation and fleet simulation.

Usage: python scripts/08_decisions.py     (about 3 minutes; development data only)
Needs the held-out distributions written by scripts/07_uncertainty.py.

1. Independence check: are the battery model's and the mission model's errors
   related, and is the predicted margin honest on real (battery, own flight) pairs?
   (The feasibility probability assumes the two errors are independent.)
2. What-if evaluation: real held-out battery states paired with real held-out
   missions; four policies compared on unsafe approvals vs wasted refusals,
   over all pairs, borderline pairs and pre-flight states only.
3. Is P(success) honest? Predicted probability vs observed success rate.
4. Threshold table for tau.
5. Fleet simulation on real battery chains.

Writes results/decisions/*.csv and figures/*.png.
"""
import json

import numpy as np
import pandas as pd

from endurosense.config import data_path, load_config, set_seed
from endurosense.data.load import load_processed
from endurosense.data.split import DEV, select
from endurosense import decision_plots as DP
from endurosense.plots import apply_style
from endurosense.scheduler import FleetSimulator, batteries_from, compare_policies, task_pool
from endurosense import whatif as W

OUT = data_path("results") / "decisions"
UNC = data_path("results") / "uncertainty"


def main() -> None:
    set_seed()
    apply_style()
    cfg = load_config()
    dcfg, fcfg, levels = cfg["decision"], cfg["fleet"], np.array(cfg["uncertainty"]["quantiles"])
    rng = np.random.default_rng(cfg["seed"])
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    # held-out battery states -------------------------------------------------
    oof = pd.read_parquet(UNC / "model_a_oof_distribution.parquet")
    q_a = pd.read_parquet(UNC / "model_a_oof_quantiles.parquet")
    feats = pd.read_parquet(data_path("features") / "model_a.parquet")
    a = feats.loc[oof.index].assign(fold=oof["fold"].astype(int))
    typical_w = W.typical_power_by_fold(a)
    states, pre = W.decision_states(a, dcfg["state_every_s"]), W.preflight_states(a)

    # held-out missions -----------------------------------------------------------
    parts = pd.read_parquet(UNC / "model_b_oof_parts.parquet")
    fb = select(pd.read_parquet(data_path("features") / "model_b_flights.parquet"), DEV)
    missions, q_b = W.build_missions(parts, fb, levels)
    print(f"{len(states):,} battery states ({len(pre)} pre-flight) from {a['battery_chain'].nunique()} chains; "
          f"{len(missions)} missions ({(missions.sorties == 1).sum()} single flights, {(missions.sorties > 1).sum()} multi-sortie)", flush=True)

    # 1. independence of the two models' errors -------------------------------------
    indep = W.real_pair_check(pre, parts.index, q_a, missions, q_b, levels, rng)
    (OUT / "error_independence.json").write_text(json.dumps(indep, indent=1))
    print(f"error correlation (battery model vs mission model, {indep['flights']} real flights): "
          f"{indep['correlation']:+.3f} [{indep['ci_low']:+.3f}, {indep['ci_high']:+.3f}], rank {indep['rank_correlation']:+.3f} "
          f"(p = {indep['rank_p_value']:.3f})\n"
          f"  margin error spread {indep['margin_error_sd_actual_wh']:.2f} Wh (independence would give "
          f"{indep['margin_error_sd_if_independent_wh']:.2f}); 90% range of the margin covers "
          f"{indep['margin_90_range_covers']:.1%}, truth below it {indep['margin_truth_below_90_range']:.1%}", flush=True)

    # 2. what-if evaluation --------------------------------------------------------------
    sets = {}
    pairs = W.make_pairs(states, missions, dcfg["n_pairs"], rng)
    sets["all pairs"] = pairs
    big = W.make_pairs(states, missions, 8 * dcfg["n_pairs"], rng)
    sets["borderline pairs"] = big[big["true_margin_wh"].abs() < dcfg["borderline_wh"]].head(dcfg["n_pairs"]).reset_index(drop=True)
    sets["pre-flight states"] = W.make_pairs(pre, missions, dcfg["n_pairs"], rng)
    tau_list = [0.90, dcfg["tau"], 0.99]
    summaries, points, gains, all_scores = [], [], [], {}
    for name, p in sets.items():
        src = pre if name == "pre-flight states" else states
        sc = W.policy_scores(p, src, q_a, missions, q_b, levels, typical_w)
        all_scores[name] = (sc, p)
        summaries.append(W.summary_table(sc, p).assign(pairs=name, n=len(p), feasible_share=p["feasible"].mean()))
        points.append(W.operating_points(sc, p, tau_list).assign(pairs=name))
        gains += [W.paired_gain(sc, p, "P3", other).assign(pairs=name, comparison=f"P3 minus {other}") for other in ("P2", "P1")]
    summary, points, gains = pd.concat(summaries), pd.concat(points), pd.concat(gains)
    gains.to_csv(OUT / "paired_gain.csv", index=False)
    # the margins that give P1 and P2 EnduroSense's unsafe rate here; the final evaluation applies them, unchanged, to test data
    (OUT / "matched_margins.json").write_text(json.dumps(
        {name: {str(t): m for t, m in W.matched_margins(sc_, p_, tau_list).items()} for name, (sc_, p_) in all_scores.items()}, indent=1))
    summary.to_csv(OUT / "policy_summary.csv", index=False)
    points.to_csv(OUT / "operating_points.csv", index=False)
    sc, p = all_scores["all pairs"]
    pd.concat([W.tradeoff_curve(sc[c].to_numpy(), p["feasible"].to_numpy()).assign(policy=W.POLICIES[c]) for c in sc]).to_csv(
        OUT / "tradeoff_curves.csv", index=False)

    # 3. is the probability honest? ---------------------------------------------------------
    rel = W.probability_reliability(sc["P3"].to_numpy(), p["feasible"].to_numpy(), p["battery_chain"].to_numpy())
    rel.to_csv(OUT / "probability_reliability.csv", index=False)

    # 4. threshold table ---------------------------------------------------------------------
    taus = [0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995]
    tau_table = pd.DataFrame([{"tau": t, **W.rates(sc["P3"] >= t, p["feasible"].to_numpy()),
                               **{f"preflight_{k}": v for k, v in W.rates(
                                   all_scores["pre-flight states"][0]["P3"] >= t,
                                   all_scores["pre-flight states"][1]["feasible"].to_numpy()).items()}} for t in taus])
    tau_table.to_csv(OUT / "tau_table.csv", index=False)

    # what the mistakes at the chosen tau look like, and the pairs themselves (reused by the demo)
    p.join(sc).to_parquet(OUT / "whatif_pairs.parquet")
    mistakes = W.mistakes_at(sc, p, dcfg["tau"])
    (OUT / "mistakes_at_tau.json").write_text(json.dumps(mistakes, indent=1))

    # 5. fleet simulation -----------------------------------------------------------------------
    full = load_processed("chains").set_index("battery_chain")["starts_full"]
    bats = batteries_from(a[a["battery_chain"].map(full)])
    sim = FleetSimulator(bats, a, q_a, missions, q_b, levels, typical_w, fcfg["drones"])
    f_all = p["feasible"].to_numpy()
    unsafe_p3 = W.rates(sc["P3"] >= dcfg["tau"], f_all)["unsafe_approval_rate"]
    thr = {"P1 minutes left (brief)": ("P1", 1.0), "P2 energy, point estimates": ("P2", 0.0),
           "P1 + margin (same unsafe rate as P3)": ("P1", W.approved_at_unsafe(sc["P1"].to_numpy(), f_all, unsafe_p3)[1]),
           "P2 + margin (same unsafe rate as P3)": ("P2", W.approved_at_unsafe(sc["P2"].to_numpy(), f_all, unsafe_p3)[1]),
           f"P3 EnduroSense (tau = {dcfg['tau']})": ("P3", dcfg["tau"]), "P4 oracle": ("P4", 0.0)}
    pool = task_pool(missions, fcfg["two_sortie_share"], cfg["seed"])
    fleet = compare_policies(sim, thr, fcfg["days"], fcfg["tasks_per_day"], pool, cfg["seed"])
    fleet.to_csv(OUT / "fleet_simulation.csv", index=False)

    DP.tradeoff(all_scores, dcfg["tau"], OUT / "figures" / "tradeoff.png")
    DP.probability_reliability(rel, OUT / "figures" / "probability_reliability.png")
    DP.fleet(fleet, OUT / "figures" / "fleet_simulation.png")
    pd.set_option("display.width", 220)
    print("\nranking quality and approvals at fixed unsafe rates:\n" + summary[["pairs", "policy", "auc", "approved_at_1pct_unsafe", "approved_at_2pct_unsafe", "approved_at_5pct_unsafe"]].round(3).to_string(index=False))
    print("\nEnduroSense minus the other policies, same pairs:\n" + gains[["pairs", "comparison", "metric", "gain", "ci_low", "ci_high", "share_of_resamples_positive"]].round(3).to_string(index=False))
    print("\noperating points (all pairs):\n" + points[points.pairs == "all pairs"].drop(columns="pairs").round(4).to_string(index=False))
    print("\nis P(success) honest?\n" + rel.round(3).to_string(index=False))
    print("\ntau table:\n" + tau_table.round(4).to_string(index=False))
    print("\nmistakes at the chosen tau:", mistakes)
    print("\nfleet simulation:\n" + fleet.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
