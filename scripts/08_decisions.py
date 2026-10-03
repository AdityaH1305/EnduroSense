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
from scipy.stats import spearmanr

from endurosense.config import data_path, load_config, set_seed
from endurosense.data.load import load_processed
from endurosense.data.split import DEV, select
from endurosense.feasibility import p_success_batch
from endurosense.plots import INK_MUTED, SERIES, apply_style, plt, save
from endurosense.scheduler import FleetSimulator, batteries_from
from endurosense import whatif as W

OUT = data_path("results") / "decisions"
UNC = data_path("results") / "uncertainty"
COLOR = {"P1": SERIES[1], "P2": SERIES[3], "P3": SERIES[0], "P4": INK_MUTED}


def operating_points(scores, pairs, tau_list) -> pd.DataFrame:
    """Each policy as literally specified, plus P1/P2 with the margin that gives them
    the same unsafe-approval rate as EnduroSense (a like-for-like comparison)."""
    f = pairs["feasible"].to_numpy()
    rows = [{"policy": "P1 minutes left, as in the brief (ratio >= 1)", **W.rates(scores["P1"] >= 1.0, f)},
            {"policy": "P2 energy point estimates (margin >= 0 Wh)", **W.rates(scores["P2"] >= 0.0, f)}]
    for tau in tau_list:
        r = W.rates(scores["P3"] >= tau, f)
        rows.append({"policy": f"P3 EnduroSense, tau = {tau}", **r})
        for col, name in (("P1", "P1 with margin"), ("P2", "P2 with margin")):
            tpr, thr = W.approved_at_unsafe(scores[col].to_numpy(), f, r["unsafe_approval_rate"])
            rows.append({"policy": f"   {name} matched to P3 tau = {tau} (threshold {thr:.2f})",
                         **W.rates(scores[col] >= thr, f)})
    rows.append({"policy": "P4 oracle", **W.rates(scores["P4"] >= 0.0, f)})
    return pd.DataFrame(rows)


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
    mid = int(np.argmin(np.abs(levels - 0.5)))
    both = pre[pre["flight"].isin(parts.index)]                    # the battery just before the flight it then flew
    own = pd.Series(np.arange(len(parts)), index=parts.index)[both["flight"]].to_numpy()   # single-flight missions come first
    qa_own, qb_own = q_a.loc[both.index].to_numpy(), q_b[own]
    true_a, true_b = both["remaining_wh"].to_numpy(), missions["true_wh"].to_numpy()[own]
    err_a, err_b = qa_own[:, mid] - true_a, qb_own[:, mid] - true_b
    corr = float(np.corrcoef(err_a, err_b)[0, 1])
    boot = [np.corrcoef(err_a[i], err_b[i])[0, 1] for i in (rng.integers(0, len(err_a), len(err_a)) for _ in range(2000))]
    rank = spearmanr(err_a, err_b)
    # where the true margin falls in the predicted margin distribution: P(A - B <= truth) = 1 - P(A >= B + truth)
    pit = 1.0 - p_success_batch(qa_own, qb_own + (true_a - true_b)[:, None], levels)
    indep = {"flights": int(len(both)), "correlation": corr, "ci_low": float(np.percentile(boot, 2.5)),
             "ci_high": float(np.percentile(boot, 97.5)), "rank_correlation": float(rank.statistic), "rank_p_value": float(rank.pvalue),
             "battery_error_sd_wh": float(err_a.std()), "mission_error_sd_wh": float(err_b.std()),
             "margin_error_sd_if_independent_wh": float(np.hypot(err_a.std(), err_b.std())),
             "margin_error_sd_actual_wh": float((err_a - err_b).std()),
             "margin_90_range_covers": float(np.mean((pit >= 0.05) & (pit <= 0.95))),
             "margin_truth_below_90_range": float(np.mean(pit < 0.05))}
    (OUT / "error_independence.json").write_text(json.dumps(indep, indent=1))
    print(f"error correlation (battery model vs mission model, {indep['flights']} real flights): "
          f"{corr:+.3f} [{indep['ci_low']:+.3f}, {indep['ci_high']:+.3f}], rank {rank.statistic:+.3f} (p = {rank.pvalue:.3f})\n"
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
        points.append(operating_points(sc, p, tau_list).assign(pairs=name))
        gains += [W.paired_gain(sc, p, "P3", other).assign(pairs=name, comparison=f"P3 minus {other}") for other in ("P2", "P1")]
    summary, points, gains = pd.concat(summaries), pd.concat(points), pd.concat(gains)
    gains.to_csv(OUT / "paired_gain.csv", index=False)
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
    approve = (sc["P3"] >= dcfg["tau"]).to_numpy()
    refused, unsafe = p.loc[~approve & p["feasible"], "true_margin_wh"], p.loc[approve & ~p["feasible"], "true_margin_wh"]
    mistakes = {"tau": dcfg["tau"], "wasted_refusals": int(len(refused)), "refused_margin_wh_median": float(refused.median()),
                "refused_margin_wh_p90": float(refused.quantile(0.9)), "refused_share_within_10wh": float((refused < 10).mean()),
                "unsafe_approvals": int(len(unsafe)), "unsafe_shortfall_wh_worst": float(unsafe.min()) if len(unsafe) else 0.0,
                "unsafe_shortfall_wh_median": float(unsafe.median()) if len(unsafe) else 0.0}
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
    single_idx, two_idx = np.flatnonzero(missions.sorties == 1), np.flatnonzero(missions.sorties == 2)
    n_two = int(round(len(single_idx) * fcfg["two_sortie_share"] / (1 - fcfg["two_sortie_share"])))
    pool = np.r_[single_idx, np.random.default_rng(cfg["seed"]).choice(two_idx, n_two)]
    fleet_rows = []
    for name, (pol, t) in thr.items():
        d = sim.run(pol, t, fcfg["days"], fcfg["tasks_per_day"], pool, cfg["seed"])
        se = lambda c: 1.96 * d[c].std() / np.sqrt(len(d))
        fleet_rows.append({"policy": name, "threshold": t,
                           "completed_per_day": d["completed"].mean(), "completed_ci": se("completed"),
                           "unsafe_per_day": d["unsafe"].mean(), "unsafe_ci": se("unsafe"),
                           "unsafe_per_100_missions": 100 * d["unsafe"].sum() / d["completed"].sum(),
                           "dropped_per_day": d["dropped"].mean(), "swaps_per_day": d["swaps"].mean(), "swaps_ci": se("swaps"),
                           "energy_left_at_swap_wh": d["leftover_wh"].mean(),
                           "missions_per_battery": d["completed"].sum() / d["swaps"].sum()})
    fleet = pd.DataFrame(fleet_rows)
    fleet.to_csv(OUT / "fleet_simulation.csv", index=False)

    figures(all_scores, rel, fleet, dcfg["tau"])
    pd.set_option("display.width", 220)
    print("\nranking quality and approvals at fixed unsafe rates:\n" + summary[["pairs", "policy", "auc", "approved_at_1pct_unsafe", "approved_at_2pct_unsafe", "approved_at_5pct_unsafe"]].round(3).to_string(index=False))
    print("\nEnduroSense minus the other policies, same pairs:\n" + gains[["pairs", "comparison", "metric", "gain", "ci_low", "ci_high", "share_of_resamples_positive"]].round(3).to_string(index=False))
    print("\noperating points (all pairs):\n" + points[points.pairs == "all pairs"].drop(columns="pairs").round(4).to_string(index=False))
    print("\nis P(success) honest?\n" + rel.round(3).to_string(index=False))
    print("\ntau table:\n" + tau_table.round(4).to_string(index=False))
    print("\nmistakes at the chosen tau:", mistakes)
    print("\nfleet simulation:\n" + fleet.round(2).to_string(index=False))


def figures(all_scores, rel, fleet, tau) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, ("all pairs", "borderline pairs")):
        sc, p = all_scores[name]
        for c in ("P1", "P2", "P3"):
            cur = W.tradeoff_curve(sc[c].to_numpy(), p["feasible"].to_numpy())
            ax.plot(100 * cur["unsafe_approval_rate"], 100 * cur["wasted_refusal_rate"], color=COLOR[c], lw=2, label=W.POLICIES[c])
        r = W.rates(sc["P3"] >= tau, p["feasible"].to_numpy())
        ax.scatter([100 * r["unsafe_approval_rate"]], [100 * r["wasted_refusal_rate"]], s=50, color=COLOR["P3"], zorder=5,
                   edgecolor="white", label=f"EnduroSense at tau = {tau}")
        for c, thr, lab in (("P1", 1.0, "P1 as in the brief"), ("P2", 0.0, "P2 with no margin")):
            r = W.rates(sc[c] >= thr, p["feasible"].to_numpy())
            ax.scatter([100 * r["unsafe_approval_rate"]], [100 * r["wasted_refusal_rate"]], s=50, color=COLOR[c], zorder=5,
                       marker="s", edgecolor="white", label=lab)
        ax.set_xlim(0, 30 if name == "all pairs" else 60); ax.set_ylim(0, 40 if name == "all pairs" else 100)
        ax.set_xlabel("unsafe approvals (% of missions that would fail)")
        ax.set_ylabel("wasted refusals (% of missions that would succeed)")
        ax.set_title(f"{name}: closer to the bottom-left corner is better")
    axes[0].legend(fontsize=7)
    save(fig, OUT / "figures" / "tradeoff.png")

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, ls="--", lw=1)
    ax.plot(rel["mean_predicted"], rel["observed_success"], "o-", color=SERIES[0], label="all pairs")
    ax.plot(rel["mean_predicted"], rel["observed_by_chain"], "o-", color=SERIES[2], label="each battery chain counted equally")
    ax.set_xlabel("predicted P(success)"); ax.set_ylabel("share that actually succeeded")
    ax.set_title("Is P(success) honest?"); ax.legend(fontsize=8)
    save(fig, OUT / "figures" / "probability_reliability.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    y = np.arange(len(fleet))
    for ax, (col, ci, title) in zip(axes, (("unsafe_per_100_missions", None, "unsafe missions per 100 flown"),
                                           ("completed_per_day", "completed_ci", "missions completed per day"),
                                           ("swaps_per_day", "swaps_ci", "battery swaps per day"))):
        ax.barh(y, fleet[col], xerr=fleet[ci] if ci else None, color=[SERIES[0] if p.startswith("P3") else INK_MUTED for p in fleet["policy"]], capsize=3)
        ax.set_yticks(y, fleet["policy"] if ax is axes[0] else [""] * len(y), fontsize=7)
        ax.invert_yaxis(); ax.set_title(title, fontsize=9)
    fig.suptitle("Fleet simulation on real battery chains (blue = EnduroSense)", x=0.01, ha="left")
    save(fig, OUT / "figures" / "fleet_simulation.png")


if __name__ == "__main__":
    main()
