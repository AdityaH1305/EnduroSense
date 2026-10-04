"""Phase 7: the final evaluation on the locked test set.

Usage:
    python scripts/09_final_test.py --final                  open the test set and evaluate (every opening is logged)
    python scripts/09_final_test.py --final --reason "..."   required when code, config or split changed since the last opening
    python scripts/09_final_test.py --rehearsal              the same code on development readings (fold 0); opens nothing

Nothing is fitted, tuned or calibrated here. The script loads the models that
`scripts/run_all.py` built from development data and scores them on batteries
and flights they have never seen. What is measured and what counts as success
was fixed beforehand in docs/final_evaluation_protocol.md.

1. Model A: point error of all nine models, test next to cross-validation.
2. Model A: coverage and width of the calibrated ranges.
3. Model B: mission error per variant; seen route vs unseen routes.
4. Model B: coverage and width of the calibrated mission ranges.
5. Decisions: the Phase 6 what-if evaluation and fleet simulation on test data, with
   the margins of the comparison policies fixed from development data.

Writes results/final/* (results/final_rehearsal/* in rehearsal mode).
"""
import argparse
import json
import pickle
import sys

import numpy as np
import pandas as pd

from endurosense import decision_plots as DP
from endurosense import whatif as W
from endurosense.access import ReasonRequired, code_fingerprint, register_opening
from endurosense.config import data_path, load_config, set_seed
from endurosense.data.load import load_processed
from endurosense.data.split import DEV, TEST, load_split, select
from endurosense.evaluate import metrics, paired_comparison
from endurosense.features.model_a import sequence_windows
from endurosense.models.io import load_model_a, load_model_b
from endurosense.models.model_b import FAMILIES, MAIN
from endurosense.models.sequence import WindowStore
from endurosense.plots import INK_MUTED, SERIES, apply_style, plt, save
from endurosense.scheduler import FleetSimulator, batteries_from, compare_policies, task_pool
from endurosense.uncertainty import battery as UA
from endurosense.uncertainty import metrics as UM
from endurosense.uncertainty import mission as UB

MAIN_A = "GRU ensemble, calibrated (main)"


# ------------------------------------------------------------------ helpers
def evaluation_part(df: pd.DataFrame, rehearsal: bool) -> pd.DataFrame:
    """Test rows; in rehearsal mode the development rows of fold 0 instead."""
    if rehearsal:
        fold = df["flight"].astype(str).map(load_split()["dev_fold"])
        return df[(fold == 0).to_numpy()]
    return select(df, TEST, final=True)


def chain_ci(values, chains, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """95% interval of a mean (e.g. a coverage) from resampling whole battery chains."""
    g = pd.DataFrame({"v": np.asarray(values, float), "c": np.asarray(chains)}).groupby("c")["v"].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(n), (n_boot, len(n)))
    lo, hi = np.percentile(s[idx].sum(1) / n[idx].sum(1), [2.5, 97.5])
    return float(lo), float(hi)


def range_summary(name, y, qv, levels, chains) -> dict:
    """Coverage, width and one-sided misses of calibrated ranges, with chain-resampled intervals."""
    y, chains = np.asarray(y, float), np.asarray(chains)
    t = UM.interval_table(y, qv, levels, chains).set_index("nominal")
    lo, hi = UM._interp_rows(qv, levels, 0.05), UM._interp_rows(qv, levels, 0.95)
    ci = chain_ci((y >= lo) & (y <= hi), chains)
    row = {"model": name, "rows": int(len(y)), "chains": int(len(np.unique(chains))),
           "mae_median": float(np.mean(np.abs(UM._interp_rows(qv, levels, 0.5) - y))), "pinball": UM.pinball_loss(y, qv, levels)}
    for c in t.index:
        row[f"cov{int(c * 100)}"] = t.loc[c, "coverage"]
    row.update({"cov90_ci_low": ci[0], "cov90_ci_high": ci[1], "cov90_by_chain": t.loc[0.9, "coverage_by_group"],
                "width90": t.loc[0.9, "mean_width"], "truth_below_90_range": t.loc[0.9, "too_high"],
                "truth_above_90_range": t.loc[0.9, "too_low"]})
    return row


# ------------------------------------------------------------------ Model A
def model_a(ev, dev, windows, levels, target, out):
    cv = pd.read_csv(data_path("results") / "model_a" / "cv_metrics.csv")
    y, chains, sup = ev[target].to_numpy(), ev["battery_chain"].to_numpy(), UA.supported(ev)
    air_w = float(dev.loc[dev["motors_on"] == 1, "p"].median())          # typical flying power, from development data
    flying = (ev["motors_on"].to_numpy() == 1) & (ev["p_mean_30s"].to_numpy() >= 100)
    power = np.where(flying, ev["p_mean_30s"].to_numpy(), air_w)

    preds = {r["model"]: load_model_a(r["model"], r["kind"], windows).predict(ev) for _, r in cv.iterrows()}
    cal = UA.load_calibrated(data_path("models") / "model_a" / "model_a_calibrated.pt", windows)
    evs = ev[sup]
    qv = cal.quantiles(evs)
    main = np.full(len(ev), np.nan)
    main[sup] = UM._interp_rows(qv, levels, 0.5)
    preds[MAIN_A] = main

    ref, rows = preds["Voltage lookup"], []
    for name, p in preds.items():
        ok = sup if name == MAIN_A else np.ones(len(ev), bool)           # the main model abstains without a pre-flight voltage
        m, ms = metrics(y[ok], p[ok]), metrics(y[sup], p[sup])
        c = paired_comparison(y[sup], p[sup], ref[sup], chains[sup])
        r = cv[cv["model"] == name]
        rows.append({"model": name, "kind": r["kind"].iloc[0] if len(r) else "ensemble",
                     "cv_mae": float(r["cv_mae"].iloc[0]) if len(r) else np.nan,
                     "test_mae": m["mae"] if name != MAIN_A else np.nan, "test_rmse": m["rmse"] if name != MAIN_A else np.nan,
                     "test_bias": ms["bias"], "test_mae_supported": ms["mae"],
                     "minutes_mae": float(np.mean(np.abs(p[ok] / power[ok] * 60.0 - ev["remaining_min"].to_numpy()[ok]))),
                     "vs_lookup_diff": c["mae_diff"], "vs_lookup_ci_low": c["ci_low"], "vs_lookup_ci_high": c["ci_high"],
                     "chains_better_share": c["share_groups_better"]})
    table = pd.DataFrame(rows).sort_values("test_mae_supported").reset_index(drop=True)
    table.to_csv(out / "model_a_test_metrics.csv", index=False)

    ys, cs = evs[target].to_numpy(), evs["battery_chain"].to_numpy()
    ranges = pd.DataFrame([range_summary(MAIN_A, ys, qv, levels, cs)])
    ranges["abstained_rows"], ranges["abstained_chains"] = int((~sup).sum()), int(len(set(chains) - set(cs)))
    ranges.to_csv(out / "model_a_test_ranges.csv", index=False)
    UM.coverage_by(ys, qv, levels, cs).to_csv(out / "model_a_test_coverage_by_chain.csv", index=False)
    rel = UM.reliability(ys, qv, levels, cs)
    q_a = pd.DataFrame(qv, index=evs.index, columns=[f"L{q:.3f}" for q in levels])
    q_a.to_parquet(out / "model_a_test_quantiles.parquet")
    return table, ranges, rel, evs, q_a, air_w, preds


# ------------------------------------------------------------------ Model B
def model_b(rehearsal, levels, out):
    meta = load_processed("flights").set_index("flight")[["date", "route"]]
    fb_all = pd.read_parquet(data_path("features") / "model_b_flights.parquet").join(meta, on="flight")
    legs_all = pd.read_parquet(data_path("features") / "model_b_legs.parquet")
    fb = evaluation_part(fb_all, rehearsal).reset_index(drop=True)
    legs = legs_all[legs_all["flight"].isin(fb["flight"])]
    seen_routes = set(select(fb_all, DEV)["route"]) if not rehearsal else set(fb_all["route"])
    f = fb.set_index("flight")
    y = f["total_wh"]
    groups = {"all test flights": np.ones(len(f), bool), "route seen in development": f["route"].isin(seen_routes).to_numpy(),
              "unseen routes": (~f["route"].isin(seen_routes)).to_numpy()}
    for r in sorted(set(f["route"]) - seen_routes):
        groups[f"unseen route {r}"] = (f["route"] == r).to_numpy()

    cv = pd.read_csv(data_path("results") / "model_b" / "mission_cv.csv").set_index("model")
    rows, preds = [], {}
    for name in FAMILIES + ["Best component each", MAIN]:
        p = load_model_b(name).predict_flights(fb, legs).reindex(f.index)
        preds[name] = p
        for g, mask in groups.items():
            if mask.sum():
                e = (p - y)[mask]
                rows.append({"model": name, "flights_group": g, "flights": int(mask.sum()), "mae_wh": float(e.abs().mean()),
                             "mape_pct": float((e.abs() / y[mask]).mean() * 100), "bias_wh": float(e.mean()),
                             "cv_mape_pct": float(cv.loc[name, "mape_pct"]) if g == "all test flights" else np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(out / "model_b_test_metrics.csv", index=False)

    with open(data_path("models") / "model_b" / "model_b_calibrated.pkl", "rb") as fh:
        calb = pickle.load(fh)
    parts = UB.predicted_parts(calb.model, fb, legs)
    assert np.allclose(parts.sum(axis=1), preds[MAIN].loc[parts.index]), "calibrated model and saved main model disagree"
    parts_ev = parts.add_prefix("pred_").assign(battery_chain=f.loc[parts.index, "battery_chain"])
    missions, q_b = W.build_missions(parts_ev, fb, levels, calibration=calb.tuples)
    n = len(parts_ev)
    yb, cb = y.loc[parts_ev.index].to_numpy(), parts_ev["battery_chain"].to_numpy()
    rg = [range_summary(f"{MAIN}, calibrated: {g}", yb[m], q_b[:n][m], levels, cb[m]) for g, m in groups.items() if m.sum() >= 3]
    for k in sorted(set(missions["sorties"]) - {1}):
        m = (missions["sorties"] == k).to_numpy()
        rg.append(range_summary(f"{MAIN}, calibrated: {k}-sortie missions", missions["true_wh"].to_numpy()[m], q_b[m], levels,
                                missions["battery_chain"].to_numpy()[m]))
    ranges = pd.DataFrame(rg)
    ranges.to_csv(out / "model_b_test_ranges.csv", index=False)
    rel = UM.reliability(yb, q_b[:n], levels, cb)
    detail = f.loc[parts_ev.index, ["route", "speed", "payload", "alt_cruise_m", "battery_chain", "total_wh"]].assign(
        predicted_wh=preds[MAIN].loc[parts_ev.index], low90_wh=UM._interp_rows(q_b[:n], levels, 0.05),
        high90_wh=UM._interp_rows(q_b[:n], levels, 0.95), seen_route=f.loc[parts_ev.index, "route"].isin(seen_routes))
    detail.to_csv(out / "model_b_test_flights.csv")
    return table, ranges, rel, detail, fb, parts_ev, missions, q_b


# ------------------------------------------------------------------ decisions
def decisions(evs, q_a, air_w, fb, parts_ev, missions, q_b, levels, cfg, out):
    dcfg, fcfg = cfg["decision"], cfg["fleet"]
    rng = np.random.default_rng(cfg["seed"])
    a = evs.assign(fold=0)
    typical_w = pd.Series({0: air_w})
    states, pre = W.decision_states(a, dcfg["state_every_s"]), W.preflight_states(a)
    print(f"\ndecisions: {len(states):,} battery states ({len(pre)} pre-flight) from {a['battery_chain'].nunique()} chains; "
          f"{len(missions)} missions ({(missions.sorties == 1).sum()} single flights, {(missions.sorties > 1).sum()} multi-sortie)", flush=True)

    indep = W.real_pair_check(pre, parts_ev.index, q_a, missions, q_b, levels, rng)
    (out / "error_independence.json").write_text(json.dumps(indep, indent=1))

    dev_margins = {name: {float(t): m for t, m in v.items()} for name, v in
                   json.loads((data_path("results") / "decisions" / "matched_margins.json").read_text()).items()}
    sets = {"all pairs": (states, W.make_pairs(states, missions, dcfg["n_pairs"], rng))}
    big = W.make_pairs(states, missions, 8 * dcfg["n_pairs"], rng)
    sets["borderline pairs"] = (states, big[big["true_margin_wh"].abs() < dcfg["borderline_wh"]].head(dcfg["n_pairs"]).reset_index(drop=True))
    sets["pre-flight states"] = (pre, W.make_pairs(pre, missions, dcfg["n_pairs"], rng))
    tau_list = [0.90, dcfg["tau"], 0.99]
    summaries, points, gains, all_scores = [], [], [], {}
    for name, (src, p) in sets.items():
        sc = W.policy_scores(p, src, q_a, missions, q_b, levels, typical_w)
        all_scores[name] = (sc, p)
        summaries.append(W.summary_table(sc, p).assign(pairs=name, n=len(p), feasible_share=p["feasible"].mean()))
        points.append(W.operating_points(sc, p, tau_list, margins=dev_margins[name]).assign(pairs=name, margins="tuned on development data"))
        points.append(W.operating_points(sc, p, tau_list).assign(pairs=name, margins="matched on these pairs (hindsight)"))
        gains += [W.paired_gain(sc, p, "P3", other).assign(pairs=name, comparison=f"P3 minus {other}") for other in ("P2", "P1")]
    summary, points, gains = pd.concat(summaries), pd.concat(points), pd.concat(gains)
    summary.to_csv(out / "policy_summary.csv", index=False)
    points.to_csv(out / "operating_points.csv", index=False)
    gains.to_csv(out / "paired_gain.csv", index=False)

    sc, p = all_scores["all pairs"]
    f_all = p["feasible"].to_numpy()
    rel = W.probability_reliability(sc["P3"].to_numpy(), f_all, p["battery_chain"].to_numpy())
    rel.to_csv(out / "probability_reliability.csv", index=False)
    scp, pp = all_scores["pre-flight states"]
    taus = [0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995]
    tau_table = pd.DataFrame([{"tau": t, **W.rates(sc["P3"] >= t, f_all),
                               **{f"preflight_{k}": v for k, v in W.rates(scp["P3"] >= t, pp["feasible"].to_numpy()).items()}} for t in taus])
    tau_table.to_csv(out / "tau_table.csv", index=False)
    p.join(sc).to_parquet(out / "whatif_pairs.parquet")
    pp.join(scp).to_parquet(out / "whatif_pairs_preflight.parquet")
    mistakes = {"all pairs": W.mistakes_at(sc, p, dcfg["tau"]), "pre-flight states": W.mistakes_at(scp, pp, dcfg["tau"])}
    (out / "mistakes_at_tau.json").write_text(json.dumps(mistakes, indent=1))

    fleet = None
    full = load_processed("chains").set_index("battery_chain")["starts_full"]
    bats = batteries_from(a[a["battery_chain"].map(full).to_numpy()])
    if len(bats) >= 2:
        sim = FleetSimulator(bats, a, q_a, missions, q_b, levels, typical_w, fcfg["drones"])
        m = dev_margins["all pairs"][dcfg["tau"]]
        thr = {"P1 minutes left (brief)": ("P1", 1.0), "P2 energy, point estimates": ("P2", 0.0),
               "P1 + margin tuned on development data": ("P1", m["P1"]), "P2 + margin tuned on development data": ("P2", m["P2"]),
               f"P3 EnduroSense (tau = {dcfg['tau']})": ("P3", dcfg["tau"]), "P4 oracle": ("P4", 0.0)}
        fleet = compare_policies(sim, thr, fcfg["days"], fcfg["tasks_per_day"], task_pool(missions, fcfg["two_sortie_share"], cfg["seed"]), cfg["seed"])
        fleet["batteries"] = len(bats)
        fleet.to_csv(out / "fleet_simulation.csv", index=False)

    DP.tradeoff(all_scores, dcfg["tau"], out / "figures" / "tradeoff.png", sets=("all pairs", "pre-flight states"))
    DP.probability_reliability(rel, out / "figures" / "probability_reliability.png")
    if fleet is not None:
        DP.fleet(fleet, out / "figures" / "fleet_simulation.png", f"Fleet simulation on {len(bats)} unseen batteries (blue = EnduroSense)")
    return summary, points, gains, rel, tau_table, mistakes, fleet, indep, all_scores


# ------------------------------------------------------------------ figures
def figures(a_table, a_rel, b_rel, b_detail, evs, q_a, levels, target, out, label):
    t = a_table[a_table["cv_mae"].notna()].sort_values("test_mae", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    yy = np.arange(len(t))
    ax.barh(yy - 0.2, t["cv_mae"], height=0.38, color=INK_MUTED, label="cross-validation (development)")
    ax.barh(yy + 0.2, t["test_mae"], height=0.38, color=SERIES[0], label=f"{label} (never seen)")
    ax.set_yticks(yy, t["model"]); ax.set_xlabel("mean absolute error (Wh)")
    lim = 1.25 * float(t.loc[t["model"] != "Fixed capacity (reference)", ["cv_mae", "test_mae"]].max().max())
    ax.set_xlim(0, lim)                                               # the reference baseline runs off the scale; its values are written in
    for yv, col in ((-0.2, "cv_mae"), (0.2, "test_mae")):
        for i, v in enumerate(t[col]):
            ax.text(min(v, lim) - 0.05 if v > lim else v + 0.05, yy[i] + yv, f"{v:.2f}", va="center", ha="right" if v > lim else "left",
                    fontsize=7, color="white" if v > lim else INK_MUTED)
    ax.set_title(f"Model A: energy available, development vs {label}"); ax.legend(fontsize=8)
    save(fig, out / "figures" / "model_a_cv_vs_test.png")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for rel, name, k in ((a_rel, "Model A (energy available)", 0), (b_rel, "Model B (energy required)", 1)):
        a1.plot(rel["level"], rel["observed"], "o-", ms=3, color=SERIES[k], label=name)
    a1.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls="--")
    a1.set_xlabel("predicted probability level"); a1.set_ylabel("how often the truth fell below it")
    a1.set_title(f"Calibration on {label} data (dashed = perfect)"); a1.legend(fontsize=8)
    d = b_detail.sort_values("predicted_wh").reset_index(drop=True)
    x = np.arange(len(d))
    a2.vlines(x, d["low90_wh"], d["high90_wh"], color=SERIES[0], alpha=0.5, lw=3, label="90% range")
    for seen, col, lab in ((True, INK_MUTED, "measured, route seen in development"), (False, SERIES[1], "measured, unseen route")):
        m = (d["seen_route"] == seen).to_numpy()
        if m.any():
            a2.scatter(x[m], d["total_wh"][m], s=14, color=col, zorder=3, label=lab)
    a2.set_xlabel(f"{label} flights, sorted by predicted energy"); a2.set_ylabel("mission energy (Wh)")
    a2.set_title("Model B: predicted ranges vs measured energy"); a2.legend(fontsize=7)
    save(fig, out / "figures" / "calibration_and_missions.png")

    chain = evs.groupby("battery_chain").size().idxmax()             # the longest chain as an example
    g = evs[evs["battery_chain"] == chain].sort_values("t_chain_s")
    q = q_a.loc[g.index].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(g))
    ax.fill_between(x, UM._interp_rows(q, levels, 0.05), UM._interp_rows(q, levels, 0.95), color=SERIES[0], alpha=0.25, label="90% range")
    ax.plot(x, UM._interp_rows(q, levels, 0.5), color=SERIES[0], lw=1.5, label="predicted (median)")
    ax.plot(x, g[target], color=SERIES[1], lw=1.5, label="measured")
    ax.axhline(0, color=INK_MUTED, lw=0.8)
    ax.set_xlabel("seconds of recorded flight on this battery (flights joined end to end)"); ax.set_ylabel("energy above reserve (Wh)")
    ax.set_title(f"One unseen battery followed across its flights (chain {chain})"); ax.legend(fontsize=8)
    save(fig, out / "figures" / "model_a_example_chain.png")


# ------------------------------------------------------------------ success criteria
def criteria(a_table, a_ranges, b_table, b_ranges, summary, points, tau) -> pd.DataFrame:
    a = a_table.set_index("model")
    main, look, count = (a.loc[m, "test_mae_supported"] for m in (MAIN_A, "Voltage lookup", "Energy counting (BMS)"))
    c1 = main < look and main < count
    b = b_table[b_table["model"] == MAIN].set_index("flights_group")["mape_pct"]
    unseen = b.get("unseen routes", np.nan)
    c2 = b["all test flights"] <= 5.0 and not (unseen > 5.0)
    ra, rb = a_ranges.iloc[0], b_ranges.iloc[0]
    c3 = 0.88 <= ra["cov90"] <= 0.92 and 0.88 <= rb["cov90"] <= 0.92
    within = (ra["cov90_ci_low"] <= 0.90 <= ra["cov90_ci_high"]) and (rb["cov90_ci_low"] <= 0.90 <= rb["cov90_ci_high"])
    s = summary[summary["pairs"] == "all pairs"].set_index("policy")["auc"]
    pts = points[(points["pairs"] == "all pairs") & (points["margins"] == "tuned on development data")].set_index("policy")["unsafe_approval_rate"]
    p3, p1, p2 = (s[W.POLICIES[c]] for c in ("P3", "P1", "P2"))
    u3, u1, u2 = pts[f"P3 EnduroSense, tau = {tau}"], pts["P1 minutes left, as in the brief (ratio >= 1)"], pts["P2 energy point estimates (margin >= 0 Wh)"]
    c4 = p3 >= p1 and p3 >= p2 - 0.002 and u3 < u1 and u3 < u2
    return pd.DataFrame([
        {"criterion": "1. Model A beats both non-ML baselines", "met": bool(c1),
         "evidence": f"main model {main:.2f} Wh; voltage lookup {look:.2f} Wh; energy counting {count:.2f} Wh (same readings)"},
        {"criterion": "2. Model B within 5%, including unseen routes", "met": bool(c2),
         "evidence": f"all test flights {b['all test flights']:.2f}%; unseen routes "
                     + (f"{unseen:.2f}%" if np.isfinite(unseen) else "none in this set")},
        {"criterion": "3. 90% ranges cover 88-92%", "met": bool(c3),
         "evidence": f"Model A {ra['cov90']:.1%} (chain-resampled interval {ra['cov90_ci_low']:.1%} to {ra['cov90_ci_high']:.1%}); "
                     f"Model B {rb['cov90']:.1%} ({rb['cov90_ci_low']:.1%} to {rb['cov90_ci_high']:.1%}); "
                     f"90% inside both intervals: {'yes' if within else 'no'}"},
        {"criterion": "4. EnduroSense decides better than minutes-left and point estimates", "met": bool(c4),
         "evidence": f"ranking quality (AUC) P3 {p3:.3f}, P1 {p1:.3f}, P2 {p2:.3f}; unsafe approvals as specified: P3 {u3:.2%}, P1 {u1:.2%}, P2 {u2:.2%}"}])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--final", action="store_true", help="open the locked test set")
    mode.add_argument("--rehearsal", action="store_true", help="run the same code on development rows (fold 0)")
    ap.add_argument("--reason", help="why the test set is being reopened after a change")
    args = ap.parse_args()

    set_seed()
    apply_style()
    cfg = load_config()
    levels, target = np.array(cfg["uncertainty"]["quantiles"]), cfg["model_a_training"]["target"]
    out = data_path("results") / ("final_rehearsal" if args.rehearsal else "final")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    label = "rehearsal (development fold 0)" if args.rehearsal else "test"
    try:
        fp = code_fingerprint() if args.rehearsal else register_opening(out / "test_access_log.json", args.reason)
    except ReasonRequired as e:
        sys.exit(str(e))
    print(f"final evaluation on {label} data; code fingerprint {fp}", flush=True)

    df = pd.read_parquet(data_path("features") / "model_a.parquet")
    series = pd.read_parquet(data_path("features") / "model_a_series.parquet")
    windows = WindowStore(*sequence_windows(series, df))
    dev, ev = select(df, DEV), evaluation_part(df, args.rehearsal)
    print(f"Model A: {len(ev):,} readings from {ev['battery_chain'].nunique()} chains, {ev['flight'].nunique()} flights", flush=True)

    a_table, a_ranges, a_rel, evs, q_a, air_w, _ = model_a(ev, dev, windows, levels, target, out)
    b_table, b_ranges, b_rel, b_detail, fb, parts_ev, missions, q_b = model_b(args.rehearsal, levels, out)
    summary, points, gains, rel, tau_table, mistakes, fleet, indep, _ = decisions(evs, q_a, air_w, fb, parts_ev, missions, q_b, levels, cfg, out)
    figures(a_table, a_rel, b_rel, b_detail, evs, q_a, levels, target, out, "test" if args.final else "rehearsal")
    crit = criteria(a_table, a_ranges, b_table, b_ranges, summary, points, cfg["decision"]["tau"])
    crit.to_csv(out / "success_criteria.csv", index=False)
    (out / "summary.json").write_text(json.dumps({
        "evaluated_on": label, "code_fingerprint": fp, "split_sha256": load_split()["sha256"],
        "settings": {"reserve_v": cfg["battery"]["reserve_v"], "tau": cfg["decision"]["tau"], "interval": cfg["uncertainty"]["interval"]},
        "model_a": {"readings": int(len(ev)), "chains": int(ev["battery_chain"].nunique()), "readings_with_preflight_voltage": int(len(evs))},
        "model_b": {"flights": int(len(parts_ev)), "routes": sorted(b_detail["route"].unique().tolist())},
        "criteria_met": {r["criterion"]: bool(r["met"]) for _, r in crit.iterrows()}}, indent=1))

    pd.set_option("display.width", 250)
    print("\nModel A, point error (Wh):\n" + a_table[["model", "cv_mae", "test_mae", "test_mae_supported", "test_bias", "minutes_mae",
                                                    "vs_lookup_diff", "vs_lookup_ci_low", "vs_lookup_ci_high"]].round(3).to_string(index=False))
    print("\nModel A, calibrated ranges:\n" + a_ranges.round(3).T.to_string(header=False))
    print("\nModel B, mission error:\n" + b_table.round(3).to_string(index=False))
    print("\nModel B, calibrated ranges:\n" + b_ranges[["model", "rows", "chains", "mae_median", "cov50", "cov80", "cov90", "cov90_ci_low", "cov90_ci_high",
                                                       "cov95", "width90", "truth_below_90_range", "truth_above_90_range"]].round(3).to_string(index=False))
    print("\nreal battery / own flight pairs:", json.dumps({k: round(v, 3) for k, v in indep.items()}))
    print("\nranking quality:\n" + summary[["pairs", "policy", "auc", "auc_ci_low", "auc_ci_high", "approved_at_1pct_unsafe", "approved_at_2pct_unsafe",
                                            "approved_at_5pct_unsafe", "feasible_share"]].round(3).to_string(index=False))
    print("\nEnduroSense minus the other policies:\n" + gains[["pairs", "comparison", "metric", "gain", "ci_low", "ci_high", "share_of_resamples_positive"]].round(3).to_string(index=False))
    print("\noperating points:\n" + points[points["margins"] == "tuned on development data"].drop(columns="margins").round(4).to_string(index=False))
    print("\nis P(success) honest?\n" + rel.round(3).to_string(index=False))
    print("\ntau table:\n" + tau_table.round(4).to_string(index=False))
    print("\nmistakes at tau:", json.dumps(mistakes, indent=1))
    if fleet is not None:
        print("\nfleet simulation:\n" + fleet.round(2).to_string(index=False))
    print("\nsuccess criteria:\n" + crit.to_string(index=False))


if __name__ == "__main__":
    main()
