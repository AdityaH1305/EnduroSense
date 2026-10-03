"""Phase 5: calibrated uncertainty for Model A and Model B (development data only).

Usage: python scripts/07_uncertainty.py      (about 10 minutes; the ensemble CV is cached)

Model A (energy available)
    Grouped CV of a GRU deep ensemble that predicts a mean and a spread per reading,
    then conformal calibration by battery chain. Compared: the raw ensemble, the
    calibrated ensemble, and a calibrated constant-width range.
Model B (energy required)
    Held-out per-part errors of the main mission model are replayed, jointly per
    flight, on each mission. Compared: per-part replay vs one error on the total.

Every range is evaluated "leave one fold out": the ranges for a fold's batteries are
calibrated only on the other folds' held-out errors. Final calibrated models (fitted
and calibrated on all development data) are saved for Phase 6.

Writes results/uncertainty/*, models/model_a/model_a_calibrated.pt and
models/model_b/model_b_calibrated.pkl.
"""
import hashlib
import json
import pickle

import numpy as np
import pandas as pd
from scipy.stats import norm

from endurosense.config import ROOT, data_path, load_config, set_seed
from endurosense.data.split import DEV, dev_folds, select
from endurosense.features.model_a import FEATURES, sequence_windows
from endurosense.mission import MissionSpec
from endurosense.models.model_b import MAIN, fit_mission_model, main_choice, mission_factory
from endurosense.models.probabilistic import SequenceEnsemble
from endurosense.models.sequence import WindowStore
from endurosense.plots import INK_MUTED, SERIES, apply_style, plt, save
from endurosense.uncertainty import battery as UA
from endurosense.uncertainty import metrics as UM
from endurosense.uncertainty import mission as UB

OUT = data_path("results") / "uncertainty"


def summarise(name, y, qv, levels, groups) -> dict:
    t = UM.interval_table(y, qv, levels, groups).set_index("nominal")
    row = {"method": name, "pinball": UM.pinball_loss(y, qv, levels),
           "mae_median": float(np.mean(np.abs(UM._interp_rows(qv, levels, 0.5) - y)))}
    for c in t.index:
        row[f"cov{int(c * 100)}"] = t.loc[c, "coverage"]
        row[f"cov{int(c * 100)}_by_group"] = t.loc[c, "coverage_by_group"]
        row[f"width{int(c * 100)}"] = t.loc[c, "mean_width"]
    row["truth_below_90_range"] = t.loc[0.9, "too_high"]
    row["truth_above_90_range"] = t.loc[0.9, "too_low"]
    return row


# ============================================================ Model A
def model_a_oof(dev, windows, gru_cfg, ucfg, target) -> pd.DataFrame:
    """Out-of-fold ensemble mean and spread for every dev row (cached)."""
    h = hashlib.md5()
    h.update(pd.util.hash_pandas_object(dev, index=True).values.tobytes())
    h.update((ROOT / load_config()["split"]["file"]).read_bytes())
    h.update(json.dumps([gru_cfg, ucfg["ensemble_size"], ucfg["ensemble_kind"], load_config()["model_a_training"]],
                        sort_keys=True).encode())
    for f in ("probabilistic.py", "sequence.py"):
        h.update((ROOT / "src" / "endurosense" / "models" / f).read_bytes())
    path = OUT / f"cache_model_a_oof_{h.hexdigest()[:10]}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    oof = pd.DataFrame(index=dev.index, columns=["mu", "sigma", "fold"], dtype=float)
    for k, tr, va in dev_folds(dev):
        ens = SequenceEnsemble(ucfg["ensemble_kind"], windows, n=ucfg["ensemble_size"], features=FEATURES, **gru_cfg)
        mu, sd = ens.fit(tr, target).predict_dist(va)
        oof.loc[va.index, ["mu", "sigma", "fold"]] = np.c_[mu, sd, np.full(len(va), k)]
        print(f"  fold {k}: MAE {np.mean(np.abs(mu - va[target].to_numpy())):.3f} Wh, mean sigma {sd.mean():.2f}", flush=True)
    oof.to_parquet(path)
    return oof


def leave_fold_out_a(y, oof, groups, levels, scale, correction=None) -> np.ndarray:
    """Quantile values for every row, calibrated on the other folds only."""
    return UA.leave_fold_out_quantiles(y, oof["mu"], oof["sigma"], groups, oof["fold"], levels, scale, correction)


def run_model_a(cfg, levels):
    ucfg, target = cfg["uncertainty"], cfg["model_a_training"]["target"]
    df = pd.read_parquet(data_path("features") / "model_a.parquet")
    dev_all = select(df, DEV)
    dev = dev_all[UA.supported(dev_all)]            # the model abstains without a pre-flight voltage reading
    excluded = {"rows": int(len(dev_all) - len(dev)),
                "chains": sorted(int(c) for c in set(dev_all["battery_chain"]) - set(dev["battery_chain"]))}
    (OUT / "model_a_unsupported.json").write_text(json.dumps(excluded, indent=1))
    print(f"Model A: abstaining on {excluded['rows']} dev rows without a pre-flight voltage (chains {excluded['chains']})", flush=True)
    series = pd.read_parquet(data_path("features") / "model_a_series.parquet")
    windows = WindowStore(*sequence_windows(series, df))
    gru_cfg = json.loads((data_path("results") / "model_a" / "best_configs.json").read_text())[
        "GRU" if ucfg["ensemble_kind"] == "gru" else "LSTM"]
    print(f"Model A: {ucfg['ensemble_size']}-member {ucfg['ensemble_kind'].upper()} ensemble {gru_cfg}, "
          f"{len(dev):,} dev rows, {dev['battery_chain'].nunique()} chains", flush=True)
    oof = model_a_oof(dev, windows, gru_cfg, ucfg, target)
    y, groups = dev[target].to_numpy(), dev["battery_chain"].to_numpy()

    variants = {
        "Raw ensemble (uncalibrated)": UA.quantile_values(oof["mu"], oof["sigma"], norm.ppf(levels)),
        "Calibrated, constant width": leave_fold_out_a(y, oof, groups, levels, "constant"),
        "Calibrated, per-reading spread (main)": leave_fold_out_a(y, oof, groups, levels, "sigma"),
    }
    table = pd.DataFrame([summarise(n, y, qv, levels, groups) for n, qv in variants.items()])
    table.to_csv(OUT / "model_a_intervals.csv", index=False)
    main = variants["Calibrated, per-reading spread (main)"]
    # the small-sample correction is a choice: show what each option does to the main method
    pd.DataFrame([summarise(f"correction = {c}", y, leave_fold_out_a(y, oof, groups, levels, "sigma", c), levels, groups)
                  for c in ("none", "smooth", "ceil")]).to_csv(OUT / "model_a_correction_comparison.csv", index=False)

    pd.concat([UM.reliability(y, qv, levels, groups).assign(method=n) for n, qv in variants.items()]).to_csv(
        OUT / "model_a_reliability.csv", index=False)
    mu = oof["mu"].to_numpy()
    sub = {"battery state": np.where(y <= 0, "below reserve", "above reserve"),
           "predicted energy": np.select([mu <= 10, mu <= 30], ["a: <= 10 Wh", "b: 10-30 Wh"], "c: > 30 Wh"),
           "motors": np.where(dev["motors_on"] == 1, "on", "off"),
           "label": dev["label_source"].to_numpy(), "fold": oof["fold"].astype(int).to_numpy()}
    by = pd.concat([UM.coverage_by(y, main, levels, v).assign(subgroup=k) for k, v in sub.items()])
    by.to_csv(OUT / "model_a_coverage_by_subgroup.csv", index=False)
    per_chain = UM.coverage_by(y, main, levels, groups)
    per_chain.to_csv(OUT / "model_a_coverage_by_chain.csv", index=False)

    # final model: fit on all dev data, calibrate on all held-out scores
    final = SequenceEnsemble(ucfg["ensemble_kind"], windows, n=ucfg["ensemble_size"], features=FEATURES, **gru_cfg).fit(dev, target)
    q_hat = UA.calibrate(y, oof["mu"], oof["sigma"], groups, levels)
    UA.save_calibrated(UA.CalibratedBatteryModel(final, levels, q_hat), ucfg["ensemble_kind"], gru_cfg,
                       data_path("models") / "model_a" / "model_a_calibrated.pt")
    pd.DataFrame({"level": levels, "q_hat_sigmas": q_hat, "normal_would_be": norm.ppf(levels)}).to_csv(
        OUT / "model_a_calibration_multipliers.csv", index=False)

    out = dev[["flight", "battery_chain", "time", "phase", target]].copy()
    out["mu"], out["sigma"], out["fold"] = oof["mu"], oof["sigma"], oof["fold"]
    for q in (0.05, 0.5, 0.95):
        out[f"q{int(q * 100):02d}"] = UM._interp_rows(main, levels, q)
    out.to_parquet(OUT / "model_a_oof_distribution.parquet")
    # full held-out distribution of every reading, for the decision evaluation (Phase 6)
    pd.DataFrame(main, index=dev.index, columns=[f"L{q:.3f}" for q in levels]).to_parquet(OUT / "model_a_oof_quantiles.parquet")
    figures_a(variants, y, levels, groups, out, per_chain, target)
    print(table[["method", "mae_median", "cov90", "cov90_by_group", "width90", "truth_below_90_range", "pinball"]].round(3).to_string(index=False))
    print(by[["subgroup", "by", "rows", "coverage", "width", "truth_below_range"]].round(3).to_string(index=False), flush=True)
    return table


def figures_a(variants, y, levels, groups, out, per_chain, target):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, (name, qv) in enumerate(variants.items()):
        r = UM.reliability(y, qv, levels, groups)
        a1.plot(r["level"], r["observed"], "o-", ms=3, color=SERIES[k], label=name)
    a1.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls="--")
    a1.set_xlabel("predicted probability level"); a1.set_ylabel("how often the truth fell below it")
    a1.set_title("Model A calibration (on the dashed line = perfectly calibrated)"); a1.legend(fontsize=7)
    a2.hist(per_chain["coverage"], bins=np.linspace(0, 1, 21), color=SERIES[0], edgecolor="white")
    a2.axvline(0.9, color=SERIES[1], ls="--")
    a2.set_xlabel("coverage of the 90% range within one battery chain"); a2.set_ylabel("battery chains")
    a2.set_title("Coverage per battery (90% is the target on average)")
    save(fig, OUT / "figures" / "model_a_calibration.png")

    chain = out.groupby("battery_chain").size().idxmax()             # the longest chain as an example
    g = out[out["battery_chain"] == chain].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(g))
    ax.fill_between(x, g["q05"], g["q95"], color=SERIES[0], alpha=0.25, label="90% range")
    ax.plot(x, g["q50"], color=SERIES[0], lw=1.5, label="predicted (median)")
    ax.plot(x, g[target], color=SERIES[1], lw=1.5, label="measured")
    ax.axhline(0, color=INK_MUTED, lw=0.8)
    ax.set_xlabel("seconds of recorded flight on this battery (flights joined end to end)")
    ax.set_ylabel("energy above reserve (Wh)")
    ax.set_title(f"Example: one battery followed across its flights (chain {chain}, held out)"); ax.legend(fontsize=8)
    save(fig, OUT / "figures" / "model_a_example_chain.png")


# ============================================================ Model B
def run_model_b(cfg, levels):
    legs = select(pd.read_parquet(data_path("features") / "model_b_legs.parquet"), DEV)
    fb = select(pd.read_parquet(data_path("features") / "model_b_flights.parquet"), DEV)
    choice = main_choice()
    print(f"Model B: {MAIN} { {c: v[0] for c, v in choice.items()} }, {len(fb)} dev flights", flush=True)

    pred, fold = UB.oof_parts(mission_factory(choice), fb, legs, dev_folds(fb))
    tuples = UB.residual_tuples(pred, fb)
    # held-out parts and error sets of every flight, for the decision evaluation (Phase 6)
    pred.add_prefix("pred_").join(tuples.add_prefix("err_")).assign(fold=fold).to_parquet(OUT / "model_b_oof_parts.parquet")
    f = fb.set_index("flight").loc[pred.index]
    y, groups = f["total_wh"].to_numpy(), f["battery_chain"].to_numpy()

    def leave_fold_out_b(method, correction=None):
        return UB.leave_fold_out_quantiles(pred, tuples, fold, levels, method, correction)

    variants = {"One error on the whole mission": leave_fold_out_b("total"),
                "Per-part errors, replayed jointly (main)": leave_fold_out_b("parts")}
    pd.DataFrame([summarise(f"correction = {c}", y, leave_fold_out_b("parts", c), levels, groups)
                  for c in ("none", "smooth", "ceil")]).to_csv(OUT / "model_b_correction_comparison.csv", index=False)
    table = pd.DataFrame([summarise(n, y, qv, levels, groups) for n, qv in variants.items()])
    table.to_csv(OUT / "model_b_intervals.csv", index=False)
    main_qv = variants["Per-part errors, replayed jointly (main)"]
    pd.concat([UM.reliability(y, qv, levels, groups).assign(method=n) for n, qv in variants.items()]).to_csv(
        OUT / "model_b_reliability.csv", index=False)
    by = pd.concat([UM.coverage_by(y, main_qv, levels, f[c].to_numpy()).assign(subgroup=n)
                    for c, n in (("speed", "speed"), ("payload", "payload"), ("alt_cruise_m", "altitude"))])
    by.to_csv(OUT / "model_b_coverage_by_subgroup.csv", index=False)
    tuples.drop(columns="battery_chain").corr().round(3).to_csv(OUT / "model_b_error_correlations.csv")

    final = fit_mission_model(mission_factory(choice), fb, legs)
    calibrated = UB.CalibratedMissionModel(final, tuples, levels, cfg["uncertainty"]["mission_method"])
    with open(data_path("models") / "model_b" / "model_b_calibrated.pkl", "wb") as fh:
        pickle.dump(calibrated, fh)

    examples = [("R1-like loop, 250 g, 8 m/s, 50 m", MissionSpec.loop([140, 198, 119], 250, 8, 50)),
                ("Delivery 300 m, 500 g, 8 m/s, 50 m", MissionSpec.delivery(300, 500, 8, 50)),
                ("Delivery 600 m, 500 g, 8 m/s, 50 m", MissionSpec.delivery(600, 500, 8, 50)),
                ("Delivery 300 m, 500 g, 12 m/s, 100 m", MissionSpec.delivery(300, 500, 12, 100))]
    ex = []
    for label, spec in examples:
        d = calibrated.distribution(spec)
        lo, hi = d.interval(0.9)
        ex.append({"mission": label, "median_wh": d.median, "low90_wh": lo, "high90_wh": hi, "width90_wh": hi - lo,
                   "q99_wh": float(d.quantile(0.99)), "warnings": "; ".join(calibrated.warnings(spec))})
    pd.DataFrame(ex).round(2).to_csv(OUT / "model_b_example_missions.csv", index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, (name, qv) in enumerate(variants.items()):
        r = UM.reliability(y, qv, levels, groups)
        a1.plot(r["level"], r["observed"], "o-", ms=3, color=SERIES[k], label=name)
    a1.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls="--")
    a1.set_xlabel("predicted probability level"); a1.set_ylabel("how often the truth fell below it")
    a1.set_title("Model B calibration"); a1.legend(fontsize=7)
    order = np.argsort(UM._interp_rows(main_qv, levels, 0.5))
    lo, hi = UM._interp_rows(main_qv, levels, 0.05)[order], UM._interp_rows(main_qv, levels, 0.95)[order]
    a2.fill_between(np.arange(len(y)), lo, hi, color=SERIES[0], alpha=0.25, label="90% range")
    inside = (y[order] >= lo) & (y[order] <= hi)
    a2.scatter(np.arange(len(y))[inside], y[order][inside], s=8, color=SERIES[0], label="measured, inside")
    a2.scatter(np.arange(len(y))[~inside], y[order][~inside], s=14, color=SERIES[1], label="measured, outside")
    a2.set_xlabel("held-out flights, sorted by predicted energy"); a2.set_ylabel("mission energy (Wh)")
    a2.set_title("Model B: 90% ranges vs measured energy"); a2.legend(fontsize=7)
    save(fig, OUT / "figures" / "model_b_calibration.png")

    print(table[["method", "mae_median", "cov90", "cov90_by_group", "width90", "truth_above_90_range", "pinball"]].round(3).to_string(index=False))
    print(by[["subgroup", "by", "rows", "coverage", "width"]].round(3).to_string(index=False))
    print(pd.DataFrame(ex).round(2).drop(columns="warnings").to_string(index=False), flush=True)
    return table


def main() -> None:
    set_seed()
    apply_style()
    cfg = load_config()
    levels = np.array(cfg["uncertainty"]["quantiles"])
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    run_model_a(cfg, levels)
    run_model_b(cfg, levels)


if __name__ == "__main__":
    main()
