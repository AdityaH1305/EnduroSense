"""Phase 3: compare algorithms for Model A (energy available) with grouped CV.

Usage: python scripts/05_model_a.py     (about an hour; resumable - finished configs are cached)

Compares, on development data only (test stays locked):
  baselines   Fixed capacity (reference), Energy counting (BMS), Voltage lookup
  the brief   Linear Regression, Random Forest, XGBoost, LSTM, GRU
  extra       XGBoost + physics (voltage-lookup estimate as an input)

Each algorithm's hyperparameters are chosen by mean grouped-CV MAE. Because the
same folds pick and score the configuration, CV numbers are slightly optimistic;
the unbiased estimate is the locked test set in Phase 7.

Writes results/model_a/{search_log.csv, cv_metrics.csv, oof_predictions.parquet,
best_configs.json, error_by_*.csv, figures/*.png} and models/model_a/*.
"""
import hashlib
import json
import pickle

import numpy as np
import pandas as pd
import torch

from endurosense.config import data_path, load_config, ROOT, set_seed
from endurosense.data.load import load_processed
from endurosense.data.split import DEV, select
from endurosense.evaluate import cross_validate, latency_ms, metrics, paired_comparison, single_threaded, size_mb
from endurosense.features.model_a import sequence_windows
from endurosense.models import tabular as T
from endurosense.models.baselines import EnergyCounting, FixedCapacity, VoltageLookup
from endurosense.models.sequence import SequenceModel, WindowStore
from endurosense.plots import INK_MUTED, SERIES, apply_style, plt, save

OUT = data_path("results") / "model_a"
CACHE = OUT / "cache"
MODELS = data_path("models") / "model_a"


def fingerprint(dev: pd.DataFrame) -> str:
    """Identifies everything a cached CV result depends on: the development data
    itself, the locked split, the relevant config sections and the model/feature
    source code. Any change gives a new fingerprint, so stale results are never reused."""
    cfg = load_config()
    h = hashlib.md5()
    h.update(pd.util.hash_pandas_object(dev, index=True).values.tobytes())
    h.update((ROOT / cfg["split"]["file"]).read_bytes())
    h.update(json.dumps({k: cfg[k] for k in ("seed", "battery", "model_a", "model_a_training")},
                        sort_keys=True).encode())
    src = ROOT / "src" / "endurosense"
    for p in sorted(list((src / "models").glob("*.py")) + [src / "features" / "model_a.py", src / "evaluate.py"]):
        h.update(p.read_bytes())
    return h.hexdigest()[:10]


def _key(kind: str, cfg: dict, fp: str) -> str:
    return f"{kind}_{fp}_" + hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def run_cv(kind: str, cfg: dict, factory, dev: pd.DataFrame, target: str, fp: str):
    """Cross-validate one configuration, with an on-disk cache keyed by ``fingerprint``."""
    path = CACHE / f"{_key(kind, cfg, fp)}.parquet"
    side = path.with_suffix(".json")
    if path.exists() and side.exists():
        return pd.read_parquet(path)["oof"], json.loads(side.read_text())
    oof, fold_mae = cross_validate(factory, dev, target)
    oof.rename("oof").to_frame().to_parquet(path)
    side.write_text(json.dumps(fold_mae))
    return oof, fold_mae


def robust_latency_ms(predict, one_row: pd.DataFrame, repeats: int, rounds: int = 5) -> float:
    """Single-row prediction time: the *minimum* of several rounds' medians.

    One timing pass is easily inflated by other work on the machine (a first
    run measured every model 2-4x slower while other scripts ran). The minimum
    over rounds estimates the model's own cost. Run this script on a quiet machine.
    """
    return min(latency_ms(predict, one_row, max(1, repeats // rounds)) for _ in range(rounds))


def measurement_conditions() -> dict:
    """Machine state during timing. Laptops slow the CPU on battery: measured
    latencies were ~3-4x higher unplugged, so timings are only comparable within
    one run and the power state is recorded with them."""
    import datetime
    import os

    out = {"measured_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads()}
    try:
        import psutil

        bat = psutil.sensors_battery()
        out["power_plugged"] = None if bat is None else bool(bat.power_plugged)
        out["battery_percent"] = None if bat is None else bat.percent
    except ImportError:
        out["power_plugged"] = "unknown (psutil not installed)"
    return out


def minutes_from_wh(pred_wh: np.ndarray, df: pd.DataFrame, fallback_w) -> np.ndarray:
    """The brief's output: remaining minutes = remaining Wh / power. Power is the
    last 30 s average while flying, else a typical flight power (``fallback_w``,
    a number or one value per row)."""
    p = df["p_mean_30s"].to_numpy()
    p = np.where((df["motors_on"].to_numpy() == 1) & (p >= 100), p, fallback_w)
    return pred_wh / p * 60.0


def main() -> None:
    set_seed()
    apply_style()
    cfg = load_config()
    tcfg, target = cfg["model_a_training"], cfg["model_a_training"]["target"]
    for d in (OUT, CACHE, OUT / "figures", MODELS):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path("features") / "model_a.parquet")
    flights = load_processed("flights")
    dev = select(df, DEV)
    series = pd.read_parquet(data_path("features") / "model_a_series.parquet")
    windows = WindowStore(*sequence_windows(series, df))
    print(f"dev rows {len(dev):,}, chains {dev['battery_chain'].nunique()}", flush=True)

    log, best = [], {}
    fp = fingerprint(dev)
    print(f"cache fingerprint {fp}", flush=True)

    def consider(kind, name, cfg_, factory):
        oof, fm = run_cv(kind, cfg_, factory, dev, target, fp)
        row = {"kind": kind, "model": name, "config": json.dumps(cfg_, default=str),
               "cv_mae_mean": float(np.mean(fm)), "cv_mae_std": float(np.std(fm)), "fold_mae": json.dumps(fm)}
        log.append(row)
        print(f"  {name:28s} {row['cv_mae_mean']:.3f} +/- {row['cv_mae_std']:.3f}  {cfg_}", flush=True)
        if name not in best or row["cv_mae_mean"] < best[name]["cv_mae_mean"]:
            best[name] = {**row, "cfg": cfg_, "oof": oof, "factory": factory}

    print("baselines", flush=True)
    for cls in (FixedCapacity, EnergyCounting, VoltageLookup):
        consider("baseline", cls.name, {"baseline": cls.__name__}, lambda cls=cls: cls(flights))

    for kind, n in (("linear", 0), ("random_forest", tcfg["search_trials"]["random_forest"]),
                    ("xgboost", tcfg["search_trials"]["xgboost"])):
        print(kind, flush=True)
        for c in T.sample_configs(kind, n, cfg["seed"]):
            consider(kind, T.FACTORIES[kind]().name, c, lambda c=c, kind=kind: T.FACTORIES[kind](**c))

    grid = tcfg["nn_grid"]
    for kind in ("lstm", "gru"):
        print(kind, flush=True)
        for h in grid["hidden"]:
            for L in grid["layers"]:
                for dr in grid["dropout"]:
                    c = {"hidden": h, "layers": L, "dropout": dr}
                    consider(kind, "LSTM" if kind == "lstm" else "GRU", c,
                             lambda c=c, kind=kind: SequenceModel(kind, windows, **c))

    xgb_cfg = best["XGBoost"]["cfg"]
    print("physics-informed", flush=True)
    consider("physics_xgb", "XGBoost + physics", xgb_cfg,
             lambda: T.PhysicsInformedModel(VoltageLookup(flights), T.xgboost(**xgb_cfg)))

    pd.DataFrame(log).to_csv(OUT / "search_log.csv", index=False)

    # ---------------------------------------------------------- final summary
    oof_tab = dev[["flight", "battery_chain", "time", "phase", target, "remaining_min", "label_source"]].copy()
    fold_of = dev["flight"].astype(str).map(json.loads((ROOT / cfg["split"]["file"]).read_text())["dev_fold"])
    oof_tab["fold"] = fold_of.to_numpy()
    # typical flying power for the minutes conversion, from each fold's *training* folds only
    air_w = np.empty(len(dev))
    for k in np.unique(oof_tab["fold"]):
        trn = (oof_tab["fold"] != k).to_numpy() & (dev["motors_on"] == 1).to_numpy()
        air_w[(oof_tab["fold"] == k).to_numpy()] = float(np.median(dev["p"].to_numpy()[trn]))
    rows, configs = [], {}
    one_row = dev.iloc[[len(dev) // 2]]
    for name, b in best.items():
        oof = b["oof"].reindex(dev.index).to_numpy()
        oof_tab[name] = oof
        m = metrics(dev[target], oof)
        mins = minutes_from_wh(oof, dev, air_w)
        final = b["factory"]().fit(dev, target)                    # refit on all dev data
        if b["kind"] in ("lstm", "gru"):                           # time every model on the CPU, one thread
            final.net.to("cpu")
            final.device = "cpu"
        single_threaded(final)
        torch.set_num_threads(1)
        lat = robust_latency_ms(final.predict, one_row, tcfg["latency_repeats"] if b["kind"] not in ("lstm", "gru") else 200)
        rows.append({"model": name, "kind": b["kind"], "cv_mae": b["cv_mae_mean"], "cv_mae_std": b["cv_mae_std"],
                     "oof_mae": m["mae"], "oof_rmse": m["rmse"], "oof_r2": m["r2"], "oof_bias": m["bias"],
                     "minutes_mae": float(np.mean(np.abs(mins - dev["remaining_min"].to_numpy()))),
                     "latency_ms": lat, "size_mb": size_mb(final)})
        configs[name] = b["cfg"]
        if b["kind"] in ("lstm", "gru"):
            torch.save({"state": final.net.state_dict(), "config": b["cfg"], "kind": b["kind"],
                        "scalers": {k: getattr(final, k) for k in ("med", "s_mu", "s_sd", "q_mu", "q_sd", "y_mu", "y_sd")}},
                       MODELS / f"{b['kind']}.pt")
        else:
            with open(MODELS / f"{name.replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '')}.pkl", "wb") as fh:
                pickle.dump(final, fh)
    table = pd.DataFrame(rows).sort_values("cv_mae").reset_index(drop=True)
    (OUT / "latency_conditions.json").write_text(json.dumps(measurement_conditions(), indent=1))
    # is each model really better than the strongest non-ML baseline?
    ref = "Voltage lookup"
    for i, r in table.iterrows():
        c = paired_comparison(dev[target], oof_tab[r["model"]], oof_tab[ref], dev["battery_chain"])
        fold_diff = np.array(json.loads(best[r["model"]]["fold_mae"])) - np.array(json.loads(best[ref]["fold_mae"]))
        table.loc[i, ["vs_lookup_diff", "vs_lookup_ci_low", "vs_lookup_ci_high", "chains_better_share"]] = (
            c["mae_diff"], c["ci_low"], c["ci_high"], c["share_groups_better"])
        table.loc[i, "folds_better"] = int((fold_diff < 0).sum())
    table.to_csv(OUT / "cv_metrics.csv", index=False)
    oof_tab.to_parquet(OUT / "oof_predictions.parquet")
    (OUT / "best_configs.json").write_text(json.dumps(configs, indent=1, default=str))

    error_analysis(oof_tab, table, target)
    print("\n" + table.round(3).to_string(index=False))


def error_analysis(oof: pd.DataFrame, table: pd.DataFrame, target: str) -> None:
    ml = table[~table["kind"].eq("baseline")].iloc[0]["model"]
    ref = "Voltage lookup"
    bins = pd.cut(oof[target], [-20, 0, 10, 20, 30, 40, 50, 60, 80])
    by_rem = pd.DataFrame({m: (oof[m] - oof[target]).abs().groupby(bins, observed=True).mean() for m in (ml, ref)})
    by_rem["rows"] = bins.value_counts().sort_index()
    by_rem.to_csv(OUT / "error_by_remaining.csv")
    by_phase = pd.DataFrame({m: (oof[m] - oof[target]).abs().groupby(oof["phase"]).mean() for m in (ml, ref)})
    by_phase.to_csv(OUT / "error_by_phase.csv")
    by_chain = (oof[ml] - oof[target]).abs().groupby(oof["battery_chain"]).mean().sort_values(ascending=False)
    by_chain.to_csv(OUT / "error_by_chain.csv", header=["mae"])
    # error when the battery's pre-flight rest voltage is known vs not (3 chains' recordings
    # start with the motors running, so it is unknown there)
    feats = pd.read_parquet(data_path("features") / "model_a.parquet").loc[oof.index]
    known = (feats["v_rest_flight_start"].notna() & feats["v_rest_chain_start"].notna()).to_numpy()
    models = table["model"].tolist()
    pd.DataFrame({"all_rows": [(oof[m] - oof[target]).abs().mean() for m in models],
                  "known_preflight_voltage": [(oof[m] - oof[target])[known].abs().mean() for m in models],
                  "unknown_preflight_voltage": [(oof[m] - oof[target])[~known].abs().mean() for m in models]},
                 index=models).to_csv(OUT / "mae_known_rest_voltage.csv", index_label="model")
    below = (oof[target] <= 0).to_numpy()
    pd.DataFrame({"bias_below_reserve": [(oof[m] - oof[target])[below].mean() for m in models],
                  "mae_below_reserve": [(oof[m] - oof[target])[below].abs().mean() for m in models],
                  "share_overpredict_5wh": [((oof[m] - oof[target]) > 5).mean() for m in models]},
                 index=models).to_csv(OUT / "safety_errors.csv", index_label="model")

    # figure 1: MAE comparison
    t = table.sort_values("cv_mae", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = [INK_MUTED if k == "baseline" else (SERIES[1] if "physics" in k else SERIES[0]) for k in t["kind"]]
    ax.barh(t["model"], t["cv_mae"], xerr=t["cv_mae_std"], color=colors, capsize=3)
    ax.axvspan(1.5, 2.2, color=SERIES[2], alpha=0.15)
    ax.text(1.85, len(t) - 0.45, "label noise floor", ha="center", va="bottom", fontsize=7.5, color=INK_MUTED)
    for y, (v, sd) in enumerate(zip(t["cv_mae"], t["cv_mae_std"])):
        ax.text(v + sd + 0.2, y, f"{v:.2f}", va="center", fontsize=8)
    ax.set_ylim(-0.6, len(t) + 0.2)
    ax.set_xlabel("grouped-CV mean absolute error (Wh), mean ± spread across 5 folds")
    ax.set_title("Model A: energy available, all algorithms (grey = non-ML baselines)")
    save(fig, OUT / "figures" / "mae_comparison.png")

    # figure 2: error by remaining energy
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = [str(i) for i in by_rem.index]
    ax.plot(x, by_rem[ml], "o-", color=SERIES[0], label=ml)
    ax.plot(x, by_rem[ref], "o-", color=INK_MUTED, label=ref)
    ax.set_xlabel("true energy remaining (Wh)"); ax.set_ylabel("mean abs. error (Wh)")
    ax.set_title("Where the errors are"); ax.legend(); ax.tick_params(axis="x", labelsize=7)
    save(fig, OUT / "figures" / "error_by_remaining.png")

    # figure 3: predicted vs true for the best ML model
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    hb = ax.hexbin(oof[target], oof[ml], gridsize=45, cmap="Blues", mincnt=1, bins="log")
    lim = [oof[target].min() - 2, oof[target].max() + 2]
    ax.plot(lim, lim, color=SERIES[1], lw=1)
    ax.set_xlabel("true remaining (Wh)"); ax.set_ylabel("predicted (Wh)")
    ax.set_title(f"{ml}: out-of-fold predictions")
    fig.colorbar(hb, ax=ax, label="rows (log)")
    save(fig, OUT / "figures" / "pred_vs_true.png")


if __name__ == "__main__":
    main()
