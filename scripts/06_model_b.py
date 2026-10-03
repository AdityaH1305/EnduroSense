"""Phase 4: Model B (energy a mission needs) - components, missions, generalisation.

Usage: python scripts/06_model_b.py      (a few minutes; development data only)

1. Component comparison: for each mission part (leg power, leg time, climb,
   descent, hover) compare Physics, Linear Regression, Random Forest, XGBoost
   and Physics + XGBoost with grouped CV; hyperparameters by random search.
2. Mission comparison: rebuild every held-out flight's total energy from its
   plan (leg lengths, speed, payload, altitude, wind) with each family's
   components, two combinations ("best component each" and the main model,
   "physics-first"), and a naive baseline. Also reports each model's systematic
   bias per speed and payload.
3. Generalisation: leave out every flight at one speed / payload / altitude /
   day, or the longer R5 route, and predict those missions.
4. Plausibility: example missions of a shape absent from the data (deliveries
   that drop the payload half-way).

Also checks how robust the headline number is: with the algorithm for each part
chosen inside every training fold (no selection optimism), with no wind input,
and with only a typical wind known at planning time.

"The plan" of a recorded flight uses its measured leg lengths: the dataset does
not give the programmed waypoints, and GPS leg lengths are the closest record.

Writes results/model_b/*.csv, figures/*.png and models/model_b/*.pkl.
"""
import json
import pickle

import numpy as np
import pandas as pd

from endurosense.config import data_path, load_config, set_seed
from endurosense.data.load import load_processed
from endurosense.data.split import DEV, dev_folds, select
from endurosense.evaluate import metrics, paired_comparison
from endurosense.mission import MissionSpec
from endurosense.models import tabular as T
from endurosense.models.model_b import (COMPONENTS, FAMILIES, LEG_COMPONENTS, MAIN, choose_family,
                                        fit_mission_model, make_component, mission_factory)
from endurosense.plots import SERIES, apply_style, plt, save

OUT = data_path("results") / "model_b"
MODELS = data_path("models") / "model_b"
COMPARED = FAMILIES + ["Best component each", MAIN]


def component_rows(comp: str, flights_b: pd.DataFrame, legs: pd.DataFrame) -> pd.DataFrame:
    return legs[legs["at_commanded_speed"]] if comp in LEG_COMPONENTS else flights_b


def cv_component(family, comp, cfg, rows, target):
    errs, oof = [], pd.Series(np.nan, index=rows.index)
    for _, tr, va in dev_folds(rows):
        p = make_component(family, comp, cfg).fit(tr, target).predict(va)
        oof.loc[va.index] = p
        errs.append(float(np.mean(np.abs(p - va[target].to_numpy()))))
    return float(np.mean(errs)), oof


def search_components(flights_b, legs, n_trials: int, seed: int):
    """Best configuration per (family, component) by grouped-CV MAE."""
    best, log = {}, []
    for comp, (target, _) in COMPONENTS.items():
        rows = component_rows(comp, flights_b, legs)
        for family in FAMILIES:
            if family == "Physics":
                grid = [{}]
            elif family == "Physics + XGBoost":
                grid = T.sample_configs("xgboost", n_trials, seed + 1)
            else:
                grid = T.sample_configs({"Linear Regression": "linear", "Random Forest": "random_forest", "XGBoost": "xgboost"}[family], n_trials, seed)
            for cfg in grid:
                mae, oof = cv_component(family, comp, cfg, rows, target)
                log.append({"component": comp, "family": family, "config": json.dumps(cfg, default=str), "cv_mae": mae})
                if (family, comp) not in best or mae < best[(family, comp)]["cv_mae"]:
                    best[(family, comp)] = {"cfg": cfg, "cv_mae": mae, "oof": oof}
        print(f"  {comp:12s} " + "  ".join(f"{f}: {best[(f, comp)]['cv_mae']:.3f}" for f in FAMILIES), flush=True)
    return best, pd.DataFrame(log)


def mission_cv(choice, flights_b, legs, drop: tuple = (), typical_wind: bool = False) -> pd.Series:
    """Out-of-fold mission energy for every dev flight.

    ``drop`` withholds inputs from every component. ``typical_wind`` trains with
    the measured wind but predicts with the training folds' average wind, i.e.
    as if only a typical wind were known when planning.
    """
    pred = pd.Series(np.nan, index=flights_b["flight"].to_numpy())
    for _, tr, va in dev_folds(flights_b):
        tr_legs, va_legs = legs[legs["flight"].isin(tr["flight"])], legs[legs["flight"].isin(va["flight"])]
        model = fit_mission_model(mission_factory(choice, drop), tr, tr_legs)
        if typical_wind:
            w = float(tr["ambient_wind"].mean())
            va, va_legs = va.assign(ambient_wind=w), va_legs.assign(ambient_wind=w)
        pred.loc[va["flight"].to_numpy()] = model.predict_flights(va, va_legs).to_numpy()
    return pred


def nested_selection_cv(best, flights_b, legs, rule: str) -> tuple[pd.Series, pd.DataFrame]:
    """A combination with the algorithm for each part chosen *inside* every training
    fold (by CV over the other four folds), so the fold being scored never
    influences the choice. Measures how optimistic the plain result is."""
    folds = list(dev_folds(flights_b))
    pred, picks = pd.Series(np.nan, index=flights_b["flight"].to_numpy()), []
    for k, tr, va in folds:
        tr_legs = legs[legs["flight"].isin(tr["flight"])]
        sel = {}
        for comp, (target, _) in COMPONENTS.items():
            rows = component_rows(comp, tr, tr_legs)
            score = {}
            for fam in FAMILIES:
                errs = []
                for j, _, inner_va in folds:
                    if j == k:
                        continue
                    r_tr, r_va = rows[~rows["flight"].isin(inner_va["flight"])], rows[rows["flight"].isin(inner_va["flight"])]
                    p = make_component(fam, comp, best[(fam, comp)]["cfg"]).fit(r_tr, target).predict(r_va)
                    errs.append(np.mean(np.abs(p - r_va[target].to_numpy())))
                score[fam] = float(np.mean(errs))
            fam = choose_family(score, rule)
            sel[comp] = (fam, best[(fam, comp)]["cfg"])
        picks.append({"fold": k, **{c: v[0] for c, v in sel.items()}})
        model = fit_mission_model(mission_factory(sel), tr, tr_legs)
        pred.loc[va["flight"].to_numpy()] = model.predict_flights(va, legs[legs["flight"].isin(va["flight"])]).to_numpy()
    return pred, pd.DataFrame(picks)


def holdout(choice, flights_b, legs, test_mask) -> tuple[np.ndarray, np.ndarray]:
    tr, te = flights_b[~test_mask], flights_b[test_mask]
    model = fit_mission_model(mission_factory(choice), tr, legs[legs["flight"].isin(tr["flight"])])
    return te["total_wh"].to_numpy(), model.predict_flights(te, legs[legs["flight"].isin(te["flight"])]).to_numpy()


def main() -> None:
    set_seed()
    apply_style()
    cfg = load_config()
    for d in (OUT, OUT / "figures", MODELS):
        d.mkdir(parents=True, exist_ok=True)

    legs = select(pd.read_parquet(data_path("features") / "model_b_legs.parquet"), DEV)
    flights_b = select(pd.read_parquet(data_path("features") / "model_b_flights.parquet"), DEV)
    meta = load_processed("flights").set_index("flight")[["date", "route"]]
    flights_b = flights_b.join(meta, on="flight")
    print(f"dev: {len(flights_b)} flights, {len(legs)} legs", flush=True)

    # 1. components ----------------------------------------------------------
    print("component search (grouped CV MAE: W for leg power, s for leg overhead, Wh otherwise)", flush=True)
    best, log = search_components(flights_b, legs, n_trials=10, seed=cfg["seed"])
    log.to_csv(OUT / "component_search_log.csv", index=False)
    comp_table = pd.DataFrame({f: {c: best[(f, c)]["cv_mae"] for c in COMPONENTS} for f in FAMILIES})
    comp_table.index.name = "component"
    comp_table.to_csv(OUT / "component_cv_mae.csv")

    # 2. missions --------------------------------------------------------------
    choices = {f: {c: (f, best[(f, c)]["cfg"]) for c in COMPONENTS} for f in FAMILIES}
    for name, rule in (("Best component each", "best"), (MAIN, "physics_first")):
        fam = {c: choose_family(comp_table.loc[c].to_dict(), rule) for c in COMPONENTS}
        choices[name] = {c: (fam[c], best[(fam[c], c)]["cfg"]) for c in COMPONENTS}
    y = flights_b.set_index("flight")["total_wh"]
    oof = {}
    for name, choice in choices.items():
        oof[name] = mission_cv(choice, flights_b, legs).reindex(y.index)
    naive = pd.Series(np.nan, index=y.index)                       # average flight energy of the training folds
    for _, tr, va in dev_folds(flights_b):
        naive.loc[va["flight"].to_numpy()] = tr["total_wh"].mean()
    oof["Average flight (naive)"] = naive

    fmeta = flights_b.set_index("flight").loc[y.index]
    groups = fmeta["battery_chain"]
    # systematic error per setting: a model can be accurate on average yet biased for,
    # say, the fastest flights; a negative bias means it under-predicts the energy needed
    bias_rows = []
    for name, p in oof.items():
        for col in ("speed", "payload"):
            g = (p - y).groupby(fmeta[col]).agg(["mean", "size"])
            for val, r in g[g["size"] >= 3].iterrows():
                bias_rows.append({"model": name, "setting": col, "value": val, "flights": int(r["size"]), "bias_wh": r["mean"]})
    bias = pd.DataFrame(bias_rows)
    bias.to_csv(OUT / "mission_bias.csv", index=False)
    worst = bias.assign(a=bias["bias_wh"].abs()).sort_values("a").groupby("model").tail(1).set_index("model")
    rows = []
    for name, p in oof.items():
        m = metrics(y, p)
        c = paired_comparison(y, p, oof["Physics"], groups)
        rows.append({"model": name, "mae_wh": m["mae"], "mape_pct": float(np.mean(np.abs(p - y) / y) * 100),
                     "rmse_wh": m["rmse"], "r2": m["r2"], "bias_wh": m["bias"],
                     "worst_group_bias_wh": float(worst.loc[name, "bias_wh"]),
                     "worst_group": f"{worst.loc[name, 'setting']} {worst.loc[name, 'value']:g}",
                     "vs_physics_diff": c["mae_diff"], "vs_physics_ci_low": c["ci_low"], "vs_physics_ci_high": c["ci_high"]})
    mission_table = pd.DataFrame(rows).sort_values("mae_wh").reset_index(drop=True)
    mission_table.to_csv(OUT / "mission_cv.csv", index=False)
    pd.DataFrame({"total_wh": y, **{k: v for k, v in oof.items()}}).to_parquet(OUT / "mission_oof.parquet")
    (OUT / "combinations.json").write_text(json.dumps(
        {name: {c: v[0] for c, v in choices[name].items()} for name in ("Best component each", MAIN)}, indent=1))

    # 2b. robustness of the headline result ---------------------------------------
    mae = lambda p: float(np.mean(np.abs(p.reindex(y.index) - y)))
    mape = lambda p: float(np.mean(np.abs(p.reindex(y.index) - y) / y) * 100)
    nested, picks = nested_selection_cv(best, flights_b, legs, "physics_first")
    nested_best, _ = nested_selection_cv(best, flights_b, legs, "best")
    checks = {f"{MAIN} (as reported)": oof[MAIN],
              "... algorithm chosen inside each training fold (nested)": nested,
              "... no wind input at all": mission_cv(choices[MAIN], flights_b, legs, drop=("ambient_wind",)),
              "... only a typical wind known when planning": mission_cv(choices[MAIN], flights_b, legs, typical_wind=True),
              "Best component each (as reported)": oof["Best component each"],
              "... nested": nested_best}
    robustness = pd.DataFrame([{"variant": k, "mae_wh": mae(v), "mape_pct": mape(v)} for k, v in checks.items()])
    robustness.to_csv(OUT / "robustness.csv", index=False)
    picks.to_csv(OUT / "nested_selection_choices.csv", index=False)
    print(robustness.round(3).to_string(index=False), flush=True)

    # 3. generalisation --------------------------------------------------------
    tests = []
    fb = flights_b.reset_index(drop=True)
    for col, label in (("speed", "speed"), ("payload", "payload"), ("alt_cruise_m", "altitude")):
        for val in sorted(fb[col].unique()):
            mask = (fb[col] == val).to_numpy()
            if mask.sum() >= 3:
                tests.append((f"unseen {label}", f"{val:g}", mask))
    for day in sorted(fb["date"].unique()):
        tests.append(("unseen day", day, (fb["date"] == day).to_numpy()))
    # R5 (~505 m) is the only longer route in development, but its 4 flights are also the only
    # dev flights from the first flying day (2019-04-07), so route length and day are confounded
    tests.append(("R5 route = day 1 (4 flights)", "R5 (~505 m)", (fb["route"] == "R5").to_numpy()))
    gen = []
    for test, value, mask in tests:
        for name in COMPARED:
            yt, pt = holdout(choices[name], fb, legs, mask)
            gen.append({"test": test, "held_out": value, "model": name, "flights": int(mask.sum()),
                        "mae_wh": float(np.mean(np.abs(pt - yt))), "mape_pct": float(np.mean(np.abs(pt - yt) / yt) * 100),
                        "bias_wh": float(np.mean(pt - yt))})
    gen = pd.DataFrame(gen)
    gen.to_csv(OUT / "generalisation_detail.csv", index=False)
    gen_summary = (gen.assign(w_err=gen["mape_pct"] * gen["flights"])
                   .groupby(["test", "model"]).agg(flights=("flights", "sum"), w_err=("w_err", "sum"))
                   .assign(mape_pct=lambda d: d["w_err"] / d["flights"]).drop(columns="w_err").reset_index())
    gen_summary.to_csv(OUT / "generalisation_summary.csv", index=False)

    # 4. final models + plausibility --------------------------------------------
    final = {}
    for name in COMPARED:
        final[name] = fit_mission_model(mission_factory(choices[name]), fb, legs)
        with open(MODELS / f"{name.replace(' ', '_').replace('+', 'plus')}.pkl", "wb") as fh:
            pickle.dump(final[name], fh)
    with open(MODELS / "model_b.pkl", "wb") as fh:                 # the main Model B for later phases
        pickle.dump(final[MAIN], fh)
    examples = [("Delivery 300 m, 500 g, 8 m/s, 50 m", MissionSpec.delivery(300, 500, 8, 50)),
                ("Delivery 600 m, 500 g, 8 m/s, 50 m", MissionSpec.delivery(600, 500, 8, 50)),
                ("Delivery 300 m, 500 g, 12 m/s, 100 m", MissionSpec.delivery(300, 500, 12, 100)),
                ("Delivery 300 m, 0 g, 4 m/s, 25 m", MissionSpec.delivery(300, 0, 4, 25)),
                ("R1-like loop 140/198/119 m, 250 g, 8 m/s, 50 m", MissionSpec.loop([140, 198, 119], 250, 8, 50)),
                ("Out of range: delivery 300 m, 1000 g, 15 m/s, 150 m", MissionSpec.delivery(300, 1000, 15, 150))]
    ex = pd.DataFrame({name: {label: m.mission(spec)["total_wh"] for label, spec in examples}
                       for name, m in final.items()})
    main_model = final[MAIN]
    ex["outside_tested_range"] = ["; ".join(main_model.extrapolation_warnings(spec)) for _, spec in examples]
    ex.index.name = "mission"
    ex.round(2).to_csv(OUT / "example_missions.csv")
    (OUT / "tested_ranges.json").write_text(json.dumps(main_model.ranges, indent=1))
    physics_params = {"leg_power": dict(zip(["airframe_mass_kg", "P0_w", "k_ind", "k_par"],
                                            map(float, final["Physics"].predictors["leg_power"].theta))),
                      "leg_overhead": {"inv_accel_s2_per_m": float(final["Physics"].predictors["leg_overhead"].inv_a),
                                       "offset_s": float(final["Physics"].predictors["leg_overhead"].c)}}
    (OUT / "physics_parameters.json").write_text(json.dumps(physics_params, indent=1))

    figures(comp_table, mission_table, oof, y, gen_summary)
    print("\ncomponents (CV MAE):\n" + comp_table.round(3).to_string())
    print("\nmissions:\n" + mission_table.round(3).to_string(index=False))
    print("\ngeneralisation (MAPE %):\n" + gen_summary.pivot(index="model", columns="test", values="mape_pct").round(2).to_string())
    print("\nexample missions (Wh):\n" + ex.round(2).to_string())


def figures(comp_table, mission_table, oof, y, gen_summary) -> None:
    units = {"leg_power": "W", "leg_overhead": "s", "climb": "Wh", "descent": "Wh", "hover": "Wh"}
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
    for ax, comp in zip(axes, comp_table.index):
        v = comp_table.loc[comp]
        ax.barh(range(len(v)), v.values, color=[SERIES[1] if f == "Physics" else SERIES[0] for f in v.index])
        ax.set_yticks(range(len(v)), v.index if comp == comp_table.index[0] else [""] * len(v))
        ax.set_title(comp.replace("_", " "), fontsize=9)
        ax.set_xlabel(f"CV MAE ({units[comp]})")
        ax.invert_yaxis()
    fig.suptitle("Model B components: grouped-CV error per algorithm (orange = physics)", x=0.01, ha="left")
    save(fig, OUT / "figures" / "component_comparison.png")

    best = MAIN
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(y, oof[best], s=14, color=SERIES[0], alpha=0.7)
    lim = [y.min() - 1, y.max() + 1]
    ax.plot(lim, lim, color=SERIES[1], lw=1)
    ax.set_xlabel("measured flight energy (Wh)"); ax.set_ylabel("predicted from the plan (Wh)")
    ax.set_title(f"{best}: out-of-fold missions")
    save(fig, OUT / "figures" / "mission_pred_vs_true.png")

    piv = gen_summary.pivot(index="test", columns="model", values="mape_pct")
    models = COMPARED
    fig, ax = plt.subplots(figsize=(10, 3.6))
    w = 0.115
    for k, m in enumerate(models):
        ax.bar(np.arange(len(piv)) + (k - 3) * w, piv[m], w, color=SERIES[k], label=m)
    ax.set_xticks(range(len(piv)), piv.index)
    ax.set_ylabel("mission energy error (%)")
    ax.set_title("Predicting missions with settings never seen in training")
    ax.legend(fontsize=7, ncol=4)
    save(fig, OUT / "figures" / "generalisation.png")


if __name__ == "__main__":
    main()
