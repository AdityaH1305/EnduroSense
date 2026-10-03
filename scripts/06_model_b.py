"""Phase 4: Model B (energy a mission needs) - components, missions, generalisation.

Usage: python scripts/06_model_b.py      (a few minutes; development data only)

1. Component comparison: for each mission part (leg power, leg time, climb,
   descent, hover) compare Physics, Linear Regression, Random Forest, XGBoost
   and Physics + XGBoost with grouped CV; hyperparameters by random search.
2. Mission comparison: rebuild every held-out flight's total energy from its
   plan (leg lengths, speed, payload, altitude, wind) with each family's
   components, plus a "best component each" combination and a naive baseline.
3. Generalisation: leave out every flight at one speed / payload / altitude /
   day, or the longer R5 route, and predict those missions.
4. Plausibility: example missions of a shape absent from the data (deliveries
   that drop the payload half-way).

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
from endurosense.models.model_b import (COMPONENTS, LEG_COMPONENTS, HybridModel, fit_mission_model,
                                        physics_model)
from endurosense.plots import SERIES, apply_style, plt, save

OUT = data_path("results") / "model_b"
MODELS = data_path("models") / "model_b"
FAMILIES = ["Physics", "Linear Regression", "Random Forest", "XGBoost", "Physics + XGBoost"]
KIND = {"Linear Regression": "linear", "Random Forest": "random_forest", "XGBoost": "xgboost"}


def make(family: str, comp: str, cfg: dict | None):
    """Unfitted predictor for one component."""
    feats = COMPONENTS[comp][1]
    if family == "Physics":
        return physics_model(comp)
    if family == "Physics + XGBoost":
        resid = T.xgboost(**cfg)
        resid.features = feats
        return HybridModel(physics_model(comp), resid)
    m = T.FACTORIES[KIND[family]](**cfg)
    m.features = feats
    return m


def component_rows(comp: str, flights_b: pd.DataFrame, legs: pd.DataFrame) -> pd.DataFrame:
    return legs[legs["at_commanded_speed"]] if comp in LEG_COMPONENTS else flights_b


def cv_component(family, comp, cfg, rows, target):
    errs, oof = [], pd.Series(np.nan, index=rows.index)
    for _, tr, va in dev_folds(rows):
        p = make(family, comp, cfg).fit(tr, target).predict(va)
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
                grid = T.sample_configs(KIND[family], n_trials, seed)
            for cfg in grid:
                mae, oof = cv_component(family, comp, cfg, rows, target)
                log.append({"component": comp, "family": family, "config": json.dumps(cfg, default=str), "cv_mae": mae})
                if (family, comp) not in best or mae < best[(family, comp)]["cv_mae"]:
                    best[(family, comp)] = {"cfg": cfg, "cv_mae": mae, "oof": oof}
        print(f"  {comp:12s} " + "  ".join(f"{f}: {best[(f, comp)]['cv_mae']:.3f}" for f in FAMILIES), flush=True)
    return best, pd.DataFrame(log)


def mission_factory(choice: dict):
    """choice: component -> (family, cfg)."""
    return lambda comp: make(choice[comp][0], comp, choice[comp][1])


def mission_cv(choice, flights_b, legs) -> pd.Series:
    """Out-of-fold mission energy for every dev flight."""
    pred = pd.Series(np.nan, index=flights_b["flight"].to_numpy())
    for _, tr, va in dev_folds(flights_b):
        tr_legs, va_legs = legs[legs["flight"].isin(tr["flight"])], legs[legs["flight"].isin(va["flight"])]
        model = fit_mission_model(mission_factory(choice), tr, tr_legs)
        pred.loc[va["flight"].to_numpy()] = model.predict_flights(va, va_legs).to_numpy()
    return pred


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
    best_each = {c: (comp_table.loc[c].idxmin(), best[(comp_table.loc[c].idxmin(), c)]["cfg"]) for c in COMPONENTS}
    choices["Best component each"] = best_each
    y = flights_b.set_index("flight")["total_wh"]
    oof = {}
    for name, choice in choices.items():
        oof[name] = mission_cv(choice, flights_b, legs).reindex(y.index)
    naive = pd.Series(np.nan, index=y.index)                       # average flight energy of the training folds
    for _, tr, va in dev_folds(flights_b):
        naive.loc[va["flight"].to_numpy()] = tr["total_wh"].mean()
    oof["Average flight (naive)"] = naive

    groups = flights_b.set_index("flight").loc[y.index, "battery_chain"]
    rows = []
    for name, p in oof.items():
        m = metrics(y, p)
        c = paired_comparison(y, p, oof["Physics"], groups)
        rows.append({"model": name, "mae_wh": m["mae"], "mape_pct": float(np.mean(np.abs(p - y) / y) * 100),
                     "rmse_wh": m["rmse"], "r2": m["r2"], "bias_wh": m["bias"],
                     "vs_physics_diff": c["mae_diff"], "vs_physics_ci_low": c["ci_low"], "vs_physics_ci_high": c["ci_high"]})
    mission_table = pd.DataFrame(rows).sort_values("mae_wh").reset_index(drop=True)
    mission_table.to_csv(OUT / "mission_cv.csv", index=False)
    pd.DataFrame({"total_wh": y, **{k: v for k, v in oof.items()}}).to_parquet(OUT / "mission_oof.parquet")
    (OUT / "best_component_each.json").write_text(json.dumps({c: v[0] for c, v in best_each.items()}, indent=1))

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
        for name in ["Physics", "Linear Regression", "Random Forest", "XGBoost", "Physics + XGBoost", "Best component each"]:
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
    for name in ["Physics", "Linear Regression", "Random Forest", "XGBoost", "Physics + XGBoost", "Best component each"]:
        final[name] = fit_mission_model(mission_factory(choices[name]), fb, legs)
        with open(MODELS / f"{name.replace(' ', '_').replace('+', 'plus')}.pkl", "wb") as fh:
            pickle.dump(final[name], fh)
    examples = [("Delivery 300 m, 500 g, 8 m/s, 50 m", MissionSpec.delivery(300, 500, 8, 50)),
                ("Delivery 600 m, 500 g, 8 m/s, 50 m", MissionSpec.delivery(600, 500, 8, 50)),
                ("Delivery 300 m, 500 g, 12 m/s, 100 m", MissionSpec.delivery(300, 500, 12, 100)),
                ("Delivery 300 m, 0 g, 4 m/s, 25 m", MissionSpec.delivery(300, 0, 4, 25)),
                ("R1-like loop 140/198/119 m, 250 g, 8 m/s, 50 m", MissionSpec.loop([140, 198, 119], 250, 8, 50))]
    ex = pd.DataFrame({name: {label: m.mission(spec)["total_wh"] for label, spec in examples}
                       for name, m in final.items()})
    ex.index.name = "mission"
    ex.round(2).to_csv(OUT / "example_missions.csv")
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

    best = mission_table[mission_table["model"] != "Average flight (naive)"].iloc[0]["model"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(y, oof[best], s=14, color=SERIES[0], alpha=0.7)
    lim = [y.min() - 1, y.max() + 1]
    ax.plot(lim, lim, color=SERIES[1], lw=1)
    ax.set_xlabel("measured flight energy (Wh)"); ax.set_ylabel("predicted from the plan (Wh)")
    ax.set_title(f"{best}: out-of-fold missions")
    save(fig, OUT / "figures" / "mission_pred_vs_true.png")

    piv = gen_summary.pivot(index="test", columns="model", values="mape_pct")
    models = ["Physics", "Linear Regression", "Random Forest", "XGBoost", "Physics + XGBoost", "Best component each"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.13
    for k, m in enumerate(models):
        ax.bar(np.arange(len(piv)) + (k - 2.5) * w, piv[m], w, color=SERIES[k], label=m)
    ax.set_xticks(range(len(piv)), piv.index)
    ax.set_ylabel("mission energy error (%)")
    ax.set_title("Predicting missions with settings never seen in training")
    ax.legend(fontsize=7, ncol=3)
    save(fig, OUT / "figures" / "generalisation.png")


if __name__ == "__main__":
    main()
