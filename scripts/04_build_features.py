"""Phase 2: labels and features for Model A (energy available) and Model B (energy required).

Usage: python scripts/04_build_features.py   (needs the Phase 1 tables and the locked split)

Writes
    data/features/model_a.parquet         1 row per second of flight in labelled chains
    data/features/model_a_series.parquet  2 Hz channels for LSTM/GRU windows
    data/features/model_b_legs.parquet    1 row per cruise leg
    data/features/model_b_flights.parquet 1 row per usable flight (phase targets)
    results/phase2/summary.json, chain_labels_dev.csv, figures/*.png

Anything fitted here (discharge curve, recovery) uses development data only, and
every summary statistic and figure describes development data only: the test
set's labels are written to data/features/ for Phase 7 but never summarised.
"""
import json

import numpy as np
import pandas as pd

from endurosense.config import data_path, set_seed
from endurosense.data.load import load_processed
from endurosense.data.split import DEV, select
from endurosense.features import model_a as A
from endurosense.features import model_b as B
from endurosense.plots import SERIES, apply_style, plt, save

OUT = data_path("results") / "phase2"
FIG = OUT / "figures"


def plot_model_a(chains: pd.DataFrame, lab: pd.DataFrame, curve: A.DischargeCurve, flights, dev_ids) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    dev = flights[flights["flight"].isin(set(dev_ids))]
    for _, g in dev.groupby("battery_chain"):
        e, v, est = A.rest_points(g)
        for k in range(len(e) - 1):
            if not est[k] and not est[k + 1] and np.isfinite(v[k:k + 2]).all() and e[k + 1] - e[k] >= 5:
                a1.scatter((v[k] + v[k + 1]) / 2, (v[k] - v[k + 1]) / (e[k + 1] - e[k]), s=10, color=SERIES[0], alpha=0.6)
    vv = np.linspace(22.3, 25.9, 100)
    a1.plot(vv, curve.slope(vv), color=SERIES[1], lw=2, label="fitted curve (clipped)")
    a1.set_xlabel("rest voltage (V)"); a1.set_ylabel("volts lost per Wh drawn")
    a1.set_title("Discharge curve flattens towards the reserve (dev chains)"); a1.legend()

    m = lab.merge(chains[["battery_chain", "v_rest_first"]], on="battery_chain").dropna(subset=["e_res_wh"])
    for src, col in (("measured", SERIES[0]), ("extrapolated", SERIES[2])):
        d = m[m["label_source"] == src]
        a2.scatter(d["v_rest_first"], d["e_res_wh"], s=18, color=col, label=f"{src} ({len(d)})")
    a2.set_xlabel("rest voltage when the chain started (V)"); a2.set_ylabel("Wh until reserve")
    a2.set_title("Energy available until the reserve (dev chains)"); a2.legend()
    save(fig, FIG / "model_a_labels.png")


def plot_model_b(legs: pd.DataFrame, targets: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    g = legs.groupby("speed")["overhead_s"]
    axes[0].errorbar(g.mean().index, g.mean(), yerr=g.std(), fmt="o-", color=SERIES[0], capsize=3)
    axes[0].set_xlabel("commanded speed (m/s)"); axes[0].set_ylabel("extra time per leg (s)")
    axes[0].set_title("Leg time lost accelerating/braking (dev)")
    for k, (pay, d) in enumerate(legs[legs["payload"] <= 500].groupby("payload")):
        p = d.groupby("speed")["power_w"].mean()
        axes[1].plot(p.index, p.values, "o-", color=SERIES[k], label=f"{pay:.0f} g")
    axes[1].set_xlabel("commanded speed (m/s)"); axes[1].set_ylabel("leg power (W)")
    axes[1].set_title("Cruise power: payload matters, speed barely"); axes[1].legend()
    for k, (col, name) in enumerate((("climb_wh", "climb"), ("descent_wh", "descent"))):
        d = targets.groupby("alt_cruise_m")[col].mean()
        axes[2].plot(d.index, d.values, "o-", color=SERIES[k], label=name)
    axes[2].set_xlabel("cruise altitude (m)"); axes[2].set_ylabel("energy (Wh)")
    axes[2].set_title("Climb and descent energy"); axes[2].legend()
    save(fig, FIG / "model_b_components.png")


def main() -> None:
    set_seed()
    apply_style()
    samples, flights, chains = (load_processed(n) for n in ("samples", "flights", "chains"))
    dev_ids = select(flights, DEV)["flight"]

    curve = A.fit_discharge_curve(flights, dev_ids)
    recovery = A.dev_recovery_v(flights, dev_ids)
    lab = A.chain_labels(flights, curve)
    model_a = A.build_model_a(samples, flights, lab, recovery)
    dev_chains = set(flights.loc[flights["flight"].isin(set(dev_ids)), "battery_chain"])
    series = A.build_series(samples, model_a["battery_chain"].unique())

    legs = B.leg_table(samples, flights)
    targets = B.flight_targets(flights, legs)
    lab_dev = lab[lab["battery_chain"].isin(dev_chains)]
    a_dev = model_a[model_a["battery_chain"].isin(dev_chains)]
    legs_dev, targets_dev = legs[legs["flight"].isin(set(dev_ids))], targets[targets["flight"].isin(set(dev_ids))]

    out = data_path("features")
    out.mkdir(parents=True, exist_ok=True)
    model_a.to_parquet(out / "model_a.parquet", index=False)
    series.to_parquet(out / "model_a_series.parquet", index=False)
    legs.to_parquet(out / "model_b_legs.parquet", index=False)
    targets.to_parquet(out / "model_b_flights.parquet", index=False)

    OUT.mkdir(parents=True, exist_ok=True)
    lab_dev.to_csv(OUT / "chain_labels_dev.csv", index=False)
    summary = {
        "discharge_curve": {"a": curve.a, "b": curve.b, "clip": list(curve.clip),
                            "slope_at_22.6V": float(curve.slope(22.6)), "slope_at_24V": float(curve.slope(24.0))},
        "recovery_v_dev": recovery,
        # counts cover all chains (which chains *can* be labelled); everything describing
        # label values is development-only
        "chain_labels_all": lab["label_source"].value_counts().to_dict(),
        "chain_labels_dev": lab_dev["label_source"].value_counts().to_dict(),
        "extrapolated_wh_dev": lab_dev.loc[lab_dev["label_source"] == "extrapolated", "extrapolated_wh"]
                                      .describe().round(2).to_dict(),
        "model_a_rows": len(model_a), "model_a_chains": int(model_a["battery_chain"].nunique()),
        "model_a_flights": int(model_a["flight"].nunique()),
        "model_a_rows_dev": len(a_dev), "model_a_chains_dev": int(a_dev["battery_chain"].nunique()),
        "remaining_wh_dev": a_dev["remaining_wh"].describe().round(2).to_dict(),
        "remaining_min_dev": a_dev["remaining_min"].describe().round(2).to_dict(),
        "feature_nan_share_dev": a_dev[A.FEATURES].isna().mean().round(4)[lambda x: x > 0].to_dict(),
        "model_b_flights": len(targets), "model_b_legs": len(legs),
        "legs_per_flight": targets["n_legs"].value_counts().sort_index().to_dict(),
        "breakdown_max_abs_error_wh": float((targets[["climb_wh", "legs_wh", "hover_wh", "other_wh", "descent_wh",
                                                      "ground_wh"]].sum(axis=1) - targets["total_wh"]).abs().max()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    plot_model_a(chains, lab_dev, curve, flights, dev_ids)
    plot_model_b(legs_dev, targets_dev)

    print(f"Model A: {len(model_a):,} rows, {summary['model_a_chains']} chains, {summary['model_a_flights']} flights; "
          f"labels (all chains) {summary['chain_labels_all']}")
    print(f"  discharge slope {summary['discharge_curve']['slope_at_22.6V']:.4f} V/Wh at 22.6 V, "
          f"{summary['discharge_curve']['slope_at_24V']:.4f} at 24 V; recovery {recovery:.3f} V")
    print(f"Model B: {len(targets)} flights, {len(legs)} legs; breakdown error {summary['breakdown_max_abs_error_wh']:.1e} Wh")


if __name__ == "__main__":
    main()
