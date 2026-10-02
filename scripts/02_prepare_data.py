"""Phase 1: build the cleaned per-reading, per-flight and per-chain tables.

Usage: python scripts/02_prepare_data.py

Writes
    data/processed/{samples,flights,chains}.parquet
    results/phase1/cleaning_report.json, chain_sensitivity.csv, chain_links.csv
    results/phase1/figures/*.png  (phase examples, chain review pages, discharge curve)
"""
import json

import numpy as np
import pandas as pd

from endurosense.config import ROOT, load_config, set_seed
from endurosense.data.clean import chronological
from endurosense.data.load import load_flights, save_processed
from endurosense.data.prepare import build_tables
from endurosense.plots import INK_MUTED, PHASE_COLORS, SERIES, apply_style, plt, save

OUT = ROOT / "results" / "phase1"
FIG = OUT / "figures"
GRAY_LINE = "#b8b6ae"


def plot_phase_examples(samples: pd.DataFrame, flights: pd.DataFrame, n: int = 12) -> None:
    rng = np.random.default_rng(load_config()["seed"])
    cruise = flights[flights["route"].str.startswith("R")]
    picks = sorted(rng.choice(cruise["flight"], n, replace=False))
    picks[-1] = 250 if 250 not in picks else picks[-1]          # include the truncated flight
    fig, axes = plt.subplots(3, 4, figsize=(11, 6.5), sharey=False)
    for ax, fid in zip(axes.flat, picks):
        f = samples[samples["flight"] == fid]
        for ph, g in f.groupby("phase", observed=True):
            ax.scatter(g["time"], g["alt_rel_m"], s=2, color=PHASE_COLORS[ph], label=ph)
        row = flights.set_index("flight").loc[fid]
        ax.set_title(f"flight {fid}: {row.speed:.0f} m/s, {row.payload:.0f} g", fontsize=9)
    axes.flat[0].legend(markerscale=4, fontsize=7, loc="upper right")
    fig.subplots_adjust(hspace=0.45, wspace=0.25)
    fig.supxlabel("time (s)"); fig.supylabel("height above take-off (m)")
    fig.suptitle("Flight-phase segmentation on sample flights", x=0.01, ha="left")
    save(fig, FIG / "phase_examples.png")


def plot_chain_pages(samples: pd.DataFrame, flights: pd.DataFrame, chains: pd.DataFrame) -> None:
    reserve = load_config()["battery"]["reserve_v"]
    multi = chains[chains["n_flights"] > 1].reset_index(drop=True)
    per_page = 20
    for page in range(int(np.ceil(len(multi) / per_page))):
        sub = multi.iloc[page * per_page:(page + 1) * per_page]
        fig, axes = plt.subplots(4, 5, figsize=(12, 9), sharex=True, sharey=True)
        for ax, (_, c) in zip(axes.flat, sub.iterrows()):
            s = samples[samples["battery_chain"] == c["battery_chain"]]
            ax.plot(s["cum_chain_energy_wh"], s["battery_voltage"], color=GRAY_LINE, lw=0.5)
            fl = chronological(flights[flights["battery_chain"] == c["battery_chain"]])
            e0 = np.r_[0, fl["energy_wh"].cumsum().to_numpy()[:-1]]
            ax.scatter(e0, fl["v_rest_start"], s=14, color=SERIES[0], zorder=3)
            ax.scatter(e0[-1] + fl["energy_wh"].iloc[-1], fl["v_rest_after"].iloc[-1], s=14,
                       facecolor="white", edgecolor=SERIES[0], zorder=3)
            ax.axhline(reserve, color=SERIES[1], lw=0.8, ls="--")
            flag = " *" if c.get("uncertain_links", 0) else ""
            ax.set_title(f"chain {c['battery_chain']}: flights {', '.join(map(str, c['flights']))}{flag}",
                         fontsize=7.5)
        for ax in list(axes.flat)[len(sub):]:
            ax.axis("off")
        for ax in list(axes.flat)[:len(sub)]:
            ax.tick_params(labelbottom=True, labelleft=True)
        fig.subplots_adjust(hspace=0.45)
        fig.supxlabel("energy drawn since chain start (Wh)")
        fig.supylabel("battery voltage (V): grey = under load, dots = at rest")
        fig.suptitle(f"Battery chain review, page {page + 1}  (dashed = reserve {reserve} V; "
                     "open dot = estimated rest after last flight; * = uncertain link)", x=0.01, ha="left")
        save(fig, FIG / f"chains_page{page + 1}.png")


def plot_discharge_curve(flights: pd.DataFrame) -> None:
    reserve = load_config()["battery"]["reserve_v"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for _, fl in chronological(flights).groupby("battery_chain"):
        if len(fl) < 2:
            continue
        e = np.r_[0, fl["energy_wh"].cumsum().to_numpy()]
        v = np.r_[fl["v_rest_start"].to_numpy(), fl["v_rest_after"].iloc[-1]]
        ax.plot(e, v, color=SERIES[0], alpha=0.35, lw=1.1, marker="o", ms=2.5)
    ax.axhline(reserve, color=SERIES[1], lw=1.4, ls="--")
    ax.text(1, reserve - 0.08, f"reserve ({reserve} V)", va="top", color=INK_MUTED, fontsize=8)
    ax.set_xlabel("energy drawn since chain start (Wh)")
    ax.set_ylabel("rest voltage (V)")
    ax.set_title("Multi-flight battery chains, motors-off rest voltages")
    save(fig, FIG / "discharge_curve.png")



def main() -> None:
    set_seed()
    apply_style()
    samples, flights, chains, report = build_tables(load_flights())
    for name, df in (("samples", samples), ("flights", flights), ("chains", chains)):
        save_processed(name, df)

    OUT.mkdir(parents=True, exist_ok=True)
    report["sensitivity"].to_csv(OUT / "chain_sensitivity.csv", index=False)
    links = chronological(flights)[
        ["flight", "date", "local_time", "battery_chain", "v_rest_start", "v_rest_end", "link_jump_v",
         "link_confident", "v_rest_start_estimated", "v_rest_end_estimated"]]
    links.to_csv(OUT / "chain_links.csv", index=False)
    serialisable = {k: v for k, v in report.items() if k != "sensitivity"}
    (OUT / "cleaning_report.json").write_text(json.dumps(serialisable, indent=1, default=float))

    plot_phase_examples(samples, flights)
    plot_chain_pages(samples, flights, chains)
    plot_discharge_curve(flights)

    cruise = flights[flights["route"].str.startswith("R")]
    share = cruise[[f"{p}_energy_wh" for p in ("climb", "cruise", "descent", "hover", "ground")]].sum()
    print(f"readings {len(samples):,} | flights {len(flights)} | chains {len(chains)} "
          f"({int((chains.n_flights > 1).sum())} multi-flight, {int(chains.near_reserve.sum())} near reserve)")
    print("phase order OK:", int(cruise["phase_order_ok"].sum()), "/", len(cruise),
          "| problems:", report["phase_order_problems"])
    print("energy share by phase:", (share / share.sum()).round(3).to_dict())
    print("glitches fixed: voltage", report["voltage_glitches_fixed"], "current", report["current_glitches_fixed"])
    print("uncertain links:", report["uncertain_links"])
    print(report["sensitivity"].to_string(index=False))


if __name__ == "__main__":
    main()
