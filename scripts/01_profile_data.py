"""Profile the CMU package-delivery drone dataset (Rodrigues et al., 2021).

Answers the questions that decide whether the dataset can support EnduroSense:
  1. Is the payload x speed x altitude grid balanced?
  2. What wind conditions are covered?
  3. How much energy does a flight use, and how much of the battery is that?
  4. Are batteries reused across consecutive flights (longer discharge curves)?

Usage: python scripts/01_profile_data.py
Writes per-flight summary to data/processed/flight_summary.csv.
"""
import pandas as pd

from endurosense.config import data_path, load_config
from endurosense.data.clean import chronological
from endurosense.data.load import load_flights, load_parameters
from endurosense.data.profile import build_flight_summary

PACK_WH = load_config()["battery"]["pack_wh"]


def main() -> None:
    out = data_path("processed")
    out.mkdir(parents=True, exist_ok=True)
    flights = load_flights()
    summary = build_flight_summary(flights, load_parameters())
    summary.to_csv(out / "flight_summary.csv", index=False)

    pd.set_option("display.width", 160, "display.max_columns", 30)
    print(f"rows: {len(flights):,}   flights: {flights['flight'].nunique()}")
    print(f"median sample interval: {summary['median_dt_s'].median():.3f} s\n")

    print("== Flights by route ==")
    print(summary["route"].value_counts().sort_index().to_string(), "\n")

    cruise = summary[summary["route"].str.startswith("R")]
    print(f"== Design grid ({len(cruise)} cruise flights): count per speed x payload, by altitude ==")
    for alt, g in cruise.groupby("altitude"):
        print(f"-- altitude {alt} m --")
        print(pd.crosstab(g["speed"], g["payload"]).to_string(), "\n")
    combos = cruise.groupby(["speed", "payload", "altitude"]).size()
    print(f"combinations filled: {len(combos)} / "
          f"{cruise['speed'].nunique() * cruise['payload'].nunique() * cruise['altitude'].nunique()}"
          f"   flights per combo min/median/max: {combos.min()}/{combos.median():.0f}/{combos.max()}\n")

    print("== Per-flight stats (cruise flights) ==")
    cols = ["duration_s", "energy_wh", "pct_pack", "v_start", "v_end", "i_mean_air",
            "airspeed_mean", "ground_speed_max", "alt_max_rel"]
    print(cruise[cols].describe().round(2).to_string(), "\n")

    print("== Energy (Wh) by speed x payload, mean over cruise flights ==")
    print(cruise.pivot_table(index="speed", columns="payload", values="energy_wh").round(2).to_string(), "\n")

    print("== Ambient wind per cruise flight (m/s) ==")
    print(pd.cut(cruise["ambient_wind"], [0, 2, 3, 4, 5, 6, 8, 12]).value_counts(sort=False).to_string(), "\n")

    chains = chronological(summary).groupby("battery_chain").agg(
        n=("flight", "size"), v0=("v_start", "first"), v1=("v_end", "last"), energy_wh=("energy_wh", "sum"))
    print("== Battery chains (consecutive flights on one battery) ==")
    print(f"chains: {len(chains)}   flights per chain: {chains['n'].value_counts().sort_index().to_dict()}")
    deep = chains[(chains["v0"] >= 25.0) & (chains["v1"] <= 22.6)]
    print(f"near-full to near-reserve chains (>=25.0 V -> <=22.6 V): {len(deep)}, "
          f"energy used {deep['energy_wh'].min():.0f}-{deep['energy_wh'].max():.0f} Wh "
          f"of ~{PACK_WH:.0f} Wh pack\n")

    print(f"start-voltage range across all flights: {summary['v_start'].min():.2f} - "
          f"{summary['v_start'].max():.2f} V   (6S LiPo: 25.2 V full, ~21 V empty)")
    print(f"lowest voltage seen in any flight: {summary['v_min'].min():.2f} V")
    print(f"days of flying: {summary['date'].nunique()}")


if __name__ == "__main__":
    main()
