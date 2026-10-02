"""Phase 1: create and lock the train/test split (run once).

Usage: python scripts/03_make_split.py

Never overwrites an existing split: the test set must stay fixed for the whole
project. If the split already exists it is verified and summarised instead.
Writes the split file named in config.yaml (split.file) and results/phase1/split_summary.csv.
"""
import pandas as pd

from endurosense.config import ROOT
from endurosense.data.load import load_processed
from endurosense.data.split import DEV, TEST, split_path, check_consistency, load_split, make_split, save_split


def main() -> None:
    flights = load_processed("flights")
    chains = load_processed("chains")
    if split_path().exists():
        print(f"split already locked at {split_path()}; verifying and summarising only")
    else:
        save_split(make_split(flights, chains))
    split = load_split()
    check_consistency(flights)

    roles = flights["flight"].astype(str).map(split["flight_role"])
    folds = flights["flight"].astype(str).map(split["dev_fold"])
    f = flights.assign(role=roles, fold=folds)
    cruise = f[f["route"].str.startswith("R")]
    near = chains.set_index("battery_chain")["near_reserve"]

    rows = []
    for name, g in [("test", cruise[cruise.role == TEST])] + [
            (f"dev fold {int(k)}", cruise[(cruise.role == DEV) & (cruise.fold == k)])
            for k in sorted(cruise.fold.dropna().unique())]:
        ch = g["battery_chain"].unique()
        rows.append({"part": name, "cruise_flights": len(g), "share": round(len(g) / len(cruise), 3),
                     "chains": len(ch), "near_reserve_chains": int(near.reindex(ch).sum()),
                     "speeds": g["speed"].nunique(), "payloads": g["payload"].nunique(),
                     "altitudes": g["alt_cruise_m"].nunique(), "routes": ",".join(sorted(g["route"].unique()))})
    summary = pd.DataFrame(rows)
    out = ROOT / "results" / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "split_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("sha256:", split["sha256"])


if __name__ == "__main__":
    main()
