"""Rebuild everything from the raw CSV files, in order.

Usage:
    python scripts/run_all.py              development pipeline, steps 01-08 (about 1.5 hours from an empty cache)
    python scripts/run_all.py --from 6     start at step 06 (earlier outputs must exist)
    python scripts/run_all.py --final      also run the final evaluation on the locked test set (step 09)

Every step is one numbered script. A step that fails stops the run. The raw data
is never modified, and step 03 never overwrites the locked split (it only checks it).
Timings are printed and saved to results/run_all_log.json (not a result; it changes every run).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    (1, "01_profile_data.py", "dataset profile"),
    (2, "02_prepare_data.py", "clean, segment, energy, battery chains"),
    (3, "03_make_split.py", "locked split (verified, never overwritten)"),
    (4, "04_build_features.py", "Model A labels and features, Model B targets"),
    (5, "05_model_a.py", "Model A comparison (about 1 hour without a cache)"),
    (6, "06_model_b.py", "Model B components, missions, generalisation"),
    (7, "07_uncertainty.py", "calibrated ranges for both models"),
    (8, "08_decisions.py", "P(success), what-if comparison, fleet simulation"),
]
FINAL = (9, "09_final_test.py", "final evaluation on the locked test set")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", type=int, default=1, help="first step to run (default 1)")
    ap.add_argument("--final", action="store_true", help="also run step 09 on the locked test set")
    args = ap.parse_args()

    steps = [s for s in STEPS if s[0] >= args.start] + ([FINAL] if args.final else [])
    log, t_all = [], time.time()
    for n, script, what in steps:
        print(f"\n===== step {n:02d}: {what} ({script})", flush=True)
        t = time.time()
        cmd = [sys.executable, "-u", str(ROOT / "scripts" / script)] + (["--final"] if n == FINAL[0] else [])
        code = subprocess.run(cmd, cwd=ROOT).returncode
        log.append({"step": n, "script": script, "seconds": round(time.time() - t, 1), "exit_code": code})
        print(f"===== step {n:02d} finished in {log[-1]['seconds']:.0f} s (exit code {code})", flush=True)
        if code != 0:
            break
    out = ROOT / "results" / "run_all_log.json"
    out.write_text(json.dumps({"started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_all)),
                               "total_seconds": round(time.time() - t_all, 1), "steps": log}, indent=1))
    failed = [s for s in log if s["exit_code"] != 0]
    print(f"\n{'FAILED at step ' + str(failed[0]['step']) if failed else 'all steps finished'} "
          f"in {(time.time() - t_all) / 60:.1f} minutes", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
