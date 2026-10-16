# Extensions: evaluation protocol

**Written:** 2026-10-16, before the extensions were run on the unseen batteries. Not to be edited after that run; any later note goes at the end, dated.

## What is evaluated

The two additions requested by the guide (Phase 9):

1. **Energy-aware fleet planner** (`src/endurosense/fleet_planner.py`). Assigns tasks to drones while accounting for battery reserves (P(success)), deadlines, priorities, payload limits, route length (each task is a recorded mission), battery swaps and a charging station.
2. **Where to process a task's data** (`src/endurosense/edge.py`). On board, on an edge server at the depot, or on another drone, chosen by energy and deadline using link quality, server availability and the hover power measured in the dataset.

Both are **simulations on top of the finished models**. The batteries (recorded chains with the battery model's held-out ranges) and the missions (recorded flights with measured energy, duration, payload and distance) are real. Task arrivals, priorities, deadlines, payload limits, the charging station and every computing setting are simulated (`config.yaml`, `extensions`). No model, calibration or earlier result is changed.

## How

- Everything was built and checked on development batteries and missions only (`python scripts/10_extensions.py` → `results/extensions/dev/`).
- The settings in `config.yaml` (`extensions`) and the criteria below are fixed now.
- One run on the unseen batteries and missions, logged in `results/final/test_access_log.json` with a reason:
  `python scripts/10_extensions.py --final --reason "Phase 9 extensions requested by the guide (fleet planner, edge computing); core models and results unchanged"` → `results/extensions/final/`.
- Nothing is adjusted after that run. If a criterion is not met, it is reported as not met.

## Criteria

Policies: S1 first come with minutes left (the brief's rule), S2 first come with the EnduroSense check, S3 planner with best estimates, S4 planner with EnduroSense, S5 planner with the true energies. The "± values" are 95% intervals over simulated days.

| # | Criterion | Met if |
|---|---|---|
| 1 | The planner gets more priority-weighted work done on time than first come, first served (both with the EnduroSense check) | S4's priority-weighted on-time share minus its interval is above S2's plus its interval |
| 2 | The planner stays safe | S4's unsafe missions per 100 flown are at most a tenth of S1's, and fewer than S3's |
| 3 | The energy-aware computing choice is worth having | Its share of jobs on time is at least that of the better fixed rule (always on board, or always offload), and its computing energy per task is no higher than that rule's |

The criteria are computed by `criteria()` in `scripts/10_extensions.py` and written to `criteria.csv`.

## Reported regardless of the criteria

- Every policy's on-time, late and missed tasks, high-priority on-time share, unsafe missions, swaps, minutes without a charged pack and utilisation.
- The computing choice's options taken, energy and deadlines met, and a sweep over link quality and edge-server load.
- That the unseen-battery run uses 8 batteries, so its intervals reflect only the simulated days, not the small number of batteries.

## Development results these criteria were written against

On development batteries (34 batteries, 331 missions, 200 days):

- **Planner:** S4 had 84.9% priority-weighted work on time against 72.4% for S2 (high priority 97% against 64%).
- **Safety:** S4 had 0.23 unsafe missions per 100 flown, against 10.2 for S1, 4.0 for S3 and 0.12 for S2. The planner flies more tasks than S2 and is slightly less conservative than it; criterion 2 does not hide that comparison.
- **Computing:** the energy-aware choice had 95.7% of jobs on time at 2.0 Wh per task, against 90.1% at 2.6 Wh for always on board. "Always offload" refuses many on-site jobs at planning time, because a bad link would mean long hovering.
