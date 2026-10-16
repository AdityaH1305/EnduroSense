# Phase 9: The guide's extensions

**Status:** complete (2026-10-16).
**Outputs:** `results/extensions/dev/` (development) and `results/extensions/final/` (unseen batteries), each with `planner.csv`, `computing.csv`, `computing_sensitivity.csv`, `criteria.csv`, `summary.json` and `figures/extensions.png`.
**Run:** `python scripts/10_extensions.py` (development, about 20 minutes) and `python scripts/10_extensions.py --final` (unseen batteries; logged). Criteria: `docs/extensions_protocol.md`.

## What was asked

1. **Multi-drone energy-aware task scheduling:** battery reserves, task deadlines, payload, travel distance, charging and mission priorities, together.
2. **Energy-aware communication and edge computing:** communication energy, link quality, offloading, edge-server availability and mobility. A drone decides whether to process a task locally, offload it, or transfer it to another platform.

## What is real and what is simulated

| Real | Simulated |
|---|---|
| Batteries: recorded chains, with the battery model's held-out ranges (34 development batteries; 8 unseen) | Task arrivals, priorities and deadlines |
| Missions: recorded flights and same-battery sortie pairs, with measured energy, flight time, payload and distance (331 development; 85 unseen) | Each drone's payload limit, the charging station and spare packs |
| Model B's ranges for each mission, and P(success) | Every radio, computer and edge-server setting (typical published values) |
| Hover power: 539 W, the median on development flights | |

The dataset has none of the simulated parts. Every setting is in `config.yaml` under `extensions`. No model, calibration or earlier result changed: all 59 files in `results/final/`, `results/decisions/` and `models/` are byte-identical to before this phase, apart from one new entry in the test access log.

## Extension 1: energy-aware fleet planner (`src/endurosense/fleet_planner.py`)

- **The day:** 8 hours in one-minute steps. 100 tasks; each is a recorded mission with a priority (high 20% / normal 50% / low 30%; weights 3 / 2 / 1; due within 30 / 90 / 240 minutes).
- **The fleet:** 4 drones with payload limits of 750, 500, 500 and 250 g. 3 packs per drone in circulation, swapped at the depot in 2 minutes, recharged on 2 chargers at 100 W. There is more work than the chargers can supply (about 22 tasks a day are missed by every policy, including the oracle), so the policies must choose.
- **The planner:** every minute, all waiting tasks are matched to all free drones with `scipy.optimize.linear_sum_assignment`. A pair is allowed only if the payload fits, the task can finish before its deadline, and P(success) ≥ τ. Priority comes first, then the closest deadline, then the best-fitting battery. A free drone that is too low for a task it could carry swaps its pack early, if a charged one is ready. Tasks that can no longer be on time are not started.
- **Compared policies:**
  - S1: first come, minutes left (the brief's rule).
  - S2: first come with the EnduroSense check (the Phase 6 fleet rule).
  - S3: the planner with best estimates.
  - S4: the planner with EnduroSense.
  - S5: the planner with the true energies (oracle).

## Extension 2: where to process the data (`src/endurosense/edge.py`)

- **The job:** each task carries a computing job. Its data size is 20 to 150 MB, it takes 10 to 60 s on board, and its deadline is 30 to 120 s after reaching the site. Half the jobs need the result on site, so the drone hovers while it waits.
- **The options:**
  - **On board:** a 15 W computer.
  - **Edge server at the depot:** 8 times faster, but busy 30% of the time (30 s average wait).
  - **Another drone waiting at the depot:** it computes the job on its own battery.
- **The link:** 10 MHz of bandwidth, 20 dB signal-to-noise at 100 m, path-loss exponent 2.5 and Rayleigh fading. It is blocked 10% of the time, cut to a tenth of its rate. The radio draws 1 W. Mobility enters through the distance to the depot: the delivery site for on-site jobs, half that on the way back.
- **Energy-aware rule:** measure the link and the server's queue on site, then take the option with the least energy for this drone that meets the job's deadline. Before take-off, P(success) includes the bad-case computing energy (fade 0.3, blocked, a 60 s server wait).
- **Why it matters:** hovering, not the radio, is the cost. 30 s of hovering is 4.5 Wh; 30 s of computing is 0.13 Wh.

## Results on the unseen batteries (8 batteries, 85 missions)

| Policy | Weighted tasks on time | High priority on time | Unsafe per 100 flown | Swaps per day |
|---|---|---|---|---|
| S1 first come, minutes left | 67.2% | 53% | 11.10 | 31.7 |
| S2 first come, EnduroSense check | 71.3% | 62% | 0.62 | 34.4 |
| S3 planner, best estimates | 84.3% | 96% | 4.46 | 33.8 |
| **S4 planner, EnduroSense** | **83.8%** | **95%** | **0.61** | 37.2 |
| S5 planner, oracle | 84.3% | 96% | 0.00 | 34.0 |

| Computing rule | Energy per task | Jobs on time | On board / edge / other drone | Weighted tasks on time |
|---|---|---|---|---|
| Always on board | 2.60 Wh | 90.2% | 100 / 0 / 0% | 78.3% |
| Always offload | 0.95 Wh | 81.5% | 0 / 100 / 0% | 56.7% |
| **Energy-aware** | **2.05 Wh** | **95.6%** | 37 / 62 / 1% | 79.4% |

- **Criteria** (fixed beforehand): **all three met.**
  - The planner beats first come: 83.8% ± 0.5 against 71.3% ± 0.7.
  - The planner stays safe: 0.61 unsafe per 100, against 11.10 for minutes left and 4.46 for best estimates.
  - The energy-aware choice meets more job deadlines (95.6% against 90.2%) for less energy (2.05 against 2.60 Wh).
- **Planning gives the speed, the probability gives the safety.** The planner with best estimates is as quick as with EnduroSense but has seven times the unsafe missions. The planner with EnduroSense is as safe as the first-come queue with the same check, and gets far more urgent work done.
- **"Always offload" looks cheapest per flown task only because the planner refuses many on-site jobs before take-off,** since in a bad case the drone would hover for minutes.
- **Sensitivity:** with a weak link (10 dB) the energy-aware rule processes 61 to 66% of jobs on board; with a strong link (30 dB) it offloads 69 to 83%. Hand-overs rise from under 1% to about 3% as the server gets busier.
- **The development results were close:** S4 84.9% against S2 72.4%, with unsafe per 100 at 0.23, 0.12, 3.96 and 10.19 for S4, S2, S3 and S1. Energy-aware: 95.7% at 2.02 Wh, against 90.1% at 2.58 Wh on board.

## Limits

- **Simulated schedules and radio settings.** The numbers depend on them; the sweep shows the direction is stable, not the exact values.
- **Eight unseen batteries.** The ± values cover only the simulated days.
- **The planner is greedy in time.** It is optimal at each minute, but does not hold drones back for tasks that have not arrived.
- **Not built:**
  - re-planning during a flight;
  - wind direction;
  - flight time treated as uncertain (it is taken as known from the plan);
  - moving edge servers;
  - real radio measurements.

## Verification

- **Tests:** `tests/test_fleet_planner.py` (15 tests) covers:
  - the day's tasks and the charging station's timing;
  - payload limits, deadlines, priorities and spare packs;
  - the oracle's safety;
  - the link model, the options and the choice rule;
  - the bad-case allowance, and that computing energy enters P(success).
- **Saved verdicts:** a test in `tests/test_deliverables.py` recomputes the saved verdicts from the saved tables and checks that the run was logged.
- **Mutation testing:** 12 planted bugs. 7 were caught at first; four tests were added, and all 12 are now caught.
- **Determinism:** a second development run gave byte-identical output files.
- **The test-set log** gained exactly one entry (2026-10-16, fingerprint `768c331b14bdc307`, with its reason). The code on disk still matches it.
- **`pytest`:** 165 passed. `pyflakes` is clean.
