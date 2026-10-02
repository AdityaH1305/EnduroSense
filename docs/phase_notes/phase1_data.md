# Phase 1: Data preparation, validation and locked split

**Status:** complete (2026-10-01); revised after the verification pass (2026-10-02). The full write-up is in [../data_report.md](../data_report.md).

## Built

- **`data/clean.py`:** typed cleaning, gap flags, bounds checks, glitch repair and altitude-label parsing.
- **`data/phases.py`:** segmentation into ground, climb, cruise, descent and hover (uses current to detect motors-off ground), plus a phase-order check.
- **`data/energy.py`:** per-reading power, energy and cumulative energy, and GPS distance with a noise floor.
- **`data/wind.py`:** ambient wind mean, standard deviation and sample count.
- **`data/chains.py`:** motors-off rest voltages, the rest-voltage estimate, asymmetric-window linking, overrides, the rest voltage after each flight, the chain table and the sensitivity check.
- **`data/prepare.py`:** the pipeline that produces the samples, flights and chains tables.
- **`data/split.py`:** the locked group split, balanced grouped folds, hash protection and the test-set lock.
- **`plots.py`:** the shared figure style.
- **Scripts:** `scripts/02_prepare_data.py` and `scripts/03_make_split.py`.
- **Config:** new `cleaning`, `phases`, `chains` and `split` sections in `config.yaml`, plus `config/chain_overrides.yaml`.

## Decisions (and why)

| Decision | Reason |
|---|---|
| `velocity_z` is treated as positive going up | The data shows it (correlation +0.77 with the rate of altitude change); the README is wrong. The climb/descent energy shares in the plan document were corrected. |
| Motors-off (under 2 A) counts as ground | The altitude reading drifts by up to about 14 m after landing. Without this rule, 31 flights ended labelled "hover". |
| Energy is integrated across recording gaps | Capping the time step (the original plan) would lose real energy; the gaps are only 10 short ones. |
| Rest voltages come only from motors-off readings, otherwise they are estimated | Recordings that start or end under load gave sagged voltages (flight 81), which broke the chain links. |
| Asymmetric link window [−0.10, +0.40] V | A resting battery only recovers voltage. A symmetric ±0.3 V window would link swapped-in, partly used batteries. |
| Model A labels come from every chain ending near the reserve, not only those that start full | The energy until the reserve is measured either way. This gives 60 labelled chains instead of 25. |
| Flights are ordered by real timestamps, never by `local_time` text | Text sorting puts "9:22" after "10:05" and mis-ordered 7 days, breaking chain links (found in verification). |
| 4 mis-logged start times are corrected in code (`config/time_corrections.yaml`) | Each correction is confirmed by voltage continuity, and afterwards flight-ID order matches time order for all 209 flights. |
| R6 is forced into test and R5 stays in development | This gives an unseen-distance test at the final evaluation while still allowing a distance check during development. It updates the plan, which had the distance test in Phase 4 only. |

## Verification

- `pytest`: **42 passed**.
  - Unit tests cover cleaning rules, chronological ordering with single-digit hours, the time-correction guard, phase segmentation on a synthetic flight (including altitude drift after landing), short-fragment merging, distance, chain windows and overrides.
  - Data tests cover all readings kept, 195/196 flights in the right phase order, phase energies adding up to flight energy, chain energy never decreasing (checked in true time order), rest voltage falling along every chain, flight 81 linked, the corrected flights joining their neighbours, and corrected times matching flight-ID order.
  - Split tests cover the hash check, no chain on both sides, test share and coverage, R6 being in test, the test lock, the folds partitioning the development data, overwrite refusal and edit detection.
- Mean flight energy (21.23 Wh) matches the Phase 0 profile exactly.

## Verification pass (2026-10-02)

The full list is in `docs/data_report.md` §0. In short:

- **Bug: text-sorted start times.** Fixed with `clean.start_time()` and `clean.chronological()`.
- **4 mis-logged start times.** Corrected in code with evidence.
- **Effect:** 9 chain pairs are now joined, giving 90 chains, 71 of them multi-flight and covering 190 flights.
- **Split:** v1 was invalidated and the new **split v2** was created before any modelling. Its folds are now balanced on cruise flights.
- **Two tests were fixed** that had passed only by repeating the bug, and a misleading `PYTHONHASHSEED` line was removed.
- **Confirmed correct by independent checks:**
  - Post-landing "ground" readings at height are altitude drift.
  - Maximum height matches the planned altitude to within ±8 m.
  - Cruise power matches a recalculation.
  - The split is deterministic.

## Carried into Phase 2

- **Model A labels:** use the chains flagged `near_reserve` (60). Chains ending just above the reserve need the small extrapolation described in the plan.
- **Model B targets:** exclude flight 250 (truncated landing) from the descent and whole-mission targets. Predict cruise *time* rather than distance ÷ commanded speed, because measured cruise speed is about 0.75× commanded at 12 m/s (acceleration and braking).
- **Sensitivity:** the 6 uncertain-link chains are available for a later sensitivity check.
- **Flight 221** (hover test) has no rest voltage; it's not used for Model A.
