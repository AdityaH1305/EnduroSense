# EnduroSense data report (Phase 1)

**Purpose:** guide checkpoint 1. It covers what the data contains after cleaning, how flights are split into phases, how battery chains were rebuilt, and how the locked train/test split was made.
**Reproduce:** `python scripts/02_prepare_data.py` then `python scripts/03_make_split.py`
**Figures:** `results/phase1/figures/`

## 1. Cleaning

| Check | Result | Action |
|---|---|---|
| Missing values | none | — |
| Out-of-order timestamps | none | — |
| Recording gaps over 1 s | 10 (largest 5.3 s) | Energy is integrated across gaps, so none is lost; each gap is flagged |
| Voltage/current outside physical limits | none | — |
| Large voltage jumps | real load changes: they mirror current jumps (correlation −0.93) | kept |
| Glitches that no load change explains | 22 voltage and 54 current readings (0.03%) | replaced by the 5-reading median and flagged |
| Slightly negative current (to −0.33 A) | 29,031 readings, all with motors off | kept (sensor offset) |
| Flights with no GPS | 211–219 (ground tests A1–A3) | expected; not used for mission modelling |
| Mixed `altitude` column (e.g. `"25-50-100-25"`) | 2 flights change altitude en route | parsed into `alt_cruise_m` and `varying_altitude` |

**Correction to the plan document.** In this dataset `velocity_z` is positive going **up** (correlation with the rate of altitude change is +0.77), even though the README describes a north-east-down frame. With the sign handled correctly, **climbing uses about 17% of energy and descending about 22%**. The project plan had these the wrong way round; it has been corrected in `docs/EnduroSense_Project_Plan.docx`.

## 2. Flight phases

Each reading is labelled **ground, climb, cruise, descent or hover**, using height above take-off, vertical speed, horizontal speed and battery current. With the motors off (under 2 A) the drone must be on the ground; this matters because after landing the altitude reading can drift by up to about 4 m. Fragments shorter than 1 s are merged into the previous phase.

- **195 of 196** cruise flights follow the expected order: ground → climb → cruise → descent → ground (hover may appear in the air).
- **Flight 250**'s recording stops 2.6 m above the ground while it is still descending, so about 2 s of landing are missing. It is flagged `truncated_landing` and left out of the descent-energy and whole-mission targets.

| Phase | Share of energy | Share of time | Mean per flight (Wh) |
|---|---|---|---|
| Cruise | 50.4% | 39.6% | 10.7 |
| Descent | 22.0% | 17.7% | 4.7 |
| Climb | 16.5% | 11.1% | 3.5 |
| Hover | 5.6% | 4.3% | 1.2 |
| Ground | 5.5% | 27.3% | 1.2 |

**Finding for Model B.** Cruise power is almost flat across speeds: about 510 W from 4 to 10 m/s, and 534 W at 12 m/s. Payload changes it strongly: 468 W at 0 g, 515 W at 250 g and 560 W at 500 g. Slow missions therefore cost more mainly because they **take longer**. This supports building mission energy as power × time per phase.

Cruise distance (median): route R1 is 455 m, R5 is 505 m and R6 is 819 m.
Ambient wind per flight ranges from 1.3 to 6.7 m/s (median 3.0).

![Phase segmentation](../results/phase1/figures/phase_examples.png)

## 3. Battery chains

**The problem.** Profiling in Phase 0 read each flight's start and end voltage from its first and last 10 readings. Voltage sags under load, and some recordings start or end with the motors running. For example, flight 81 ends while drawing about 20 A, so its "end voltage" was 1–2 V too low and it wasn't linked to the next flight on the same battery.

**The fix.** Rest voltages are read only while the motors are off.
- Where that isn't possible (starts of flights 59, 60, 68 and 221–224; ends of flights 81, 86, 221 and 250), the rest voltage is estimated as `v_rest_end = 4.61 + 0.798·v_rest_start − 0.0356·E_Wh`. This was fitted on flights where both ends are measured, and its error is about ±0.13 V.
- Links now use an **asymmetric window**. A battery resting between flights can only *recover* voltage, so the next flight must start between −0.10 V (measurement noise) and +0.40 V (recovery) of the previous flight's end. A clear voltage *drop* means a different battery was swapped in.
- The typical recovery between linked flights is +0.09 V.

**Result**

| | Phase 0 profile | Phase 1 |
|---|---|---|
| Chains | 103 | **101** |
| Multi-flight chains (flights) | 62 (168) | **68 (176)** |
| Chains ending near the reserve (≤ 22.9 V rest) | 25 (full-to-reserve only) | **60** |
| …of which start nearly full (≥ 25.0 V) | 25 | 38 |
| Largest energy drawn in one chain | 88 Wh | 88.5 Wh |

A chain doesn't need to start full to be useful. For every moment in a chain that ends near the reserve, the energy still available before reaching the reserve is **measured**. This more than doubles the chains available for Model A labels.

**Sensitivity to the linking window** (`results/phase1/chain_sensitivity.csv`):

| Window | Chains | Multi-flight | Near reserve |
|---|---|---|---|
| [−0.10, +0.30] V | 103 | 67 | 61 |
| **[−0.10, +0.40] V (used)** | **101** | **68** | **60** |
| [−0.10, +0.50] V | 101 | 68 | 60 |
| [−0.30, +0.30] V | 98 | 66 | 61 |

The results are stable. Six links sit near the edge of the window and are flagged *uncertain*: flights 4, 6, 17, 119, 213 and 224. The visual review found nothing wrong with them (see [chain_review.md](chain_review.md)). They can be excluded in a later sensitivity check.

![Discharge curve](../results/phase1/figures/discharge_curve.png)

## 4. Locked train/test split

- **Split unit:** the battery chain. All data from one battery stays on one side of the split.
- **Distance test:** R6 (≈820 m, the longest route) is **forced into test**, so the final evaluation includes an unseen mission distance. R5 (≈505 m) stays in development for the distance check during development.
- **How the test chains were chosen:** from 5,000 random candidates, the split whose test set best matched the full data on share of flights, share of near-reserve chains, and the mix of speeds, payloads and altitudes.
- **Protection:** saved in `data/splits/split_v1.json` with a SHA-256 hash (`93ccf2c6…`). The code refuses to overwrite it, detects any edit, and only returns test data when `final=True` is passed.

| Part | Cruise flights | Share | Chains | Near-reserve chains | Speeds / payloads / altitudes |
|---|---|---|---|---|---|
| **Test (locked)** | 40 | 20.4% | 21 | 12 | 5 / 3 / 4, includes R6 |
| Dev fold 0 | 32 | 16.3% | 15 | 10 | 5 / 3 / 4 |
| Dev fold 1 | 31 | 15.8% | 15 | 10 | 5 / 4 / 4, includes R5 |
| Dev fold 2 | 33 | 16.8% | 15 | 9 | 5 / 3 / 4 |
| Dev fold 3 | 30 | 15.3% | 13 | 9 | 5 / 3 / 4 |
| Dev fold 4 | 30 | 15.3% | 14 | 9 | 5 / 3 / 4 |

## 5. Needs the guide's confirmation

1. **Reserve level.** 22.6 V rest voltage (about 3.77 V per cell). 31 chains reach it outright and 60 come within 0.3 V. A higher reserve would label more chains fully, while a lower one would extrapolate more.
2. **Distance test.** Is using R6 as the unseen-distance test in the final evaluation acceptable?

## 6. Output tables

| File | Rows | Contents |
|---|---|---|
| `data/processed/samples.parquet` | 257,896 | Cleaned readings plus phase, height above take-off, power, energy, cumulative flight/chain energy, distance, glitch/gap flags and battery chain |
| `data/processed/flights.parquet` | 209 | Planned settings, duration, energy and time per phase, distance, cruise power, ambient wind, rest voltages (and whether estimated), phase-order check, chain, link jump and confidence |
| `data/processed/chains.parquet` | 101 | Flights, first and last rest voltage, energy, `near_reserve`, `reaches_reserve`, `starts_full`, uncertain links |
