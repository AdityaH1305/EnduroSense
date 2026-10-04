# EnduroSense: final results

**Evaluated 2026-10-04 on the locked test set.** Every opening is logged in `results/final/test_access_log.json`: the evaluation itself (code fingerprint `64bbb0061a140831`), and one rerun after verification pass 5 changed reporting code only (`68021ebc62ad50ff`). The rerun reproduced every test number exactly.
**Reproduce:** `python scripts/run_all.py --final` (about 30 minutes from scratch). A complete rebuild from the raw data with empty caches reproduced all 74 result files, all 19 model files and every data table byte for byte.
**Settings:** reserve 22.6 V, τ = 0.95, 90% ranges (project defaults; not yet confirmed by the guide).
**Protocol:** what was measured and what counts as success was written down before the test set was opened ([final_evaluation_protocol.md](final_evaluation_protocol.md)).

The test set is 16 battery chains and 40 flights that no model, setting or calibration ever saw. 11 of those chains (30 flights, 5,708 readings) have an energy-to-reserve label and are used for the battery model; all 40 flights are used for the mission model. Routes R2, R3 and R6 (7 flights) do not exist in the development data.

## The four success criteria

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Model A beats both non-ML baselines | **Met** | 2.19 Wh against 2.90 Wh (voltage lookup) and 4.21 Wh (energy counting) |
| 2 | Model B within 5%, including unseen routes | **Not met** | 3.1% over all test flights, but 5.7% on the 7 unseen-route flights |
| 3 | 90% ranges cover 88–92% | **Not met, within sampling noise** | Model A 93.8% (interval 83–100%); Model B 80.0% (interval 65–92.5%) |
| 4 | EnduroSense decides better than minutes-left and point estimates | **Met** | Unsafe approvals 0.6% against 12.4% and 3.6%; ranking quality 0.994 against 0.975 and 0.994 |

Two met, one missed narrowly, one missed inside its noise band. The details, including the parts that did not hold up, follow.

## 1. Model A: energy available

### Point accuracy (`model_a_test_metrics.csv`)

| Model | Cross-validation (Wh) | **Test (Wh)** | Test, vs voltage lookup (95% interval) |
|---|---|---|---|
| **GRU ensemble, calibrated (the main model)** | 2.37 | **2.19** | **−0.71 (−1.33 to −0.12)** |
| LSTM | 2.45 | 2.32 | −0.58 (−1.18 to +0.06) |
| XGBoost + physics | 2.59 | 2.40 | −0.50 (−1.02 to +0.01) |
| Linear Regression | 2.64 | 2.63 | −0.27 (−0.92 to +0.41) |
| Random Forest | 2.72 | 2.71 | −0.19 (−0.85 to +0.58) |
| XGBoost | 2.91 | 2.75 | −0.14 (−0.56 to +0.30) |
| GRU (single) | 2.39 | 2.81 | −0.08 (−0.73 to +0.62) |
| *Voltage lookup (non-ML)* | 2.82 | 2.90 | — |
| *Energy counting (BMS)* | 3.98 | 4.21 | +1.31 (+0.60 to +1.93) |
| *Fixed capacity (reference)* | 13.38 | 10.16 | +7.26 (+2.88 to +10.97) |

The cross-validation column is for development readings that have a pre-flight voltage, like every test reading (column `cv_mae_supported`). Over all development readings, including three chains without one, the figures are 0.2–0.5 Wh higher; those are the ones in the Phase 3 notes.

- **The main model held up:** 2.19 Wh on unseen batteries against 2.37 Wh in cross-validation, and the only model whose advantage over the fair baseline is clear (the interval excludes zero). It was better on 9 of the 11 test batteries.
- **A single GRU did not.** It was the best single model in cross-validation (2.39 Wh) but scored 2.81 Wh on test, no better than the baseline. Averaging five networks is what made the result dependable.
- **The other algorithms scored about what cross-validation predicted** (within 0.2 Wh), and their order is not stable: with 11 batteries, differences of 0.2–0.5 Wh between them are inside the noise. The safe statement is: every ML model is at or below the baseline, and only the ensemble is clearly below.
- **Minutes left:** 0.81 min error for the main model, against 0.85 for the voltage lookup. As in development, the minutes figure is dominated by the unknown future power draw.

### Ranges (`model_a_test_ranges.csv`)

| | Development (held out) | **Test** |
|---|---|---|
| 90% range covers | 89.5% | **93.8%** (interval 83–100%) |
| Width | 11.2 Wh | 11.9 Wh |
| Truth below the range (unsafe side) | 3.7% | 6.2% |
| Truth above the range | 6.8% | 0.0% |
| Readings where the model abstains | 3 chains | none |

- **Coverage is slightly above the target band** and well inside what 11 batteries can show.
- **All the misses are on the unsafe side and come from two batteries.** 9 of 11 test batteries were covered 100% of the time. Chain 70 was over-estimated for 58% of its readings and chain 86 for 13%. This is the pattern Phase 5 predicted: the error is a per-battery offset, so risk is per battery, not per moment.
- **The error is the model's, not the label's.** Chain 70's energy-to-reserve label is a measured one, and the model over-estimated it by 7.4 Wh on average (chain 86: 5.1 Wh, with an extrapolated label). On the other nine batteries the model was within 1.5 Wh on average, mostly slightly under.

## 2. Model B: energy a mission needs

### Point accuracy (`model_b_test_metrics.csv`), main model (Physics-first)

| Flights | n | Error | Bias |
|---|---|---|---|
| Cross-validation (development) | 153 | 2.35% (0.49 Wh) | −0.02 Wh |
| **All test flights** | 40 | **3.1% (0.70 Wh)** | +0.15 Wh |
| Route seen in development (R1) | 33 | 2.61% (0.56 Wh) | −0.11 Wh |
| **Unseen routes (R2, R3, R6)** | 7 | **5.66% (1.35 Wh)** | **+1.35 Wh** |
| of which R6, the longest route | 5 | 5.86% (1.48 Wh) | +1.48 Wh |

- **On the route it was trained on, the model generalises to new batteries and days:** 2.6% against 2.35% in cross-validation.
- **On unseen routes it misses the 5% target, in the safe direction.** All 7 unseen-route flights were over-predicted (the model said they would need more energy than they did).
  - **Where the excess is:** in the cruise legs, about 1.0 Wh of R6's 1.5 Wh; climb, descent, hover and ground are each within 0.3 Wh.
  - **R6 is outside what the model was trained on:** its cruise distance is about 820 m against 443–568 m in development, and it has four legs where development flights almost always have three. One leg (375 m) is beyond the longest tested leg (334 m), so the model's range warning does fire for all five R6 flights.
  - **R2 and R3 are not flagged** (their legs are within the tested range) but were still over-predicted, by 0.6 and 1.4 Wh. That is one flight each, too few to conclude anything.
- **No variant was clearly better.** Across the seven variants the unseen-route error ranged from 3.9% (XGBoost) to 5.9%; with 7 flights that ordering means little, and the choice of main model was not revisited after seeing it.

### Ranges (`model_b_test_ranges.csv`)

| | Development (held out) | **Test** |
|---|---|---|
| 90% range covers, single flights | 91.5% | **80.0%** (interval 65–92.5%) |
| Width | 2.10 Wh | 2.09 Wh |
| Truth above the range (unsafe side) | 3.9% | 12.5% |
| 90% range covers, two-sortie missions | 94.5% | 91.2% (34 missions) |
| 90% range covers, three-sortie missions | 98.0% | 90.9% (11 missions) |

- **The single-flight ranges were too narrow on test.** 80% against a 90% target. 90% is still inside the resampling interval, so this is not proof of a fault, but the direction is unfavourable and it should be read as a real weakness.
- **Where the misses are:** 4 of the 6 seen-route misses are at 100 m altitude, all under-predicted. Phase 5 had flagged 100 m as the weak spot (82% coverage in development). The 2 unseen-route misses are the over-predictions described above.
- **The multi-sortie ranges held** (91%), which supports the shared-error rule adopted in verification pass 4.

## 3. Decisions

Battery states and missions here are all from the test set: 588 battery states from 11 unseen batteries (30 of them pre-flight) and 85 missions (40 single flights, 34 two-sortie, 11 three-sortie).

### Each policy as specified (`operating_points.csv`)

| Policy | Development: unsafe approvals | **Test: unsafe approvals** | Test: wasted refusals | Test, pre-flight: unsafe | wasted |
|---|---|---|---|---|---|
| P1 minutes left ≥ duration (the brief) | 10.5% | **12.4%** | 3.3% | 12.8% | 1.9% |
| P2 energy margin ≥ 0 | 2.9% | **3.6%** | 3.9% | 4.3% | 4.1% |
| **P3 EnduroSense, τ = 0.95** | 0.25% | **0.61%** | 24.3% | **0.00%** | 25.2% |
| P4 oracle | 0% | 0% | 0% | 0% | 0% |

- **The headline holds on unseen data:** the brief's rule approves 1 in 8 missions that would fail; EnduroSense approves 1 in 160. Of everything EnduroSense approved, 1.1% was unsafe, against 14.9% for minutes-left.
- **At pre-flight states,** where real decisions are made, EnduroSense approved no failing mission.
- **Its 71 unsafe approvals all involve two batteries** (62 on chain 70, 9 on chain 86), the same two the battery model over-estimated. The median shortfall was 2.2 Wh into the reserve and the worst 7.3 Wh.
- **The price is the same as in development:** it refuses about a quarter of feasible missions, and those are close calls (median margin 3.8 Wh, 93% under 10 Wh).

### Against the alternatives with a safety margin tuned on development data

| | Test: unsafe approvals | wasted refusals | Test, pre-flight: unsafe | wasted |
|---|---|---|---|---|
| P1 × its development margin | 0.16% | 91% | 0.00% | 37% |
| P2 + its development margin (7.0 Wh; 11.4 Wh pre-flight) | 0.55% | 24.4% | 0.00% | 31.5% |
| **P3 EnduroSense, τ = 0.95** | 0.61% | 24.3% | 0.00% | **25.2%** |

- **Minutes-left cannot be rescued with a margin:** to be safe it has to refuse 91% of feasible missions.
- **Point estimates plus a fixed margin did as well as the probability over all pairs.** The two are indistinguishable on test (0.55% and 0.61% unsafe at the same refusal rate), and none of the paired comparisons of their ranking is clearly non-zero except a small gain on borderline cases (+0.007 AUC, interval 0.001 to 0.016).
- **At pre-flight states the probability refused fewer feasible missions** (25% against 32%) at the same zero unsafe approvals. The fixed margin needed there (11.4 Wh) is different from the one needed over all pairs (7.0 Wh); τ = 0.95 was the same number in both.
- **Honest summary:** the large, robust gain is from deciding on energy with a calibrated safety margin instead of on minutes. Whether that margin is expressed as a probability or as a well-chosen number of watt-hours made little measurable difference on this data. The probability's advantages are that it needs no tuning and adapts to the situation.

### Is P(success) honest on unseen data? (`probability_reliability.csv`)

| Predicted | Pairs | Actually succeeded |
|---|---|---|
| below 5% | 9,835 | 0.0% |
| 20–40% (mean 29%) | 573 | 28% |
| 40–60% (mean 50%) | 453 | 61% |
| 80–90% (mean 86%) | 626 | 88% |
| 90–95% (mean 93%) | 567 | 91% |
| **95–99% (mean 97.4%)** | 862 | **94.2%** |
| above 99% (mean 99.9%) | 5,600 | 99.6% |

It tracks reality, but it is **slightly over-confident at the top**: missions rated 95–99% succeeded 94% of the time. The cause is the two over-estimated batteries. In development the same band was 97.6%.

### Fleet simulation on 8 unseen batteries (`fleet_simulation.csv`)

| Policy | Unsafe missions per 100 flown | Completed per day (of 40) | Battery swaps per day |
|---|---|---|---|
| P1 minutes left (the brief) | **12.0** | 40.0 | 15.5 |
| P2 energy, point estimates | 4.9 | 40.0 | 16.2 |
| P1 × development margin | 0.0 | 10.9 | 29.2 |
| P2 + development margin | 0.43 | 39.6 | 18.4 |
| **P3 EnduroSense, τ = 0.95** | **0.73** | 39.2 | **17.0** |
| P4 oracle | 0.0 | 39.9 | 16.2 |

- **Against the brief's rule: 16 times fewer unsafe missions** for 10% more battery swaps.
- **Against point estimates with a margin:** slightly more unsafe missions (0.73 against 0.43 per 100) with 8% fewer swaps. The two sit at nearby points of the same trade-off.
- In development the ratio to the brief's rule was 46 times; on test it is 16 times. Both are large; the test figure is the one to quote.

### Are the two models' errors independent?

On the 30 test flights with both predictions, the correlation of the two errors is +0.43 (interval +0.14 to +0.66), stronger than in development (+0.16). It is still in the safe direction: the real spread of the margin error was 3.37 Wh against 3.77 Wh assumed, and the 90% range of the margin covered all 30 real pairs.

## What to claim, and what not to

**Supported by the test set**
- A calibrated GRU ensemble estimates energy to reserve to about 2.2 Wh (about 3% of a pack's usable energy) on unseen batteries and beats a fair non-ML baseline.
- Mission energy is predicted to about 3% for the kind of mission flown in training.
- Deciding on energy with a calibrated margin cuts unsafe approvals from about 12% (minutes-left) to under 1%, and to zero at take-off decisions, on unseen batteries and flights.
- The method is honest about what it doesn't know: the unsafe cases trace to two batteries whose capacity the model over-estimated, exactly the per-battery risk identified during development.

**Not supported, and should not be claimed**
- That the mission model is within 5% on routes much longer than those it was trained on (5.7–5.9%, over-predicted).
- That the mission model's 90% ranges are calibrated on new data (80% on test).
- That a probability is measurably better than point estimates plus a well-chosen margin.
- That a single GRU beats the baseline.

**Limits of this evaluation**
- 11 batteries and 40 flights from one drone and one battery type. Every interval above is wide for that reason.
- The results were checked after the fact by verification pass 5 (`docs/verification_log.md`): every number above was recomputed with separate code, and no development file was found to contain a test reading, flight or battery.
- "Truth" for the battery is an energy-to-reserve label built from rest voltages, with its own error of about 1.5–2 Wh.
- Decisions are evaluated by pairing real battery states with real missions, not by flying them.
