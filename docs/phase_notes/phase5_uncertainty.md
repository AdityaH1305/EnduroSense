# Phase 5: Uncertainty and calibration

**Status:** complete (2026-10-03).
**Reproduce:** `python scripts/07_uncertainty.py` (about 10 minutes the first time; the ensemble CV is cached; development data only)
**Outputs:** `results/uncertainty/`, plus the calibrated models `models/model_a/model_a_calibrated.pt` and `models/model_b/model_b_calibrated.pkl`

## What "calibrated" means here

Every prediction now comes as a distribution, stored as 23 quantiles from 1% to 99%. A range is **calibrated** if a "90% range" contains the true value about 90% of the time on batteries or flights the model never saw. Every figure below is measured "leave one fold out": the ranges for a fold are set using only the *other* folds' held-out errors.

## Model A: energy available

### Method

1. **A deep ensemble of 5 GRUs** (`models/probabilistic.py`), each predicting a mean and a spread. Mean and spread are trained by **separate objectives**:
   - **The mean** is trained exactly like the Phase 3 GRU (Huber loss, early stopping on validation error).
   - **The spread head** learns the size of the mean's errors but passes no gradient back, so it can't disturb the mean. A test checks this.
   - **Why:** training mean and spread jointly on the likelihood (the textbook deep-ensemble recipe) made the mean worse, at 2.51 Wh against 2.39 Wh for the single GRU.
2. **Conformal calibration by battery chain** (`uncertainty/quantiles.py`, `uncertainty/battery.py`).
   - Held-out standardised errors, (truth − mean) ÷ spread, set a calibrated multiplier for each probability level.
   - Every chain gets equal weight, because thousands of readings from one battery aren't independent.

### The model abstains without a pre-flight voltage

Three development chains (13, 14 and 68; 914 rows, including the hover tests) have recordings that start with the motors already running, so the battery's starting charge is unknown.
- **What goes wrong there:** errors reach 10–20 Wh, a first version covered only 42% with its "90%" range, and three chains are too few to calibrate.
- **What the model does:** a real drone always reads its battery before take-off, so the model **requires a pre-flight voltage reading and abstains otherwise** (`battery.supported()`). Phase 6 must treat an abstention as "do not approve".
- **Scope:** everything below is on the 46 supported chains (22,970 rows).

### Results (`model_a_intervals.csv`)

| Method | Point error | 90% range covers | Width | Truth *below* the range | Pinball |
|---|---|---|---|---|---|
| Raw ensemble (uncalibrated) | 2.34 Wh | 88.7% | 10.5 Wh | **6.6%** | 0.773 |
| Calibrated, constant width | 2.38 Wh | 90.5% | 12.8 Wh | 4.0% | 0.807 |
| **Calibrated, per-reading spread (main)** | **2.38 Wh** | **89.5%** | **11.2 Wh** | **3.7%** | 0.791 |

- **Point accuracy improved.** The ensemble's 2.34 Wh beats the single GRU's 2.39 Wh on the same rows.
- **The target is met:** 89.5% (plan: 88–92%). The other levels are 50% → 52%, 80% → 81% and 95% → 94%.
- **Calibration fixes the unsafe side.** "Truth below the range" means the battery has *less* energy than the model's lower bound. The raw ensemble did this 6.6% of the time; calibrated, it's 3.7%. The price is a slightly worse pinball score than the raw ensemble.
- **The lower tail is heavier than a normal curve.** The calibrated multipliers are −2.08 spreads at the 5% level (a normal curve gives −1.64) and −2.93 at 1% (normal: −2.33). The model's optimistic errors are larger than a bell curve assumes.
- **The per-reading spread is worth having:** the same coverage with ranges 12% narrower than a constant-width range.

### How to read the 90%: it's per battery, not per moment

- **Errors are mostly a persistent per-battery offset.** The spread between batteries is 3.3 Wh, against 1.6 Wh within a battery.
- **So coverage is close to all-or-nothing per battery.** 39 of 46 chains are covered almost all the time, and 7 are mostly outside their range (figure `model_a_calibration.png`, right).
- **Of those 7,** 4 are on the safe side (the model predicts too little) and **3 of 46 (6.5%) are on the unsafe side**, with the model over-estimating by 6–8 Wh.
- **Counted per battery,** the truth falls below the 5% level 5.2% of the time, which is nominal.
- **For Phase 6:** risk is per battery. Consecutive missions on one battery share the same error, so their outcomes are correlated.

### Coverage by subgroup (`model_a_coverage_by_subgroup.csv`)

| Subgroup | Coverage of 90% range | Truth below range |
|---|---|---|
| Above / below the reserve | 89.6% / 88.1% | 3.7% / 4.4% |
| Predicted ≤ 10 / 10–30 / > 30 Wh | 87.8% / 88.7% / 90.6% | 1.9% / 3.0% / 5.1% |
| Motors off / on | 92.5% / 88.6% | 2.1% / 4.2% |
| Per fold (5) | 82% – 98% | 0% – 11.9% |

- **Below the reserve is no longer the weak spot** (Phase 3's concern): the unsafe-side rate there is 4.4%.
- **Fold-to-fold variation is the main limitation:** with about 9 chains per fold, one or two unusual batteries move a fold's coverage a lot.

### Small-sample correction: a deliberate choice (`model_a_correction_comparison.csv`)

Conformal theory widens ranges when few groups are available to calibrate on. With about 40 chains that widening is large:

| Correction | 90% range covers | Width |
|---|---|---|
| **None (used)** | 89.5% | 11.2 Wh |
| Standard q(1 + 1/G) | 93.9% | 13.8 Wh |
| Round-up | 96.8% | 19.3 Wh |

The default is **none**. The ranges are calibrated to the nominal level, and caution is applied explicitly by the decision threshold τ in Phase 6, not hidden inside a wider range. The setting is `uncertainty.finite_sample_correction`.

## Model B: energy a mission needs

### Method

On held-out flights we know how wrong each part of the estimate was. For a new mission, every held-out flight's set of errors is **replayed together** on the mission's parts, and the spread of totals gives the distribution (`uncertainty/mission.py`).
- **Why jointly:** the parts' errors are correlated. Descent and ground are at −0.63, because touchdown timing moves energy between them. Treating the parts as independent would overstate the uncertainty (0.82 against the real 0.69 Wh).
- **Units:** climb, descent, hover and ground errors are in Wh. The **cruise-leg error is a percentage** (4.3%), so missions with more or longer legs get proportionally wider ranges.

### Results (`model_b_intervals.csv`)

| Method | 90% range covers | Width | Truth *above* the range |
|---|---|---|---|
| One percentage error on the whole mission | 90.2% | 2.09 Wh | 3.9% |
| **Per-part errors, replayed jointly (main)** | **91.5%** | **2.10 Wh** | **3.9%** |

- **The target is met.** "Truth above the range" (the mission needs *more* energy than the upper bound, which is the unsafe side for Model B) is 3.9%.
- **The two methods perform alike on recorded flights,** which are all a similar size. The per-part method is the main one because it scales sensibly to other missions:

| Mission | Median | 90% range |
|---|---|---|
| R1-like loop, 250 g, 8 m/s, 50 m | 18.7 Wh | 17.7 – 19.5 |
| Delivery 300 m, 500 g, 8 m/s, 50 m | 20.4 Wh | 19.3 – 21.3 |
| Delivery 600 m, 500 g, 8 m/s, 50 m | 31.2 Wh | 29.1 – 32.8 (wider: more cruise) |
| Delivery 300 m, 500 g, 12 m/s, 100 m | 23.6 Wh | 22.6 – 24.4 |

### Weak spots (`model_b_coverage_by_subgroup.csv`)

Coverage of the 90% range is **83% at 12 m/s** (30 flights) and **82% at 100 m altitude** (34 flights); every other speed, payload and altitude is between 90% and 97%. This fits the small under-prediction at 12 m/s found in Phase 4. With about 30 flights per group these are within sampling noise of 90% (±11 points), but they are the settings to watch in Phase 6 and Phase 7.

## What Phase 6 gets

- **`uncertainty.battery.load_calibrated(path, windows)`** returns a `CalibratedBatteryModel`. Its `.distributions(rows)` gives one `PredictiveDistribution` per reading, or `None` where it abstains.
- **`models/model_b/model_b_calibrated.pkl`** holds a `CalibratedMissionModel`. Its `.distribution(MissionSpec)` gives a `PredictiveDistribution`, and `.warnings(spec)` lists out-of-range inputs.
- **`PredictiveDistribution`** has `.quantile(q)`, `.cdf(x)`, `.sample(n, rng)`, `.interval(coverage)` and `.median`, with normal-shaped tails beyond the 1% and 99% levels.

## Verification

- `pytest`: **85 passed**; `pyflakes` clean.
- **New tests:**
  - equal weight per group
  - weighted quantiles (a heavy group can't dominate)
  - the correction modes
  - the distribution's cdf and quantile being inverses, including in the tails
  - **conformal ranges covering *new groups* at the nominal rate** on synthetic data with per-group offsets
  - interval and pinball arithmetic
  - abstention without a pre-flight voltage
  - mission replay keeping errors together and scaling the leg error
  - the spread head being unable to change the mean
  - both saved calibrated models loading and behaving sensibly
- **No test data was used.** Everything fitted or calibrated uses development data, and held-out evaluation is leave-one-fold-out.
