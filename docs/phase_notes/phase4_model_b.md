# Phase 4: Model B (energy a mission needs)

**Status:** complete (2026-10-03).
**Reproduce:** `python scripts/06_model_b.py` (about 4 minutes; development data only)
**Outputs:** `results/model_b/` (tables, out-of-fold predictions, figures) and `models/model_b/` (fitted on all development data)

## How Model B works

A mission's energy is built from its parts (`mission.py`, `models/model_b.py`):

> **E = climb + Σ legs (power × time) + hover between legs + descent + ground**

| Part | Trained on | Target | Inputs (all known before take-off) |
|---|---|---|---|
| Leg power | 459 legs at commanded speed | W | speed, payload, altitude, wind |
| Leg time | the same legs | seconds beyond distance ÷ speed | speed, leg distance, payload, wind |
| Climb | 153 flights | Wh | altitude, payload, wind |
| Descent | 153 flights | Wh | altitude, payload, wind |
| Hover | 153 flights | Wh | number of legs, payload, wind |
| Ground | — | median of training flights (1.1 Wh) | — |

**Algorithms compared for each part:**
- **Physics:** rotor momentum theory for power, acceleration kinematics for leg time, and work ∝ height or number of legs for the rest.
- **ML:** Linear Regression, Random Forest and XGBoost, with 10-trial random searches.
- **Hybrid:** Physics + XGBoost, where XGBoost learns only the physics model's errors.

**LSTM and GRU aren't used,** because a mission plan isn't a time series.

## Component results (grouped CV, by battery chain)

| Part | Physics | Linear | Random Forest | XGBoost | Physics + XGBoost |
|---|---|---|---|---|---|
| Leg power (W) | 22.3 | **18.4** | 19.2 | 19.2 | 18.9 |
| Leg time (s) | 0.49 | 0.50 | 0.48 | 0.48 | **0.47** |
| Climb (Wh) | **0.20** | 0.23 | 0.28 | 0.22 | 0.20 |
| Descent (Wh) | 0.34 | 0.36 | 0.38 | 0.32 | **0.31** |
| Hover (Wh) | **0.21** | 0.21 | 0.22 | 0.23 | 0.22 |

**Physics parameters:**
- **Implied acceleration: 2.1 m/s².** Each leg loses about v ÷ 2.1 s to speeding up and braking.
- **Climb:** about 0.05 Wh per metre, plus 0.02 Wh per metre for each kg of payload.
- **Leg power:** the fit doesn't separate mass, constant power and rotor inefficiency well.
  - Left free, it gives an unphysical ~5 kg airframe at the same accuracy. So the airframe mass is limited to the Matrice 100's real range (2.4 kg with battery to 3.6 kg maximum take-off).
  - Even then, the fit settles at the 3.6 kg limit: the data shows more payload sensitivity than momentum theory predicts. One plausible cause is drag from the payload box.
  - The physics model is therefore a physically shaped curve, and its individual parameters shouldn't be read as measurements.

## Mission results (each held-out flight rebuilt from its plan)

| Model | Error (Wh) | Error (%) | R² | vs Physics, 95% CI |
|---|---|---|---|---|
| **Best component each** | **0.49** | **2.34%** | 0.979 | **−0.09 [−0.16, −0.01]** |
| Physics + XGBoost | 0.50 | 2.41% | 0.979 | −0.07 [−0.13, −0.01] |
| XGBoost | 0.52 | 2.47% | 0.978 | −0.05 [−0.13, +0.02] |
| Physics | 0.57 | 2.68% | 0.972 | — |
| Linear Regression | 0.60 | 2.84% | 0.971 | +0.03 [−0.06, +0.11] |
| Random Forest | 0.72 | 3.42% | 0.958 | +0.14 [+0.03, +0.26] |
| *Average flight (naive)* | *3.91* | *19.6%* | ≈0 | — |

- **"Best component each"** uses Linear Regression for leg power, Physics + XGBoost for leg time and descent, and Physics for climb and hover. Each part was chosen by its own CV error.
- **Every model beats the 5% target** set in the plan.
- Configurations were chosen on the same folds that score them, so these numbers are slightly optimistic; Phase 7 gives the unbiased figure.

## Generalisation: missions with settings never seen in training

Each test leaves out every development flight with one setting value, trains on the rest, then predicts the held-out missions. The table shows mission energy error (%).

| Test | Physics | Linear | Random Forest | XGBoost | Physics + XGBoost | Best each |
|---|---|---|---|---|---|---|
| Unseen speed | 3.3 | 3.1 | 3.6 | 3.3 | 3.1 | **2.6** |
| Unseen payload | 3.5 | 3.4 | **9.6** | **9.3** | 3.1 | **2.5** |
| Unseen altitude | 2.9 | 3.4 | **14.0** | **14.0** | 2.6 | **2.6** |
| Unseen day | 2.7 | 2.9 | 3.1 | 2.5 | 2.4 | **2.3** |
| R5 route = day 1 (4 flights) | 8.5 | 3.9 | 7.0 | 5.2 | 7.4 | 5.7 |

**What this shows:**
1. **Tree models (Random Forest, XGBoost) can't extrapolate.** For an altitude or payload outside their training range, they predict the nearest value they've seen, giving 9–14% error. Models with a physics-shaped structure stay at 2.5–3.5%. This is the clearest case for physics-informed modelling in the project.
2. **"Best component each" is the most accurate and the most robust:** 2.3–2.6% on every clean test. **It's the main Model B.**
3. **The R5 test is inconclusive about route length.** R5 is the only longer route in development, but its 4 flights are also the only development flights from the first flying day (2019-04-07). That day is the most over-predicted of all 17 days even in normal CV (+1.2 to +1.8 Wh), so route length and day can't be separated. The real unseen-distance test is R6, in the locked test set (Phase 7).

## Plausibility: missions unlike any in the data

`results/model_b/example_missions.csv` holds deliveries that drop the payload half-way, which no recorded flight does:
- **300 m delivery with 500 g:** about 20.5 Wh. At 600 m it's about 31.5 Wh: the extra 600 m flown costs about 11 Wh, matching ~510 W × 75 s.
- **Slow (4 m/s) empty delivery:** about 25 Wh, more than the same delivery at 8 m/s, because time aloft dominates.
- **Agreement:** all families agree within about 1.5 Wh.

## Choices for later phases

- **The main Model B is "Best component each"** (`models/model_b/Best_component_each.pkl`).
- **Phase 5 (uncertainty):** mission uncertainty will come from the parts' out-of-fold errors, resampled jointly per flight in case they're correlated.

## Verification

- `pytest`: **72 passed**.
- **New tests:** momentum-theory curve shape (power falls, then rises with speed); the mass-bounded fit; kinematic acceleration recovery; the linear work model; planning-time inputs only; folds sharing no flight or battery; and **the batch evaluation path matching the single-mission path** to 9 significant digits.
- **Reproducible:** a full re-run gives identical results.
