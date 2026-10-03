# Phase 4: Model B (energy a mission needs)

**Status:** complete (2026-10-03); revised by a verification pass the same day (see [../verification_log.md](../verification_log.md), pass 3).
**Reproduce:** `python scripts/06_model_b.py` (about 6 minutes; development data only)
**Outputs:** `results/model_b/` (tables, out-of-fold predictions, figures) and `models/model_b/`. The main model is `models/model_b/model_b.pkl`.

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

**"The plan" of a recorded flight** uses its measured leg lengths, since the dataset doesn't give the programmed waypoints.

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
  - Left free, it gives an unphysical ~5 kg airframe at the same accuracy. So the airframe mass is limited to the Matrice 100's real range (2.4–3.6 kg).
  - Even then, the fit settles at the 3.6 kg limit: the data shows more payload sensitivity than momentum theory predicts. One plausible cause is drag from the payload box.
  - The physics model is therefore a physically shaped curve, and its individual parameters shouldn't be read as measurements.

## Mission results (each held-out flight rebuilt from its plan)

| Model | Error (Wh) | Error (%) | Worst systematic bias | vs Physics, 95% CI |
|---|---|---|---|---|
| Best component each | 0.49 | 2.34% | **−0.44 Wh at 12 m/s** | −0.09 [−0.16, −0.01] |
| **Physics-first (main model)** | **0.49** | **2.35%** | −0.27 Wh at 500 g | **−0.08 [−0.14, −0.02]** |
| Physics + XGBoost | 0.50 | 2.41% | −0.29 Wh at 500 g | −0.07 [−0.13, −0.01] |
| XGBoost | 0.52 | 2.47% | −0.25 Wh at 500 g | −0.05 [−0.13, +0.02] |
| Physics | 0.57 | 2.68% | +0.67 Wh at 4 m/s | — |
| Linear Regression | 0.60 | 2.84% | −0.49 Wh at 12 m/s | +0.03 [−0.06, +0.11] |
| Random Forest | 0.72 | 3.42% | −0.68 Wh at 500 g | +0.14 [+0.03, +0.26] |
| *Average flight (naive)* | *3.91* | *19.6%* | −5.6 Wh at 4 m/s | — |

"Worst systematic bias" is the largest average signed error for any speed or payload (`mission_bias.csv`). **Negative means the model under-predicts the energy needed,** which is the unsafe direction.

### Why the main model is "physics-first", not "best component each"

- **What "best component each" does.** It picks each part's algorithm by CV error alone. For leg power it chose Linear Regression (18.4 W against 18.9 W for the hybrid).
- **The problem.** A straight line can't follow the power curve, which is flat from 4 to 10 m/s and then rises. The result is a systematic **under-prediction of 0.44 Wh at 12 m/s**.
- **The rule.** Physics-first uses a physics-structured part (Physics or Physics + XGBoost) unless a pure-ML part is more than 5% better (`model_b.physics_first_margin`).
- **What it uses.** Physics + XGBoost for leg power, leg time and descent; Physics for climb and hover.
- **The result.** The same accuracy (0.490 against 0.489 Wh; the difference is indistinguishable) with half the systematic bias at 12 m/s (−0.23 Wh).
- **The cost.** It's slightly less accurate on unseen speeds and payloads (see below).
- **Rejected alternative.** Physics + a *linear* correction removes the 12 m/s bias but creates a +0.55 Wh bias at 4 m/s and is less accurate overall (0.52 Wh).

### How robust is the headline number? (`robustness.csv`)

| Variant (physics-first) | Error (Wh) | Error (%) |
|---|---|---|
| As reported | 0.490 | 2.35% |
| Algorithm for each part chosen inside every training fold (nested, no selection optimism) | 0.512 | **2.47%** |
| No wind input at all | 0.486 | 2.32% |
| Only a typical wind known when planning | 0.505 | 2.41% |

- **The conservative estimate is 2.47%.** The reported 2.35% carries a little selection optimism. The unbiased figure comes from the locked test set in Phase 7.
- **Wind barely matters in the tested range (1.3–6.7 m/s).** The model is as accurate with no wind input at all. So using each flight's measured wind, a "perfect forecast", doesn't flatter the results. A likely reason is that only wind speed is known, not its direction along the route, and the closed-loop routes cancel head and tail winds.
- **An independent from-scratch implementation** of the physics family reproduces the script's per-flight predictions to 10⁻¹⁴ Wh.

## Generalisation: missions with settings never seen in training

Each test leaves out every development flight with one setting value, trains on the rest, then predicts the held-out missions. The table shows mission energy error (%).

| Test | Physics | Linear | Random Forest | XGBoost | Physics + XGBoost | Best each | **Physics-first** |
|---|---|---|---|---|---|---|---|
| Unseen speed | 3.3 | 3.1 | 3.6 | 3.3 | 3.1 | 2.6 | **3.0** |
| Unseen payload | 3.5 | 3.4 | **9.6** | **9.3** | 3.1 | 2.5 | **3.1** |
| Unseen altitude | 2.9 | 3.4 | **14.0** | **14.0** | 2.6 | 2.6 | **2.5** |
| Unseen day | 2.7 | 2.9 | 3.1 | 2.5 | 2.4 | 2.3 | **2.3** |
| R5 route = day 1 (4 flights) | 8.5 | 4.0 | 7.0 | 5.2 | 7.4 | 5.7 | 6.8 |

**What this shows:**
1. **Tree models (Random Forest, XGBoost) can't extrapolate.** For an altitude or payload outside their training range they predict the nearest value they've seen, giving 9–14% error. Physics-structured models stay at 2.5–3.5%. An out-of-range example in `example_missions.csv` shows the same thing: for 1 kg at 15 m/s and 150 m, the physics-based models predict 30–32 Wh while the tree models still say about 22 Wh.
2. **The main model holds 2.3–3.1% on every clean test.**
3. **The R5 test is inconclusive about route length.** R5 is the only longer route in development, but its 4 flights are also the only development flights from the first flying day (2019-04-07). That day is the most over-predicted of all 17 days even in normal CV (+1.2 to +1.8 Wh), so route length and day can't be separated. The real unseen-distance test is R6, in the locked test set (Phase 7).

## Safeguards in the model

- **No negative predictions.** Energies and leg overheads can't go negative. Without this, the leg-time formula would predict flying *faster than commanded* below about 3 m/s.
- **Range warnings.** `MissionEnergyModel.extrapolation_warnings(spec)` lists every input outside the tested range (`tested_ranges.json`):

  | Input | Tested range |
  |---|---|
  | Speed | 4–12 m/s |
  | Payload | 0–750 g (only one flight above 500 g) |
  | Altitude | 25–100 m |
  | Wind | 1.3–6.7 m/s |
  | Leg length | 12–334 m |
  | Legs per mission | 2–5 |

  The demo and scheduler should show these warnings.

## Plausibility: missions unlike any in the data

`results/model_b/example_missions.csv` holds deliveries that drop the payload half-way, which no recorded flight does:
- **300 m delivery with 500 g:** about 20.4 Wh. At 600 m it's about 31.2 Wh: the extra 600 m flown costs about 11 Wh, matching ~510 W × 75 s.
- **Slow (4 m/s) empty delivery:** about 25.7 Wh, more than the same delivery at 8 m/s, because time aloft dominates.

## Choices for later phases

- **The main Model B is physics-first** (`models/model_b/model_b.pkl`).
- **Phase 5 (uncertainty):** mission uncertainty will come from the parts' out-of-fold errors, resampled jointly per flight in case they're correlated. The remaining under-prediction at 12 m/s and 500 g (about −0.25 Wh) should be covered by the calibrated interval, and coverage should be checked per speed, not just overall.

## Verification

- `pytest`: **74 passed**; `pyflakes` clean.
- **Tests added in this phase:**
  - momentum-theory curve shape
  - the mass-bounded fit
  - kinematic acceleration recovery
  - the linear work model
  - planning-time inputs only
  - folds sharing no flight or battery
  - the batch path matching the single-mission path
  - out-of-range inputs flagged and never negative
  - the saved main model loading and behaving sensibly (further costs more, heavier costs more, higher costs more)
- **Reproducible:** repeated runs give identical results.
