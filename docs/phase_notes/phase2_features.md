# Phase 2: Labels and features

**Status:** complete (2026-10-02).
**Reproduce:** `python scripts/04_build_features.py`
**Outputs:** `data/features/`, plus `results/phase2/` (summary, chain labels, figures)

## Model A: energy available (`features/model_a.py`)

### Labels

- **The question:** how many Wh can the battery still deliver before its rest voltage reaches the 22.6 V reserve? Each battery chain has rest-voltage points at known energy drawn, and the reserve crossing is located between them.
- **The discharge curve isn't straight.** It flattens towards the reserve: the battery loses 0.054 V per Wh near full but 0.029 V per Wh at 22.6 V. Labels therefore interpolate and extrapolate along a fitted curve, slope(v) = a + b·v, clipped to the measured range. The curve is fitted on development chains only.
- **Label accuracy,** checked on development chains by leave-one-out:

  | Method | Bias | Spread | Notes |
  |---|---|---|---|
  | Straight-line interpolation | +3.3 Wh | — | rejected |
  | Curve interpolation | −0.04 Wh | 1.5 Wh | used |
  | Extrapolating up to 0.6 V | −0.6 Wh | 2.2 Wh | 18 cases |

  Labels themselves therefore carry about **1.5–2.2 Wh** of noise. No model can be judged more accurate than that.
- **Coverage:** 60 of the 90 chains can be labelled (31 measured, 29 extrapolated); 49 of them are in development (26 measured, 23 extrapolated by 5.1 Wh on average, 9.6 Wh at most).
- **Rows:** 29,592, one per second of flight, from 157 flights (23,884 rows in development). In development, `remaining_wh` ranges from −16 to 77 Wh (negative after the reserve is passed).
- **The brief's output** is also built as a label: `remaining_min`, the motors-on minutes until the reserve, following the battery's actual future use.
- **Finding:** in development, the 35 batteries that start full deliver **64.7 ± 7.3 Wh** before the reserve (range 47–77 Wh). This variation between batteries is what Model A has to learn.
- **Rule:** every statistic and figure about label values in these notes and in `results/phase2/` describes development data only. The test set's labels exist in `data/features/` for Phase 7 but are never summarised (fixed in the 2026-10-02 verification pass).

### Features: 25, all causal

| Group | Features |
|---|---|
| Instantaneous | voltage, current, power |
| Look-back windows (10 s and 30 s) | mean, spread and trend of voltage and current; mean power |
| Battery health | internal resistance (voltage drop per amp over 30 s); voltage sag below the flight's rest voltage |
| Charge state | rest voltage before this flight; rest voltage when the chain started |
| Counting | energy since the flight / chain started; time since chain start (wall clock and motors-on) |
| Context | flight index in the chain; motors on/off |

Causality was fixed and tested during this phase:
- **Pre-flight rest voltage** is a *running* value during the idle before take-off. The first version used readings a few seconds ahead.
- **Sequence windows** use completed half-second bins only. The first version included a bin that could hold readings up to 0.5 s ahead.
- **Phase labels are excluded as features,** because their smoothing looks 1 s ahead. They're kept for error analysis only.
- **`test_model_a_features_are_causal_on_real_data`** perturbs every reading after time t in a real flight and asserts that no earlier feature changes.

**Sequence data:** `model_a_series.parquet` holds 5 channels at 2 Hz. `sequence_windows()` builds 120-step (60 s) inputs with a padding mask.

### Leakage check (throwaway, development data only)

| Approach | Average error |
|---|---|
| Plain linear model, all features | 3.0 Wh |
| Naive "fixed capacity − energy used" | 13.6 Wh |
| Label spread (for comparison) | 20 Wh |

The features are informative but don't leak the answer. **For Phase 3:** the naive baseline is unfair, because chains start at different charge levels. A fair energy-counting baseline must also use the rest voltage when the battery was inserted.

## Model B: energy required (`features/model_b.py`, `mission.py`)

- **Every route is a closed loop.** R1 has 3 straight legs with short hovers between them; it ends about 2 m from take-off and goes about 140 m from home. So Model B is built **per leg**: 587 legs from 193 usable flights, with leg lengths of 119, 140 and 198 m on R1. That gives real distance variation.
- **Usable flights** exclude the truncated landing (flight 250) and the 2 flights that change altitude during cruise.
- **Two of the 587 legs are repositioning** (flown faster than commanded, e.g. flight 2's first leg). They are flagged `at_commanded_speed = False` and skipped by leg-time models.
- **The parts add up exactly:** climb + legs + hover + other + descent + ground equals each flight's total to within 10⁻¹⁴ Wh (`other` is cruise fragments under 3 s, about 0.002 Wh).
- **What the data shows:**
  - Extra time per leg grows with speed, from 0.5 s at 4 m/s to 4.4 s at 12 m/s (acceleration and braking).
  - Leg power depends on payload (469 → 514 → 560 W for 0, 250 and 500 g) and barely on speed, except at 12 m/s.
  - Climbing costs about 0.6 Wh per 10 m and descending about 0.85 Wh per 10 m. Hovering costs about 0.4 Wh per leg.
- **`MissionSpec`** describes any mission: a list of legs with a payload per leg, speed, altitude and wind. `MissionSpec.delivery(...)` gives an out-and-back with the payload dropped. `mission_energy(spec, components)` assembles the total from pluggable component models, which are trained in Phase 4.
- **Planning-time features only:** commanded speed, payload, cruise altitude, ambient wind and leg distance.

## Decisions (and why)

| Decision | Reason |
|---|---|
| Curve-aware label interpolation | Removes a +3.3 Wh bias from straight-line interpolation (measured). |
| Discharge curve and recovery fitted on development data only | No test data shapes any label. |
| Negative `remaining_wh` is kept | It's true information (below the reserve), and the feasibility layer needs to see it. |
| Model B is built per leg | Every recorded flight flies the same loop, so per-leg data is the only way to learn how distance affects energy. |
| Model B won't use LSTM/GRU | Its inputs are mission settings, not a time series. The brief's five-algorithm comparison is done on Model A. |

## Verification

- `pytest`: **57 passed**.
- **New Model A tests:** curve maths, measured and extrapolated crossings, the margin limit, the running rest voltage, rolling features that never look ahead, completed-bin sequence windows, causality on real data, and labels falling exactly 1:1 with energy drawn.
- **New Model B tests:** the delivery spec, input validation, assembly arithmetic, longer missions needing more energy, the exact breakdown, and physical legs with repositioning legs flagged.
