# Phase 3: Model A comparison (energy available)

**Status:** complete (2026-10-02).
**Reproduce:** `python scripts/05_model_a.py` (about an hour the first time; results are cached)
**Outputs:** `results/model_a/` (tables, out-of-fold predictions, figures) and `models/model_a/` (refitted on all development data)

## Setup

- **Data:** development data only: 23,884 one-second rows from 49 battery chains. The test set stays locked.
- **Scoring:** grouped 5-fold cross-validation by battery chain, so every prediction is made by a model that never saw that battery. A test checks this.
- **Hyperparameters:**

  | Model | Search |
  |---|---|
  | Linear Regression | 5 regularisation strengths |
  | Random Forest | 10 random configurations |
  | XGBoost | 30 random configurations |
  | LSTM, GRU | 8 configurations each (hidden size, layers, dropout) |

  The best configuration is chosen by mean CV error. The same folds choose and score, so CV numbers are slightly optimistic; the unbiased figure comes from the locked test set in Phase 7.
- **Neural models:** LSTM and GRU read the last 60 s of signals, combine that with the same 25 features the other models get, and use early stopping on 15% of the *training* batteries.
- **Fair baselines:** each knows the battery's charge when it was inserted (see `models/baselines.py`).

## Results

| Model | CV error (Wh) | ± over folds | vs voltage lookup, 95% CI | Folds better | Minutes error | CPU latency | Size |
|---|---|---|---|---|---|---|---|
| **GRU** | **2.61** | 0.37 | **−0.62 [−1.14, −0.15]** | 4/5 | 0.87 min | 5.5 ms | 0.18 MB |
| **LSTM** | **2.65** | 0.53 | **−0.58 [−1.18, −0.02]** | 4/5 | 0.87 min | 1.3 ms | 0.23 MB |
| Random Forest | 3.02 | 0.23 | −0.22 [−0.70, +0.24] | 3/5 | 0.90 min | 6.1 ms | 14.8 MB |
| Linear Regression | 3.02 | 0.61 | −0.21 [−0.77, +0.32] | 2/5 | 0.90 min | 0.8 ms | <0.01 MB |
| XGBoost + physics | 3.04 | 0.43 | −0.19 [−0.67, +0.35] | 3/5 | 0.90 min | 2.3 ms | 0.47 MB |
| *Voltage lookup (non-ML)* | *3.22* | 0.66 | — | — | 0.91 min | 0.4 ms | — |
| XGBoost | 3.26 | 0.28 | +0.03 [−0.54, +0.58] | 3/5 | 0.92 min | 1.2 ms | 0.47 MB |
| *Energy counting (BMS)* | *4.49* | 0.95 | +1.28 [+0.89, +1.68] | 0/5 | 1.02 min | 0.2 ms | — |
| *Fixed capacity (reference)* | *13.56* | 0.44 | — | 0/5 | 1.85 min | — | — |

How to read this table:
- **Confidence intervals** come from resampling whole batteries 2,000 times (`evaluate.paired_comparison`). Negative means better than voltage lookup.
- **Labels carry about 1.5–2.2 Wh of noise** (Phase 2), so that is roughly the best error any model can show.
- **On rows where the battery's pre-flight voltage is known** (96% of rows), errors drop for every model: GRU 2.39, LSTM 2.45, voltage lookup 2.82 Wh (`mae_known_rest_voltage.csv`).

## What this shows

1. **The recurrent models (GRU and LSTM) are the only ones clearly better than a good non-ML method.** Their errors are about 19% lower than voltage lookup, and the confidence intervals exclude zero.
   - **A likely reason:** they see how the voltage *responds* to load over the last minute, which reflects the battery's health.
   - **The tabular models (Random Forest, Linear Regression, XGBoost with or without physics)** sit around 3.0 Wh. That's within noise of the baseline.
2. **Plain XGBoost doesn't beat the baseline.** With only 49 batteries, tree ensembles can latch onto battery-specific patterns that don't carry over to new batteries. Grouped CV exposes this, and a random row-level split would have hidden it.
3. **A fair baseline matters.** Against the naive fixed-capacity method (13.6 Wh) every model looks excellent. Against a method that knows the starting charge, only the sequence models stand out.
4. **The brief's "minutes left" output barely improves with a better battery model.** Every model, from 2.6 to 4.5 Wh of energy error, lands between 0.87 and 1.02 minutes of error. Minutes-left error is dominated by not knowing how hard the drone will fly next, not by the battery estimate. **This is direct evidence for EnduroSense's framing:** predict energy, and handle the mission's demand separately (Model B).

## Issues to carry forward

- **The GRU is slightly optimistic below the reserve.** Where the battery is already below 22.6 V (1,689 rows, 26 chains), it over-predicts by +0.98 Wh on average (voltage lookup: −0.35 Wh). That's the unsafe direction, and Phase 5's calibrated uncertainty and Phase 6's decision rule must cover it.
- **3 chains (3.8% of rows) have no pre-flight resting voltage,** because their recordings started with the motors running. The worst case is flight 68, over-predicted by 11 Wh (voltage lookup by 30 Wh). A real drone always reads its battery before take-off, so this is a recording gap. Phase 5 should add a "pre-flight voltage known" flag so uncertainty widens in these cases.
- **CPU latency:** PyTorch's CPU code for LSTMs is heavily optimised and GRU's isn't (independent timing: 0.6 vs 4.8 ms per prediction). Both are well under 10 ms.

## Choices for later phases

- **The GRU is the main Model A** (most accurate). The LSTM is a close alternative if CPU speed matters.
- **Phase 5:** a deep ensemble of GRUs for uncertainty, plus conformal calibration grouped by battery.

## Verification

- `pytest`: **63 passed**.
- **New tests:** metric maths, the search sampler, network output shapes, imputation, the baselines matching their formulas, and grouped CV predicting every development row once without the model ever seeing that battery.
