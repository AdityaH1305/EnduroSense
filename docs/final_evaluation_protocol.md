# Final evaluation protocol

**Written 2026-10-04, before the test set was opened.** This file fixes what the final evaluation measures and what counts as success, so neither can be adjusted after seeing the results.

## What is frozen

- **The test set:** 16 battery chains (40 cruise flights, routes R1, R2, R3 and R6), chosen in Phase 1 before any modelling and never read since (`data/splits/split_v2.json`, hash-protected). Routes R2, R3 and R6 do not appear in the development data; R6 is the longest route.
- **The models:** exactly the files produced by `python scripts/run_all.py` from the development data. Nothing is refitted, re-tuned or re-calibrated on test data.
  - Model A comparison: the nine saved models in `models/model_a/` (three baselines, five algorithms, XGBoost + physics).
  - Model A main model: the calibrated 5-GRU ensemble, `model_a_calibrated.pt`.
  - Model B: the saved variants in `models/model_b/`; the main model is Physics-first, calibrated (`model_b_calibrated.pkl`).
- **The settings:** reserve 22.6 V, τ = 0.95, 90% headline range, as in `config.yaml`. The guide has not yet confirmed these; they are the project's defaults.
- **The comparison margins:** the safety margins that give the minutes-left and point-estimate policies EnduroSense's unsafe-approval rate were found on development data (`results/decisions/matched_margins.json`) and are applied to test data unchanged.

## What is measured (`scripts/09_final_test.py --final`)

1. **Model A, point accuracy.** Error (Wh) of all nine models on the test readings, next to their cross-validation error; each model against the voltage-lookup baseline with a 95% interval from resampling test chains.
2. **Model A, ranges.** Coverage and width of the calibrated ranges on test readings with a pre-flight voltage; how often the truth is below the range; how many readings the model abstains on.
3. **Model B, point accuracy.** Mission energy error for every variant on test flights, split into the route seen in development (R1) and the unseen routes (R2, R3, R6).
4. **Model B, ranges.** Coverage and width of the calibrated mission ranges, also per route.
5. **Decisions.** The Phase 6 what-if evaluation on test battery states and test missions: the four policies as specified, the two margin policies with their development-tuned margins, the honesty of P(success), and the fleet simulation on test batteries.

## Success criteria (from the implementation plan; the plan's "about 5%" is fixed here at 5%)

| # | Criterion | Met if |
|---|---|---|
| 1 | Model A beats both non-ML baselines | The main model's test error is lower than the voltage-lookup and energy-counting baselines on the same readings |
| 2 | Model B is accurate, including on unseen routes | Mean percentage error on test flights is 5% or less, overall and on the unseen routes taken together |
| 3 | The 90% ranges are honest | Test coverage of both models' 90% ranges is between 88% and 92% |
| 4 | EnduroSense decides better than the alternatives | Its ranking quality (AUC over all pairs) is at least that of minutes-left and of point estimates (a difference under 0.002 counts as equal), and at τ = 0.95 it approves fewer failing missions than either as specified |

## How the results will be read

- **The test set is small.** 16 chains is about a quarter of the development data. Coverage measured on 16 chains can easily land 5 to 10 points from its true value, because errors are shared within a battery. Criterion 3 is therefore reported two ways: against the fixed 88–92% band above, and by whether 90% lies inside the 95% interval from resampling test chains. A miss on the fixed band with 90% inside the interval will be reported as "not met, within sampling noise", not as a pass.
- **Whatever the numbers are, they are reported.** A criterion that is not met is stated as not met, with the most likely reason.
- **No changes after opening.** If a bug is found after the test set is opened, the fix and the reason are recorded in `results/final/test_access_log.json` (the script refuses to rerun on changed code without a stated reason) and in the verification log. The models and settings are not tuned on test results.

## Rehearsal

The script has a `--rehearsal` mode that runs the same code on development readings (fold 0), to find coding errors without opening the test set. Its numbers are meaningless (the models were trained on those readings) and are not kept.
