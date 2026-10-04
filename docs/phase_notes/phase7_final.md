# Phase 7: Final locked-test evaluation and reproducibility

**Status:** complete (2026-10-04).
**Reproduce:** `python scripts/run_all.py --final` (about 25 minutes on mains power with a GPU; the development steps alone are `python scripts/run_all.py`)
**Outputs:** `results/final/`, the summary in [`docs/results_summary.md`](../results_summary.md), the protocol in [`docs/final_evaluation_protocol.md`](../final_evaluation_protocol.md)

The results themselves are in the results summary. This note records how the evaluation was run and what was checked.

## Order of events

1. **Full rebuild from the raw CSV** (`scripts/run_all.py`, steps 01–08): 21.7 minutes. Compared with the results stored before the rebuild:
   - Phases 1, 2, 4, 5 and 6: **45 result files byte-identical.**
   - Phase 3 (Model A comparison): identical to 14 decimal places. The only differences were the latency timings (expected) and last-digit changes in the Random Forest column, because the stored Phase 3 run predated the Random Forest ordering fix of verification pass 4.
2. **Protocol written down** (`docs/final_evaluation_protocol.md`): what would be measured, the four success criteria and how a near-miss would be reported.
3. **Rehearsal.** `scripts/09_final_test.py --rehearsal` ran the whole evaluation on development readings (fold 0) to find coding errors without opening the test set. Its outputs were deleted.
4. **The test set was opened once** (`--final`, 2026-10-04 11:13). Nothing was fitted, tuned or calibrated: the script loads the models built in step 1.
5. **Reproducibility of the final step:** a second `--final` run gave 20 byte-identical output files and identical console output, and added nothing to the access log.

## How the test set is protected

- `select(df, "test")` still raises unless `final=True`; only `scripts/09_final_test.py --final` passes it.
- **Every opening is logged** (`results/final/test_access_log.json`) with a fingerprint of all source files, scripts, configuration and the split (`endurosense/access.py`).
  - Same fingerprint: a reproducibility rerun; nothing is added.
  - Changed fingerprint: the script refuses to run unless `--reason "..."` is given, and the reason is logged.
- **Consequence for Phase 8:** changing anything under `src/` or `scripts/` and re-running the final evaluation will need a stated reason. The demo should live in `app/` and only read saved models and results.

## What the final script does (`scripts/09_final_test.py`)

| Part | What is scored | Against |
|---|---|---|
| Model A, point | the nine saved models from `05_model_a.py`, plus the calibrated ensemble's median | test readings; voltage lookup as the paired reference, intervals from resampling test chains |
| Model A, ranges | the saved calibrated ensemble | coverage, width, one-sided misses, per-chain coverage |
| Model B, point | the seven saved variants from `06_model_b.py` | test flights, split by route seen / unseen in development |
| Model B, ranges | the saved calibrated main model; the development error sets are replayed on test flights | coverage per route and for multi-sortie missions |
| Decisions | Phase 6 functions unchanged, on test battery states and test missions | the comparison policies' margins fixed from development data (`results/decisions/matched_margins.json`) |

Shared code: the operating-point, margin, mistake and real-pair functions moved from `scripts/08_decisions.py` into `whatif.py`, the fleet comparison into `scheduler.py` and the figures into `decision_plots.py`, so development and test are evaluated by the same functions. `08_decisions.py` gives byte-identical outputs after the move.

## Results in brief

| # | Criterion | Result |
|---|---|---|
| 1 | Model A beats both non-ML baselines | **Met:** 2.19 Wh against 2.90 and 4.21 Wh |
| 2 | Model B within 5%, including unseen routes | **Not met:** 3.1% overall, 5.7% on the 7 unseen-route flights (all over-predicted) |
| 3 | 90% ranges cover 88–92% | **Not met, within sampling noise:** Model A 93.8% (83–100%), Model B 80.0% (65–92.5%) |
| 4 | EnduroSense decides better than minutes-left and point estimates | **Met:** unsafe approvals 0.61% against 12.4% and 3.6% |

Development against test, the figures that matter most:

| | Development (held out) | Test |
|---|---|---|
| Model A error, main model | 2.37 Wh | 2.19 Wh |
| Model A 90% range covers | 89.5% | 93.8% |
| Model B error, seen route | 2.35% | 2.61% |
| Model B 90% range covers | 91.5% | 80.0% |
| Unsafe approvals: minutes-left / point estimates / EnduroSense | 10.5% / 2.9% / 0.25% | 12.4% / 3.6% / 0.61% |
| EnduroSense unsafe approvals at pre-flight states | 0.04% | 0.00% |
| Fleet, unsafe missions per 100: minutes-left / EnduroSense | 11.1 / 0.24 | 12.0 / 0.73 |
| P(success) 95–99% band, actually succeeded | 97.6% | 94.2% |

## What the test set taught us

1. **The ensemble was the right call.** The single GRU that won cross-validation was no better than the baseline on test (2.81 Wh); the 5-network ensemble was the best model (2.19 Wh).
2. **Risk is per battery, as predicted.** 9 of 11 test batteries were inside their range 100% of the time. Two were over-estimated, and all 71 of EnduroSense's unsafe approvals involve those two.
3. **Model B's weak spots were the ones already flagged.** The 100 m altitude under-prediction (Phase 5) accounts for 4 of the 6 seen-route misses, and distance extrapolation (Phase 4) for the unseen-route error.
4. **Model B's ranges are too narrow on new data** (80% against 90%). A cautious user should widen them; the multi-sortie ranges, which are deliberately cautious, did cover 91%.
5. **Energy with a calibrated margin is what matters; the probability form adds little that can be measured here.** Point estimates plus a development-tuned margin matched EnduroSense over all pairs. The probability refused fewer feasible missions at pre-flight states (25% against 32%) and needed no tuning.
6. **The link between the two models' errors is stronger than development suggested** (+0.43 against +0.16), still in the safe direction.

## Not done, by design

- **No retuning after seeing the test results.** The weaknesses above are reported, not patched. Fixing them (for example widening Model B's ranges or adding an altitude term) and re-scoring on the same test set would make the test results optimistic.
- **Latency on other hardware** (for example a Raspberry Pi) remains a stretch goal. The laptop figures were re-measured on mains power during the rebuild: every model predicts one reading in about 5 ms or less (GRU 4.5 ms, LSTM 1.0 ms, Random Forest 5.1 ms).

## Verification

- `pytest`: **115 passed**; `pyflakes` clean.
- **New tests:**
  - every opening of the test set is logged, an unchanged rerun adds nothing, and a changed rerun needs a reason
  - missions for unseen flights use the given (development) error sets and share one error set across sorties
  - operating points with fixed margins, and the summary of mistakes
  - the real-pair check reports an honest margin when the models are right
  - the task pool and the policy comparison table
  - every saved Model A model loads from disk and predicts sensibly
- **Reproducibility:** full rebuild identical to the stored results (above); final evaluation identical on a second run.
