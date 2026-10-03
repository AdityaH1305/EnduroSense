# Phase 6: Feasibility decisions, what-if evaluation and fleet simulation

**Status:** complete (2026-10-03); revised the same day by verification pass 4 (multi-sortie missions, confidence intervals, independence check).
**Reproduce:** `python scripts/08_decisions.py` (about 3 minutes; development data only; needs the held-out distributions from `scripts/07_uncertainty.py`). Two runs give byte-identical outputs.
**Outputs:** `results/decisions/` (tables, `figures/tradeoff.png`, `figures/probability_reliability.png`, `figures/fleet_simulation.png`)

## The decision rule

`feasibility.py` turns the two calibrated distributions into one number:

    P(success) = P(energy available >= energy required)

- "Energy available" already excludes the reserve, so success means the mission ends with the reserve intact.
- It's computed on a fixed grid of 400 quantiles of the battery's distribution, not by random sampling, so the same inputs always give the same probability. It agrees with the exact answer for normal distributions (a test) and with brute-force sampling on 300 real pairs (largest difference 0.0025).
- **Rule:** GO if P(success) ≥ τ (0.95 by default, in `config.yaml`). If the battery model abstains (no pre-flight voltage reading), the answer is always NO-GO.

### Are the two models' errors independent?

The formula assumes they are. This was checked on the 120 development flights that have both a pre-flight battery prediction and a mission prediction for the flight that battery then flew (`error_independence.json`).

- **There is a small positive link.** The correlation of the two errors is +0.16 (95% interval −0.07 to +0.40); the rank correlation is +0.27 and clearly not zero (p = 0.003). When the battery model over-estimates, the mission model tends to over-estimate too.
- **It barely matters, because the mission model's error is five times smaller** (0.72 Wh against 3.82 Wh). The real spread of the margin error is 3.78 Wh; independence predicts 3.89 Wh.
- **The direction is the safe one.** On these 120 real pairs the 90% range of the predicted margin contains the true margin 92.5% of the time, and the truth falls below the range 1.7% of the time (5% would be nominal).

## How decisions are evaluated without flying anything

`whatif.py` asks: "would *this* battery, in *this* state, have managed *that* recorded mission?" Both halves are real measurements, so the answer is known.

- **Battery states:** 2,350 real held-out readings (one every 10 s) from the 46 supported development chains, each with its true energy above the reserve. 121 of them are **pre-flight states**: the last motors-off reading before a take-off, which is where a real go/no-go is made.
- **Missions:** 331 in total.
  - 153 are real single flights.
  - 178 are **multi-sortie missions**: every pair (128) and every triple (50) of flights that really were flown on the same battery. Their true energy is what those flights measured (about 42 Wh and 60 Wh on average).
- **Truth:** the mission would have succeeded if true energy available ≥ true energy required. A mission's energy and a battery's energy drop are the same measured quantity (checked: they agree to 0.02 Wh per flight).
- **Everything is out-of-fold.** The battery model never saw that battery, the mission model never saw that flight, and both distributions are calibrated on other folds only.

### Multi-sortie missions share one error

Flights on one battery are flown the same day in the same conditions, and the mission model's errors for them move together: about 46% of the error variance is shared within a battery.

| How the sorties' errors are combined | 90% range covers, real same-battery pairs | triples |
|---|---|---|
| Drawn independently for each sortie | 85% | 81% |
| **One shared error set for all sorties (used)** | **94.5%** | **98%** |

Independent draws are over-confident, so `replay_compound` applies the same held-out error set to every sortie. That errs on the cautious side (the truth lies between the two).

### Three sets of pairs

| Set | Pairs | Would succeed |
|---|---|---|
| All pairs (any battery state × any mission) | 20,000 | 42.0% |
| Borderline pairs (true margin within ±5 Wh) | 20,000 | 49.0% |
| Pre-flight states × any mission | 20,000 | 57.3% |

### The four policies

| | Policy | Approves when |
|---|---|---|
| P1 | Minutes left (the brief) | predicted minutes left ≥ mission duration |
| P2 | Energy, point estimates | predicted energy available ≥ predicted energy required |
| P3 | **EnduroSense** | P(success) ≥ τ |
| P4 | Oracle | the true margin is positive (the best possible) |

P1 is given the mission's *true* airborne duration, which favours it. P1 and P2 are also evaluated with a safety margin added, so every policy gives a curve of unsafe approvals against wasted refusals, not just one point.

Two kinds of mistake:
- **Unsafe approval:** approving a mission that would have cut into the reserve (as a share of missions that would fail).
- **Wasted refusal:** refusing a mission that would have succeeded (as a share of missions that would succeed).

## Results

### Each policy as specified (`operating_points.csv`)

| Policy | All pairs: unsafe approvals | wasted refusals | Pre-flight: unsafe approvals | wasted refusals |
|---|---|---|---|---|
| P1 minutes left ≥ duration | **10.5%** | 3.6% | 11.4% | 1.8% |
| P2 energy margin ≥ 0 | 2.9% | 4.8% | 5.1% | 3.2% |
| **P3 EnduroSense, τ = 0.95** | **0.25%** | 22.2% | **0.04%** | 18.6% |
| P4 oracle | 0% | 0% | 0% | 0% |

- **The brief's rule approves 1 in 10 missions that would fail.** Of everything it approves, 13% is unsafe.
- **Point estimates with no margin** approve 2.9% of failing missions. A point estimate is wrong about half the time in each direction, so "margin ≥ 0" is a coin flip for close calls.
- **EnduroSense at τ = 0.95** approved 29 failing missions out of about 11,600 (0.25%); of everything it approved, 0.44% was unsafe. Their median shortfall was 1.8 Wh into the reserve and the worst was 4.2 Wh (`mistakes_at_tau.json`).
- **The price is caution.** It refuses 22% of missions that would have succeeded. Those are close calls: their median true margin was 3.6 Wh and 89% had less than 10 Wh to spare. This comes straight from Model A's uncertainty (a 90% range about 11 Wh wide).

### Like-for-like: every policy at the same unsafe-approval rate

Comparing the points above isn't fair, because P1 and P2 could also be made safer by adding a margin. So each is given the margin that matches EnduroSense's unsafe-approval rate, tuned with hindsight on these same pairs.

| At the unsafe rate of τ = 0.95 | All pairs: wasted refusals | Margin needed | Pre-flight: wasted refusals | Margin needed |
|---|---|---|---|---|
| P1 + margin | 91% | 3.7× the duration | 36% | 1.6× the duration |
| P2 + margin | 23% | 7.0 Wh | 29% | 11.4 Wh |
| **P3 EnduroSense** | **22%** | none (τ) | **19%** | none (τ) |

Ranking quality over the whole curve (`policy_summary.csv`, `paired_gain.csv`). The 95% intervals resample battery chains on both sides of a pair (the battery and the mission's flights), with both policies on the same resample.

| | AUC, all pairs | AUC, borderline | Approved at 2% unsafe, borderline | Approved at 1% unsafe, pre-flight |
|---|---|---|---|---|
| P1 minutes left | 0.976 | 0.734 | 4.2% | 83.6% |
| P2 energy, point estimates | 0.995 | 0.842 | 9.2% | 84.7% |
| P3 EnduroSense | 0.995 | 0.851 | 17.8% | 89.5% |
| **P3 − P2** | 0.000 (−0.001 to +0.001) | +0.009 (0.000 to +0.016) | +8.6 points (−0.9 to +14.1) | **+4.8 points (+1.6 to +9.9)** |
| **P3 − P1** | +0.019 (+0.015 to +0.022) | +0.116 (+0.089 to +0.144) | +13.7 points (+3.0 to +31.2) | +5.9 points (+2.8 to +9.3) |

**What this says, stated plainly:**

1. **Energy beats minutes, clearly.** Every P3 − P1 difference is positive in at least 99.6% of resamples. Over all pairs at a 1% unsafe rate, EnduroSense approves 88% of feasible missions and minutes-left approves 44% (+44 points, 33 to 55). Minutes-left fails mid-flight because the power being drawn *now* (hovering, descending) says little about the power the next mission needs. At pre-flight states, where a typical flying power is used, minutes-left is much closer (AUC 0.990 against 0.995), though still behind at every level.
2. **Against point estimates with a well-tuned margin, the probability is modestly better, and only in the places it should be.**
   - Over all pairs the two curves are the same within noise.
   - At pre-flight states, where real decisions are made, the gain is clear at strict safety levels: +4.8 points at 1% unsafe and +2.2 points at 2% unsafe (+0.3 to +5.2).
   - On borderline cases the gain is about 8 points, positive in 95% of resamples, with an interval that just includes zero.
   - The pattern is the same for six different random draws of pairs; the uncertainty comes from the limited number of batteries, not from sampling.
3. **The main practical advantage of P(success) is that its threshold means something and transfers.** A fixed margin has to be found by trial: it was 7.0 Wh over all pairs but 11.4 Wh at pre-flight states, and it was tuned here with knowledge of the outcomes. τ = 0.95 needed no tuning.

### Is P(success) honest? (`probability_reliability.csv`)

| Predicted P(success) | Pairs | Actually succeeded | Counting each battery equally |
|---|---|---|---|
| below 5% | 10,094 | 0.3% | 0.3% |
| 5–20% (mean 11%) | 839 | 13% | 12% |
| 20–40% (mean 29%) | 518 | 31% | 26% |
| 40–60% (mean 50%) | 396 | 54% | 51% |
| 60–80% (mean 71%) | 557 | 76% | 72% |
| 80–90% (mean 86%) | 539 | 89% | 86% |
| 90–95% (mean 93%) | 490 | 93% | 91% |
| 95–99% (mean 97.4%) | 851 | 97.6% | 96.8% |
| above 99% (mean 99.9%) | 5,716 | 99.8% | 99.7% |

The probability tracks reality across the whole range. In the band that matters for the decision, 95–99%, missions succeeded 97.6% of the time against 97.4% predicted.

### Choosing τ (`tau_table.csv`)

| τ | Unsafe approvals | Wasted refusals | Pre-flight: unsafe | Pre-flight: wasted |
|---|---|---|---|---|
| 0.50 | 2.96% | 4.8% | 5.16% | 3.2% |
| 0.80 | 1.09% | 11.2% | 1.60% | 8.8% |
| 0.90 | 0.56% | 16.8% | 0.36% | 13.5% |
| **0.95** | **0.25%** | **22.2%** | **0.04%** | **18.6%** |
| 0.99 | 0.08% | 32.1% | 0.00% | 26.7% |

τ = 0.95 stays the default: going from 0.90 to 0.95 halves unsafe approvals (and at pre-flight states cuts them nine-fold) for about 5 more points of refusals; going on to 0.99 costs another 10 points. The value is still pending the guide's confirmation and lives in `config.yaml`. At pre-flight states the unsafe counts at τ ≥ 0.95 are single digits, so those percentages are rough.

## Fleet simulation (`scheduler.py`, `fleet_simulation.csv`)

**Setup:** 4 drones, 40 tasks a day (70% single flights, 30% two-sortie missions), 200 simulated days; every policy sees the same tasks and the same battery draws.

**What is real and what is simulated:** each battery is one of the 34 real development chains that start full, starting at its pre-flight reading. Its true energy and the telemetry the battery model sees come from that chain, and each task's true cost is the measured energy of real flights. Only the *pairing* of batteries with tasks is simulated.

**One day:** each task goes to the approved drone with the least predicted energy (fuller batteries are kept for bigger tasks). If no drone is approved, the emptiest one swaps its battery and is asked again; if it's still refused the task is dropped. A mission that draws more than the battery truly had above the reserve is an **unsafe mission**.

| Policy | Unsafe missions per 100 flown | Completed per day (of 40) | Battery swaps per day | Energy left at swap |
|---|---|---|---|---|
| P1 minutes left (the brief) | **11.1** | 40.0 | 14.9 | 4.4 Wh |
| P2 energy, point estimates | 4.2 | 40.0 | 15.7 | 6.0 Wh |
| P1 + margin (matched to P3) | 0.1 | 12.1 | 28.0 | 56.9 Wh |
| P2 + 7.0 Wh margin (matched to P3) | 0.30 | 39.9 | 17.9 | 12.8 Wh |
| **P3 EnduroSense, τ = 0.95** | **0.24** | 39.6 | **16.8** | 10.7 Wh |
| P4 oracle | 0.0 | 39.9 | 15.6 | 5.9 Wh |

- **Against the brief's rule: 46 times fewer unsafe missions** (0.24 against 11.1 per 100) for 13% more battery swaps and 0.4 tasks a day dropped.
- **Against point estimates with no margin: 18 times fewer unsafe missions** for 7% more swaps.
- **Against point estimates with a tuned margin:** the same safety within noise (0.10 ± 0.04 against 0.12 ± 0.05 unsafe missions a day), with **6% fewer battery swaps** (16.8 ± 0.2 against 17.9 ± 0.2) and 2 Wh less left unused in each battery. It drops slightly more tasks (0.44 against 0.09 a day): two-sortie tasks that a fresh battery can't be 95% sure of.
- **Minutes-left can't be rescued with a margin.** Matching EnduroSense's safety needs a margin so large that the fleet completes 12 of 40 tasks.
- **The gap to the oracle** (1.3 swaps a day, 5 Wh per battery) is the cost of Model A's remaining uncertainty.
- The ± values are over simulated days, for this fixed set of batteries and tasks.

## Artefacts found and fixed

1. **Fresh batteries were scored too early.** The first run of the simulation showed EnduroSense dropping 3.6 tasks a day. A fresh battery was scored at the **first reading of its recording**, when the battery model has seen a single sample and its 90% range is 24–45 Wh wide. A real decision is made at the pre-flight reading about 16 s later, where the range is about 19 Wh. `batteries_from()` now starts each battery at its pre-flight reading, for every policy.
2. **Multi-sortie missions assumed independent errors (verification pass 4).** They were first built from random flights with an independently drawn error per sortie. Real back-to-back sorties share conditions and their errors move together, so those ranges were over-confident (81–85% coverage). Multi-sortie missions are now real same-battery flights with one shared error set. The headline conclusions did not change; the numbers above are the corrected ones.
3. **Confidence intervals only resampled the battery side (verification pass 4).** They now resample chains on the mission side too. This widened the intervals and turned the borderline gain over point estimates from "significant" into "likely".

A related, real finding: **a fresh battery is the hardest state to judge.** At the first flight's pre-flight reading the range is about 19 Wh wide and the mean error is 4.2 Wh, against 11 Wh and 2.3 Wh overall. The discharge curve is flat near full charge and each battery's capacity is slightly different.

## Limitations

- **What-if pairs are not flights.** A battery state is paired with a mission flown on another day in other conditions. Battery temperature and ageing effects that would link the two are not captured, beyond the real-pair check above.
- **Multi-sortie missions ignore the ground time between sorties** in the real recordings; their energy is the sum of the flights. Their ranges are deliberately cautious (94–98% coverage for a nominal 90%).
- **In the fleet simulation,** after a mission the battery's telemetry is taken from the chain reading with the same energy drawn, which is usually a mid-flight reading. Phase 5 showed the battery model is equally well calibrated in flight and on the ground, so this doesn't favour a policy. About half of the recordings stop a few Wh before the reserve, so even the oracle leaves some energy unused.
- **The matched margins for P1 and P2 are tuned on the same pairs they're scored on,** which flatters them slightly. EnduroSense's τ is not tuned.
- **Very low rates rest on few events.** 0.25% is 29 pairs from a handful of batteries. Phase 7 repeats everything on the locked test set.
- **All of this is development data.** No test data was used.

## Changes from the plan

- P(success) uses a deterministic quantile grid in place of 10,000 random draws (same answer, repeatable, faster). No Gaussian closed form was needed.
- Multi-sortie missions are real same-battery pairs and triples, not random groups of 2–4 flights.
- The borderline set is reported on its own, not mixed in at 30%.
- Pre-flight states were added as a separate set, because that is where real decisions are made.
- Model B's selection helpers moved from `scripts/06_model_b.py` into `models/model_b.py`, and the leave-fold-out calibration helpers into `uncertainty/`, so Phases 5 and 6 share one implementation. Re-running Phases 4 and 5 afterwards reproduced every number.
- `scripts/07_uncertainty.py` now also saves each held-out reading's full set of quantiles and each flight's held-out predicted parts, which this phase reads.

## What Phase 7 and Phase 8 get

- `feasibility.p_success(available, required)` and `decide(p, tau)` for single decisions (the demo's mission-check page); `p_success_batch` for tables of them.
- `whatif.build_missions`, `policy_scores`, `summary_table`, `paired_gain`, `tradeoff_curve` and `probability_reliability` run unchanged on test-set states and missions.
- `uncertainty.mission.replay_compound` gives the distribution of a multi-sortie mission (shared error set).
- `scheduler.FleetSimulator` runs on any set of chains.
- `results/decisions/whatif_pairs.parquet` holds every pair with all four policy scores: the source for the demo's "minutes-left says go, EnduroSense says no" examples (development data; Phase 7 produces the test-set version).

## Verification

- `pytest`: **109 passed**; `pyflakes` clean.
- **Independent recomputation** of every headline number from the saved files with separate code (pass 4).
- **Mutation testing:** 33 deliberate one-line bugs in the Phase 5 and 6 code; every one is caught by a test.
- **Tests (`tests/test_decisions.py`):**
  - P(success) matches the closed form for normal distributions; batch and single versions agree
  - P(success) falls as the mission grows and rises with more energy; wider uncertainty pulls it towards 50%
  - the decision rule, including NO-GO on abstention
  - the two mistake rates on a hand-counted case
  - a perfect score approves every feasible mission with no unsafe approvals
  - pairs carry the right truth; the pre-flight state is the last reading before take-off; typical power comes from other folds
  - multi-sortie missions contain only flights that shared a battery, and their range is the sum of the single ranges
  - the bootstrap reweights both the battery side and the mission side
  - in a toy world where the models are right, the oracle makes no mistakes and τ = 0.95 approves nothing unsafe
  - the reliability table recovers an honest probability
  - batteries follow their chain and start at the pre-flight reading; best-fit assignment
  - fleet bookkeeping on a case worked out by hand (completed, unsafe, dropped, swaps, energy left)
