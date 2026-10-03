# Phase 6: Feasibility decisions, what-if evaluation and fleet simulation

**Status:** complete (2026-10-03).
**Reproduce:** `python scripts/08_decisions.py` (about 3 minutes; development data only; needs the held-out distributions from `scripts/07_uncertainty.py`)
**Outputs:** `results/decisions/` (tables, `figures/tradeoff.png`, `figures/probability_reliability.png`, `figures/fleet_simulation.png`)

## The decision rule

`feasibility.py` turns the two calibrated distributions into one number:

    P(success) = P(energy available >= energy required)

- "Energy available" already excludes the reserve, so success means the mission ends with the reserve intact.
- It's computed on a fixed grid of 400 quantiles of the battery's distribution, not by random sampling, so the same inputs always give the same probability. A test checks it against the exact answer for normal distributions (within 0.01).
- **Rule:** GO if P(success) ≥ τ (0.95 by default, in `config.yaml`). If the battery model abstains (no pre-flight voltage reading), the answer is always NO-GO.

### Are the two models' errors independent?

The formula assumes they are. Checked on the 120 development flights that have both a pre-flight battery prediction and a mission prediction (`error_independence.json`): the correlation between the two errors is **+0.16, with a 95% interval of −0.08 to +0.40**. That is not distinguishable from zero. If it's real, a positive correlation makes the true margin *less* uncertain than assumed, so treating the errors as independent errs on the cautious side.

## How decisions are evaluated without flying anything

`whatif.py` asks: "would *this* battery, in *this* state, have managed *that* recorded mission?" Both halves are real measurements, so the answer is known.

- **Battery states:** 2,350 real held-out readings (one every 10 s) from the 46 supported development chains, each with its true energy above the reserve. 121 of them are **pre-flight states**: the last motors-off reading before a take-off, which is where a real go/no-go is made.
- **Missions:** 603 in total. 153 are real single flights; 450 are **multi-sortie missions** of 2, 3 or 4 recorded flights flown one after another on one battery. The true energy is what those flights measured.
- **Truth:** the mission would have succeeded if true energy available ≥ true energy required.
- **Everything is out-of-fold.** The battery model never saw that battery, the mission model never saw that flight, and both distributions are calibrated on other folds only.

Three sets of pairs are scored:

| Set | Pairs | Would succeed |
|---|---|---|
| All pairs (any battery state × any mission) | 20,000 | 24.7% |
| Borderline pairs (true margin within ±5 Wh) | 15,569 | 47.6% |
| Pre-flight states × any mission | 20,000 | 35.5% |

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
| P1 minutes left ≥ duration | **7.6%** | 4.4% | 6.3% | 2.4% |
| P2 energy margin ≥ 0 | 1.5% | 5.9% | 2.4% | 5.1% |
| **P3 EnduroSense, τ = 0.95** | **0.05%** | 27.5% | **0.02%** | 24.1% |
| P4 oracle | 0% | 0% | 0% | 0% |

- **The brief's rule approves 1 in 13 missions that would fail.** Of everything it approves, 20% is unsafe.
- **Point estimates with no margin** are five times safer than that, but still approve 1.5% of failing missions. A point estimate is wrong about half the time in each direction, so "margin ≥ 0" is a coin flip for close calls.
- **EnduroSense at τ = 0.95** approved 8 failing missions out of about 15,000 (0.05%). Their median shortfall was 1.4 Wh into the reserve and the worst was 3.9 Wh (`mistakes_at_tau.json`).
- **The price is caution.** It refuses 27.5% of missions that would have succeeded. Those are close calls: their median true margin was 4.1 Wh and 87% had less than 10 Wh to spare. This comes straight from Model A's uncertainty (a 90% range about 11 Wh wide).

### Like-for-like: every policy at the same unsafe-approval rate

Comparing the points above isn't fair, because P1 and P2 could also be made safer by adding a margin. So each is given the margin that matches EnduroSense's unsafe-approval rate, tuned with hindsight on these same pairs.

| At the unsafe rate of τ = 0.95 | All pairs: wasted refusals | Margin needed | Pre-flight: wasted refusals | Margin needed |
|---|---|---|---|---|
| P1 + margin | 94% | 4.3× the duration | 43% | 1.6× the duration |
| P2 + margin | 30% | 7.9 Wh | 35% | 12.1 Wh |
| **P3 EnduroSense** | **28%** | none (τ) | **24%** | none (τ) |

Ranking quality over the whole curve (`policy_summary.csv`, `paired_gain.csv`; 95% intervals resample battery chains, both policies on the same resample):

| | AUC, all pairs | AUC, borderline | Feasible missions approved at 2% unsafe, borderline |
|---|---|---|---|
| P1 minutes left | 0.978 | 0.722 | 3.9% |
| P2 energy, point estimates | 0.997 | 0.838 | 9.6% |
| P3 EnduroSense | 0.997 | 0.848 | 18.3% |
| **P3 − P2** | 0.000 (−0.000 to 0.000) | +0.010 (−0.002 to +0.017) | **+8.7 points (+0.8 to +13.7)** |
| **P3 − P1** | +0.019 (0.016 to 0.021) | +0.126 (0.097 to 0.155) | +14.5 points (+6.7 to +29.9) |

**What this says, stated plainly:**

1. **Energy beats minutes, clearly.** Every P3 − P1 difference is positive in 100% of resamples. Over all pairs at a 1% unsafe rate, EnduroSense approves 91% of feasible missions and minutes-left approves 44% (+47 points, 41 to 54). Minutes-left fails mid-flight because the power being drawn *now* (hovering, descending) says little about the power the next mission needs. At pre-flight states, where a typical flying power is used, minutes-left is much closer (AUC 0.993 against 0.996), though still behind at every level.
2. **Against point estimates with a well-tuned margin, the probability is only modestly better.** Over all pairs the two curves are the same within noise. The probability helps where it should: on borderline cases (+8.7 points at 2% unsafe) and at pre-flight states at the strictest level (+2.2 points at 1% unsafe, 0.3 to 6.1). The other differences have intervals that include zero.
3. **The main practical advantage of P(success) is that its threshold means something and transfers.** A fixed margin has to be found by trial: it was 7.9 Wh over all pairs but 12.1 Wh at pre-flight states, and it was tuned here with knowledge of the outcomes. τ = 0.95 gave 0.05% and 0.02% unsafe approvals in the two settings with no tuning at all.

### Is P(success) honest? (`probability_reliability.csv`)

| Predicted P(success) | Pairs | Actually succeeded | Counting each battery equally |
|---|---|---|---|
| below 5% | 13,984 | 0.1% | 0.1% |
| 5–20% (mean 11%) | 632 | 12% | 11% |
| 20–40% (mean 30%) | 350 | 33% | 29% |
| 40–60% (mean 50%) | 312 | 61% | 57% |
| 60–80% (mean 71%) | 405 | 76% | 69% |
| 80–90% (mean 86%) | 377 | 89% | 86% |
| 90–95% (mean 93%) | 347 | 93% | 92% |
| 95–99% (mean 97%) | 469 | 99% | 99% |
| above 99% | 3,124 | 99.9% | 99.8% |

The probability tracks reality across the whole range, and where it's off it is slightly cautious (missions it calls 50% succeed about 57–61% of the time). In the band that matters for the decision, 95–99%, missions succeeded 99% of the time.

### Choosing τ (`tau_table.csv`)

| τ | Unsafe approvals | Wasted refusals | Pre-flight: unsafe | Pre-flight: wasted |
|---|---|---|---|---|
| 0.50 | 1.49% | 5.9% | 2.40% | 5.1% |
| 0.80 | 0.48% | 14.2% | 0.87% | 12.6% |
| 0.90 | 0.21% | 21.0% | 0.22% | 18.5% |
| **0.95** | **0.05%** | **27.5%** | **0.02%** | **24.1%** |
| 0.99 | 0.03% | 36.9% | 0.00% | 33.0% |

τ = 0.95 stays the default: going from 0.90 to 0.95 cuts unsafe approvals about four-fold for 6 more points of refusals, while going on to 0.99 costs another 9 points for almost nothing. The value is still pending the guide's confirmation and lives in `config.yaml`. At these levels the unsafe counts are single digits, so the exact percentages are rough.

## Fleet simulation (`scheduler.py`, `fleet_simulation.csv`)

**Setup:** 4 drones, 40 tasks a day (70% single flights, 30% two-sortie missions), 200 simulated days; every policy sees the same tasks and the same battery draws.

**What is real and what is simulated:** each battery is one of the 34 real development chains that start full. Its true energy and the telemetry the battery model sees come from that chain, and each task's true cost is the measured energy of real flights. Only the *pairing* of batteries with tasks is simulated.

**One day:** each task goes to the approved drone with the least predicted energy (fuller batteries are kept for bigger tasks). If no drone is approved, the emptiest one swaps its battery and is asked again; if it's still refused the task is dropped. A mission that draws more than the battery truly had above the reserve is an **unsafe mission**.

| Policy | Unsafe missions per 100 flown | Completed per day (of 40) | Battery swaps per day | Energy left at swap |
|---|---|---|---|---|
| P1 minutes left (the brief) | **11.1** | 40.0 | 15.1 | 4.3 Wh |
| P2 energy, point estimates | 4.3 | 40.0 | 15.8 | 5.9 Wh |
| P1 + margin (matched to P3) | 0.0 | 7.0 | 33.0 | 60.9 Wh |
| P2 + 7.9 Wh margin (matched to P3) | 0.35 | 39.9 | 18.5 | 13.9 Wh |
| **P3 EnduroSense, τ = 0.95** | **0.23** | 39.4 | **16.9** | 10.7 Wh |
| P4 oracle | 0.0 | 39.9 | 15.7 | 5.7 Wh |

- **Against the brief's rule: 48 times fewer unsafe missions** (0.23 against 11.1 per 100) for 12% more battery swaps and 0.6 tasks a day dropped.
- **Against point estimates with no margin: 19 times fewer unsafe missions** for 7% more swaps.
- **Against point estimates with a tuned margin:** the same safety within noise (0.09 ± 0.04 against 0.14 ± 0.05 unsafe missions a day), with **8% fewer battery swaps** (16.9 ± 0.2 against 18.5 ± 0.2) and 3 Wh less left unused in each battery. It drops slightly more tasks (0.6 against 0.15 a day): two-sortie tasks that a fresh battery can't be 95% sure of.
- **Minutes-left can't be rescued with a margin.** Matching EnduroSense's safety needs a margin so large that the fleet completes 7 of 40 tasks.
- **The gap to the oracle** (1.2 swaps a day, 5 Wh per battery) is the cost of Model A's remaining uncertainty.

## An artefact found and fixed

The first run of the simulation showed EnduroSense dropping 3.6 tasks a day. The cause was the simulator, not the model: a fresh battery was scored at the **first reading of its recording**, when the battery model has seen a single sample and its 90% range is 24–45 Wh wide. A real decision is made at the pre-flight reading about 16 s later, where the range is about 19 Wh. `batteries_from()` now starts each battery at its pre-flight reading. This applies to every policy and a test covers it.

A related, real finding: **a fresh battery is the hardest state to judge.** At the first flight's pre-flight reading the range is about 19 Wh wide and the mean error is 4.2 Wh, against 11 Wh and 2.3 Wh overall. The discharge curve is flat near full charge and each battery's capacity is slightly different.

## Limitations

- **What-if pairs are not flights.** A battery state is paired with a mission flown on another day in other conditions. Battery temperature and ageing effects that would link the two are not captured.
- **Multi-sortie missions ignore the ground time between sorties** in the real recordings; their energy is the sum of the flights.
- **In the fleet simulation,** after a mission the battery's telemetry is taken from the chain reading with the same energy drawn, which is usually a mid-flight reading. Phase 5 showed the battery model is equally well calibrated in flight and on the ground, so this doesn't favour a policy.
- **The matched margins for P1 and P2 are tuned on the same pairs they're scored on,** which flatters them slightly. EnduroSense's τ is not tuned.
- **Very low rates rest on few events.** 0.05% is 8 pairs. Phase 7 repeats everything on the locked test set.
- **All of this is development data.** No test data was used.

## Changes from the plan

- P(success) uses a deterministic quantile grid in place of 10,000 random draws (same answer, repeatable, faster). No Gaussian closed form was needed.
- The borderline set is reported on its own, not mixed in at 30%.
- Pre-flight states were added as a separate set, because that is where real decisions are made.
- Model B's selection helpers moved from `scripts/06_model_b.py` into `models/model_b.py`, and the leave-fold-out calibration helpers into `uncertainty/`, so Phases 5 and 6 share one implementation. Re-running Phases 4 and 5 afterwards reproduced every number exactly.
- `scripts/07_uncertainty.py` now also saves each held-out reading's full set of quantiles and each flight's held-out predicted parts, which this phase reads.

## What Phase 7 and Phase 8 get

- `feasibility.p_success(available, required)` and `decide(p, tau)` for single decisions (the demo's mission-check page); `p_success_batch` for tables of them.
- `whatif.policy_scores`, `summary_table`, `paired_gain`, `tradeoff_curve` and `probability_reliability` run unchanged on test-set states and missions.
- `scheduler.FleetSimulator` runs on any set of chains.
- `results/decisions/whatif_pairs.parquet` holds every pair with all four policy scores: the source for the demo's "minutes-left says go, EnduroSense says no" examples (development data; Phase 7 produces the test-set version).

## Verification

- `pytest`: **97 passed**; `pyflakes` clean.
- **New tests (`tests/test_decisions.py`):**
  - P(success) matches the closed form for normal distributions; batch and single versions agree
  - P(success) falls as the mission grows and rises with more energy; wider uncertainty pulls it towards 50%
  - the decision rule, including NO-GO on abstention
  - the two mistake rates on a hand-counted case
  - a perfect score approves every feasible mission with no unsafe approvals
  - pairs carry the right truth; pre-flight states are motors-off readings before take-off
  - in a toy world where the models are right, the oracle makes no mistakes and τ = 0.95 approves nothing unsafe
  - the reliability table recovers an honest probability
  - batteries follow their chain and start at the pre-flight reading
  - fleet simulation: every task is flown or dropped, the oracle is never unsafe, a reckless policy is, and the same seed gives the same days
