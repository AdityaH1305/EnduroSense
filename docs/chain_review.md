# Battery chain review (Phase 1)

**Reviewed on:** 2026-10-01
**Material:** `results/phase1/figures/chains_page1-4.png` (all 71 multi-flight chains) and `results/phase1/chain_links.csv`
**Revised:** 2026-10-02, after the chronological-ordering fix

## What each review panel shows

- **Grey line:** voltage under load across the whole chain, plotted against the energy drawn since the chain started.
- **Filled dots:** motors-off rest voltage before each flight.
- **Open dot:** the estimated rest voltage after the last flight.
- **Dashed line:** the reserve (22.6 V).

**What a correct chain looks like:** rest voltage falls at every step, and each drop fits the energy used on that flight (about 0.045–0.05 V per Wh at mid charge).

## Findings

| Item | Finding | Decision |
|---|---|---|
| All 71 multi-flight chains | Rest voltage falls at every step; no voltage rise between flights (also checked automatically in `tests/test_prepare.py`) | accept |
| Flights 81 → 82 → 83 | Flight 81's recording ends at about 20 A. With the Phase 0 end voltage it was wrongly left unlinked; with the estimated rest voltage it links to 82 and the curve is smooth | accept (fixed by the method, no override needed) |
| Uncertain links: flights 4 (−0.05 V), 6 (−0.08 V), 119 (−0.08 V) | Small negative jumps within measurement noise; curves are consistent | accept, keep the *uncertain* flag |
| Uncertain link: flight 17 (+0.37 V) | Larger recovery after a 6-minute rest; slope matches the rest of its chain (flights 16–17) | accept, keep the *uncertain* flag |
| Uncertain links: flights 213 (A2 ground test) and 224 (hover test, estimated start) | Not used for Model A labels on their own | accept |
| Chains of flights 191–192, 193–194, 204–205, 263–264 and 277–279 | Much deeper voltage sag under the same load: likely older or weaker batteries (higher internal resistance) | keep; motivates the internal-resistance feature for Model A |
| Ground-test chains (flights 211–219) | Ground tests with almost no energy drawn | keep; they don't affect modelling |
| 9 chains joined by the 2026-10-02 ordering fix: 77–80, 108–109, 110–112, 149–152, 172–174, 189–190, 209–210, 216–218, 277–279 | Each starts on a fresh battery (jump of +1.95 to +3.23 V), then falls smoothly with links of +0.04 to +0.20 V | accept |

**Overrides:** none (see `config/chain_overrides.yaml`).

*Chain ID numbers change whenever chains are rebuilt, so this review refers to chains by their flights.*
