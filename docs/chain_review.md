# Battery chain review (Phase 1)

**Reviewed on:** 2026-10-01
**Material:** `results/phase1/figures/chains_page1-4.png` (all 68 multi-flight chains) and `results/phase1/chain_links.csv`

## What each review panel shows

- **Grey line:** voltage under load across the whole chain, plotted against the energy drawn since the chain started.
- **Filled dots:** motors-off rest voltage before each flight.
- **Open dot:** the estimated rest voltage after the last flight.
- **Dashed line:** the reserve (22.6 V).

**What a correct chain looks like:** rest voltage falls at every step, and each drop fits the energy used on that flight (about 0.045–0.05 V per Wh at mid charge).

## Findings

| Item | Finding | Decision |
|---|---|---|
| All 68 multi-flight chains | Rest voltage falls at every step; no voltage rise between flights (also checked automatically in `tests/test_prepare.py`) | accept |
| Chain 16 (flights 81, 82, 83) | Flight 81's recording ends at about 20 A. With the Phase 0 end voltage it was wrongly left unlinked; with the estimated rest voltage it links to 82 and the curve is smooth | accept (fixed by the method, no override needed) |
| Uncertain links: flights 4 (−0.05 V), 6 (−0.08 V), 119 (−0.08 V) | Small negative jumps within measurement noise; curves are consistent | accept, keep the *uncertain* flag |
| Uncertain link: flight 17 (+0.37 V) | Larger recovery after a 6-minute rest; slope matches chain 9 | accept, keep the *uncertain* flag |
| Uncertain links: flights 213 (A2 ground test) and 224 (hover test, estimated start) | Not used for Model A labels on their own | accept |
| Chains 63, 64, 68, 93, 98 | Much deeper voltage sag under the same load: likely older or weaker batteries (higher internal resistance) | keep; motivates the internal-resistance feature for Model A |
| Chains 76, 77, 78 (flights 212–217) | Ground tests with almost no energy drawn | keep; they don't affect modelling |

**Overrides:** none (see `config/chain_overrides.yaml`).
