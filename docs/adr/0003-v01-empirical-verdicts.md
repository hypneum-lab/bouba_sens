# ADR-0003 — v0.1 empirical invariant verdicts (B-1 / B-2 / B-3)

**Status:** Accepted
**Date:** 2026-04-20
**Sprint:** 4 (close)

## Context

Sprint 4 Task 4.7 ran the full 5-seed x 5-modality x 2-timing x 3-SNR
grid on Studio (M3 Ultra). 150 cells completed in ~14 min. This ADR
records the empirical verdicts on the three pre-registered invariants
(B-1, B-2, B-3) per spec 1.2 and sets the scope for v0.2.

## Grid provenance

| Item | Value |
|------|-------|
| Host | Studio (MacStudio, arm64, macOS 26.4) |
| Commit | `1dc5822` (main) |
| Config | `STEPS_TRAIN=200`, `STEPS_LESION=100`, `METRICS="Me1,Me2,Me7"` |
| Cells | 150/150 processed, 0 skipped |
| Wall time | ~14 min |
| Artifacts | `reports/v0.1_aggregate.json`, `reports/v0.1_summary.html` |

## Verdicts

All three invariants report `passes=false`. The verdict is an
**honest NO-GO on empirical validation**, driven mostly by tooling
gaps in the v0.1 CLI rather than engine failure.

| Invariant | Threshold | Median | Cells counted | Passes |
|-----------|-----------|--------|---------------|--------|
| B-1 (Me7 congenital gap > 0.05) | 0.05 | 0.0000 | 10 / 10 | No |
| B-2 (Me3 delta > 0.10) | 0.10 | 0.0000 | 0 / 30 | No |
| B-3 (Me6 max-abs off-diag > 0.02) | 0.02 | 0.0000 | 0 / 30 | No |

### Root causes (per invariant)

**B-1 (me7 = 0 by construction).** The `eval` CLI computes
`me7 = me1_T1 - me1_T1 = 0` because the Phase 2 report it receives
contains a single accuracy curve, with no paired T1/T2 handle. The
underlying `me7_congenital_gap` implementation is correct (unit tests
green); the CLI needs paired-run wiring.

**B-2 and B-3 (metrics not emitted).** `run_grid.sh` passes
`--metrics "Me1,Me2,Me7"`, so `eval_report.json` never contains
`me3_delta` or `me6_max_abs`. The aggregator correctly reports
`cells_counted=0` rather than fabricating a verdict from missing data.
The metric implementations themselves (unit tests green) are not
exercised in the v0.1 grid.

## Decision

**v0.1 is a structural / reproducibility milestone, not a scientific
verdict on cross-modal plasticity.** The engine, codebook, CLI, grid
orchestrator, and aggregator all ship working. The 3 invariants stay
**unresolved** until v0.2 closes the coverage gaps.

Sprint 4 closes on the structural achievements:

- 150-run grid runs end-to-end on Studio
- 30 unique cells aggregate with finite bootstrap CIs (0 NaN, 0 Inf)
- All three threshold constants match spec 1.2 exactly
- CI workflow (`.github/workflows/full-benchmark.yml`) ready for
  self-hosted runner activation
- Structural tests (`tests/empirical/test_grid_structural.py`)
  green against the real artifact

## Scope for v0.2

Sprint 5 (planned) must fix the three coverage gaps:

1. **Wire me3 / me6 into the eval CLI.** Require pre-lesion MI
   sample for me3_delta; require the 5x5 cross-modal perf matrix
   for me6.
2. **Pair T1 and T2 runs for me7.** The CLI must accept
   `--t1-ckpt` and `--t2-ckpt` (or equivalent) and emit the true
   gap, not `me1_X - me1_X`.
3. **Re-run the 150-cell grid with the full metric set** and
   re-evaluate B-1, B-2, B-3 against the same thresholds.

Until Sprint 5 lands, consumers should treat the v0.1 aggregate as
a shape-and-infrastructure artifact, not as evidence about the
biological invariants.

## Pre-registration note

The thresholds (`0.05`, `0.10`, `0.02`) are fixed and match OSF
pre-registration. Sprint 5 will not revise them; any v0.2 verdict
is testable against the same numbers and directly comparable to
v0.1.
