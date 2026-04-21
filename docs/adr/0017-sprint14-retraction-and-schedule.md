# ADR-0017 — Sprint 14: per-seed retraction + phase-transition schedule

**Status:** Accepted — ADR-0016 B-2 migration-peak claim is
**retracted**; codebook entropy is the wrong resolution; B-grid
result (phase-transition schedule) pending
**Date:** 2026-04-22
**Sprint:** 14

## Context

Sprint 13 (ADR-0016) reported three findings from grid-level
median aggregates:

1. **B-2 bimodal** positive structure at gumbel tau=0.20 and
   tau=0.30 (+0.0116, +0.0180).
2. **Codebook freeze destroys B-1** — compound
   LOCK=100 + CODEBOOK_LOCK=100 wipes the +0.0125 peak to 0.0000.
3. **Three distinct plasticity roles**: mux preserves T1/T2
   asymmetry, hard transducer filters gate noise, codebook must
   stay plastic.

Honest pre-registration demands three falsification tests before
those claims are load-bearing in the paper:

- **14a — per-seed stability.** Do the grid-median B-1 and B-2
  peaks replicate at the level of individual seeds, or are they
  bootstrap smoothing of near-zero distributions?
- **14b — mechanistic visibility.** Does the codebook entropy
  scalar *itself* differ between LOCK=100 (peak) and
  LOCK=100+CBFREEZE (no peak) configurations? If not, the
  "codebook must stay plastic" story needs a finer observable.
- **14c — phase-transition compound.** The Sprint 10 B-1 peak
  (+0.0125) uses HARD gating; the Sprint 12/13 tau=0.30 "B-2
  peak" uses GUMBEL gating. These are mutually exclusive at a
  given step. A phase-transition schedule (HARD during Phase 1,
  GUMBEL during Phase 2) could in principle combine both signals.

## 14a — tau=0.30 per-seed stability

Re-aggregation of `runs/v05_s12_tau0_3` with
`scripts/aggregate_grid_per_seed.py`, grouping cells by seed
before invariant computation:

| seed | B-1 Me7 | B-2 Me3_delta | B-3 Me6 |
|-----:|--------:|--------------:|--------:|
| 0 | -0.0063 | 0.0000 | 0.1094 |
| 1 | +0.0125 | 0.0000 | 0.1406 |
| 2 | +0.0125 | 0.0000 | 0.1328 |
| 3 | +0.0125 | 0.0000 | 0.1172 |
| 4 | -0.0063 | 0.0000 | 0.1172 |
| **mean ± std** | **+0.0050 ± 0.0103** | **0.0000 ± 0.0000** | **+0.1234 ± 0.0128** |
| **sign stability** | 3/5 positive | 0/5 positive | 5/5 positive |
| **passes threshold** | 0/5 | 0/5 | 5/5 |

### Decision — retract the ADR-0016 B-2 positive peak claim

**B-2 is exactly 0.000 at every one of the 5 seeds.** The grid-
median value of +0.0180 reported in ADR-0016 arose from
`me9_bootstrap` resampling a distribution of near-zero `me3_delta`
values where the kNN-Kraskov MI estimator quantises to integer
bits and most deltas round to 0.0. A few positive outliers at the
T1-vs-T2 pair level shifted the bootstrap median upward. This is
**bootstrap smoothing, not a B-2 signal**.

Accordingly, ADR-0016 §13a "two positive B-2 peaks" is retracted:
tau=0.30 does not produce a robust positive B-2, and the bimodal
pattern reported there (tau=0.20: +0.0116, tau=0.30: +0.0180)
should be re-examined under the same per-seed lens before being
claimed as an architectural phenomenon.

B-1 is **weakly** robust at tau=0.30: 3/5 seeds show +0.0125
(the Sprint 10 peak value), 2/5 show -0.0063. Mean +0.0050 is
10× below the 0.05 threshold. None of the 5 seeds pass the
pre-registered threshold individually.

B-3 remains architecturally invariant (5/5 seeds pass, 6.2× the
threshold on average), confirming once more its decoupling from
the critical-period mechanism.

## 14b — codebook entropy trajectory is the wrong resolution

`scripts/plot_codebook_entropy.py` walks per-cell `report.pkl`
files, extracts `AdaptationReport.codebook_entropy_trajectory`
(shape `(steps//stats_every + 1,)`, 1 value per 10 training
steps), and computes median + IQR across the 150 cells of each
grid.

| Grid | H(0) nats | H(final) nats | ΔH |
|------|----------:|--------------:|---:|
| `v05_dr_lock100` (Sprint 10 peak) | 4.15451 | 4.15445 | -6e-5 |
| `v05_s13_cbfreeze` (freeze, no peak) | 4.15449 | 4.15444 | -5e-5 |

### Decision — codebook entropy is not the right observable

Both trajectories are **flat** at ~4.1545 nats (within 2e-4 of
`ln(64) = 4.1589`, the theoretical maximum). The expected
"rising entropy during P2 for LOCK=100 vs flat for CBFREEZE"
signature is **not visible at the median-over-cells resolution**.

The per-seed / per-cell curves do oscillate at the 1e-3 nats
scale (visible in individual T1 audio cells), but median
aggregation collapses them to the noise floor.

Conclusion: the Sprint 13b mechanism ("codebook freeze destroys
B-1") is real at the outcome level (B-1 = +0.0125 vs 0.0000),
but it does **not** operate via a detectable reshape of the
codebook entropy profile. The mechanism is finer — it concerns
**which** codebook entries move during Phase 2 (directional
structure of `codebook.grad`, pair-wise re-distribution patterns),
not **how much** entropy the codebook carries globally.

ADR-0016's "codebook re-distributes information across the 64-
entry alphabet" narrative is preserved in spirit but requires a
finer observable (per-entry drift, pair-wise L2, or activation-
conditioned entropy) to be visible. Paper v0.2 §5.8 should soften
the mechanistic interpretation accordingly.

## 14c — phase-transition HARD→GUMBEL schedule (pending)

**Config:** LOCK_AFTER=100, TRANSDUCER_GATING=hard,
GATING_SCHEDULE=200 (= STEPS_TRAIN, checkpoint boundary),
GATING_TARGET=gumbel, GUMBEL_TAU=0.30,
STEPS_TRAIN=200, STEPS_LESION=100, 150 cells.

**Hypothesis:** T2 cells inherit a checkpoint with
`codebook_step=200`, so the first Phase 2 forward pass resolves
`_active_gating()` → "gumbel". T1 cells never cross 200 (only
100 lesion steps), so they stay HARD throughout. This matches
Amedi's clinical picture: late-acquired modalities experience
a critical-period regime change that congenital ones do not.

**Pre-registered outcomes:**

- If B-1 ≥ +0.0125 AND B-2 ≥ +0.01: first joint positive result
  in the benchmark's history; Sprint 14c is the main finding.
- If B-1 ≥ +0.0125 AND B-2 ≤ 0: the HARD→GUMBEL schedule just
  reproduces Sprint 10 (mux lock dominates), no interaction.
- If B-1 ≤ 0 AND B-2 ≥ +0.01: migration channel opens at cost
  of asymmetry; the two effects are antagonistic.
- If B-1 ≤ 0 AND B-2 ≤ 0: null result, schedule adds no value.

Grid running on studio (`runs/v05_s14_joint`, PID 53333).
Per-seed aggregator will be run on the output before any claim.

**Results (grid-level):**

| Invariant | Value | Threshold | Passes? |
|-----------|------:|----------:|:-------:|
| B-1 Me7 | +0.0063 | 0.05 | ❌ (0.13×) |
| B-2 Me3_delta | **-0.0220** | 0.10 | ❌ |
| B-3 Me6 | +0.1094 | 0.02 | ✅ (5.5×) |

**Results (per-seed):**

| seed | B-1 Me7 | B-2 Me3_delta | B-3 Me6 |
|-----:|--------:|--------------:|--------:|
| 0 | +0.0062 | -0.0022 | 0.0781 |
| 1 | 0.0000 | 0.0000 | 0.1250 |
| 2 | +0.0125 | 0.0000 | 0.0625 |
| 3 | +0.0063 | -0.0184 | 0.1250 |
| 4 | **-0.0375** | 0.0000 | 0.1094 |
| **mean ± std** | **-0.0025 ± 0.0201** | **-0.0041 ± 0.0080** | **+0.1000 ± 0.0284** |
| **sign stability** | 3/5 positive, 1/5 negative | 0/5 positive | 5/5 positive |
| **passes threshold** | 0/5 | 0/5 | 5/5 |

### Decision — phase-transition schedule is null to regressive

The HARD→GUMBEL tau=0.30 schedule at step 200 **does not**
combine the Sprint 10 HARD peak (B-1=+0.0125, 3/5 seeds) with
the Sprint 12 "gumbel tau=0.30 peak" (already retracted in
§14a). The compound configuration:

- degrades B-1 grid-median from +0.0125 (Sprint 10 HARD)
  to +0.0063 — a **2× reduction**, with seed 4 collapsing
  to -0.0375 (no prior single-mode run on this bridge produced
  a seed that negative);
- pushes B-2 negative (-0.0220 grid, mean -0.0041 per-seed,
  seed 3 at -0.0184) — worse than pure HARD (B-2 ≈ 0) or pure
  GUMBEL at any tau from Sprints 11-13;
- preserves B-3 (architectural invariant, 5/5 seeds pass).

The hypothesis "phase-transition captures both B-1 and B-2" is
**falsified**. Worse, the schedule appears to introduce
**destructive interference** between the two regimes — seed 4's
-0.0375 B-1 is below anything observed in the single-mode
series. Changing the gating mode at the P1/P2 boundary may
disrupt whatever structure the HARD gate established during
pretrain, without the GUMBEL regime being able to productively
re-use it.

## Sprint 10-14 synthesis — what survives

After five sprints on the 4.5-modal real bridge, exactly one
configuration produces a positive B-1 peak of practical
significance:

| Config | B-1 (grid) | Seeds positive | B-2 | B-3 |
|--------|-----------:|---------------:|----:|----:|
| LOCK=100, HARD (Sprint 10 peak) | **+0.0125** | **3+/5** | ~0 | 0.109 |
| LOCK=100, GUMBEL tau=0.30 (Sprint 12/13a) | +0.0063 | 3/5 | 0 (per-seed) | 0.125 |
| LOCK=100, HARD + CBFREEZE (Sprint 13b) | 0.0000 | 0/5 | -0.012 | 0.109 |
| LOCK=100, HARD→GUMBEL sched (Sprint 14c) | +0.0063 | 3/5, 1 neg | -0.022 | 0.109 |

The **pure HARD Sprint 10 peak remains the only load-bearing
B-1 result** in the paper. Every compound attempted since has
either preserved, weakened, or destroyed it. B-2 is never
robustly positive in any configuration; B-3 is always positive
and architecturally invariant regardless of P1/P2 knobs.

## Pre-registration fidelity

- No threshold changes.
- No metric-math changes.
- All prior ADRs 0010-0015 remain canonical.
- ADR-0016 §13a "bimodal positive B-2" is **retracted** in
  favour of "per-seed B-2 is exactly 0.000 at tau=0.30".
- ADR-0016 §13b outcome (codebook freeze destroys B-1) is
  **preserved** but its *mechanistic* claim (entropy re-
  distribution) is **softened** pending a finer observable.
- Sprint 14 adds one hyperparameter
  (`transducer_gating_schedule`) and one complementary one
  (`transducer_gating_target`). OSF amendment v0.5 "follow-up"
  covers both.

## Future directions

- Per-seed replication of the full Sprint 13a tau grid
  {0.20, 0.25, 0.35, 0.40} to confirm that the B-2 retraction
  generalises beyond tau=0.30.
- Finer codebook-movement observable: per-entry L2 drift
  between Phase 1 end and Phase 2 end, aggregated by gate-
  proximal vs gate-distal pairs.
- MINE / InfoNCE replacement for the kNN-Kraskov Me3_delta
  estimator (§6.3 limitation): test whether the ~0 B-2 is a
  property of the network (no migration happens) or a property
  of the estimator (migration happens but Kraskov cannot see it).
- If 14c is null: paper v0.2 scope narrows to the **mux-lock
  + hard-transducer + plastic-codebook** three-way story, with
  B-2 and the migration-peak line of argument dropped entirely.
