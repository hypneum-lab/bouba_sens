# ADR-0016 — Finer tau + codebook freeze (Sprint 13)

**Status:** Accepted — B-2 sweet spot is bimodal; codebook freeze destroys B-1 peak
**Date:** 2026-04-22
**Sprint:** 13

## Context

ADR-0015 (Sprint 12) found a single anomalous positive B-2 at
`gumbel_tau=0.3`; all other tau values and the hard baseline
gave B-2 ≤ 0 on the 4.5-modal real bridge. Sprint 13 tests
two orthogonal extensions:

- **13a** — finer tau scan `{0.2, 0.25, 0.35, 0.4}` around the
  0.3 anomaly to characterise the width and shape of the
  selectivity-vs-migration "Goldilocks zone".
- **13b** — `codebook_lock_after=100` as third compound
  critical-period component, testing whether the noise-filter
  interpretation of the hard transducer gate (ADR-0014)
  transfers to the `AdaptiveCodebook` Parameter.

## Grid provenance

| Item | Value |
|------|-------|
| Host | Studio |
| Commit | `9f53eaa` (main) |
| nerve-wml | `v1.5.3` |
| World | `StudyforrestRealWorld(data_dir=data/sf_phase2_adapted)` |
| Config | `LOCK_AFTER=100 STEPS_TRAIN=200 STEPS_LESION=100 METRICS="Me1,Me2,Me3"` |
| Grids | 4 tau grids + 1 cbfreeze grid, 150 cells each |
| Wall time | ~55 min with 5-way + external concurrency |

## 13a — finer gumbel_tau scan

| tau | B-1 Me7 | B-2 Me3_delta | B-3 Me6 |
|----:|--------:|--------------:|--------:|
| 0.20 | +0.0063 | **+0.0116** | 0.125 |
| 0.25 | +0.0062 | -0.0109 | 0.125 |
| **0.30** (Sprint 12) | +0.0063 | **+0.0180** | 0.125 |
| 0.35 | 0.0000 | -0.0183 | 0.109 |
| 0.40 | 0.0000 | -0.0036 | 0.125 |

### Decision — B-2 sweet spot is bimodal, not a single plateau

**Two positive B-2 values bracket a negative one.** tau=0.20
and tau=0.30 both give B-2 ≈ +0.01 to +0.02, but the
intermediate tau=0.25 gives B-2 = -0.011. This non-monotone
pattern rules out a clean "Goldilocks zone" reading: the
positive B-2 is not a stable plateau around tau=0.30 but a
**narrow phase structure** where at least two preferred tau
values exist, separated by a destructive-interference trough.

B-1 plateaus at ~+0.006 for tau ∈ {0.2, 0.25, 0.3} (the same
three tau values where the hard peak doubling is best
approximated), then collapses to exactly 0.0000 for
tau ∈ {0.35, 0.4}. The 0.3 / 0.35 boundary is thus a **critical
transition**: above it, all sigmoid interpolation rigidifies
just enough to erase T1/T2 history; below it, the three
positive B-2 values (and the B-1 plateau) appear.

Mechanistic interpretation: the MI-migration channel requires
the gate to cross the `dst > 0.3` threshold *fast enough* to
preserve phase-1 selectivity, yet *slowly enough* to let
distribution mass leak across related modality pairs during
Phase 2. A narrower sigmoid (low tau) preserves selectivity
but blocks migration; a wider sigmoid (high tau) enables
migration but dissolves selectivity. The two positive B-2
peaks are two distinct "trigonometric beats" between these
two constraints — B-2 and B-1 do not have a joint monotone
improvement direction in this family.

## 13b — codebook freeze verdict

| Config | B-1 Me7 | B-2 Me3_delta | B-3 Me6 |
|--------|--------:|--------------:|--------:|
| LOCK=100, HARD (Sprint 10 peak) | **+0.0125** | -0.0190 | 0.109 |
| LOCK=100, HARD, CODEBOOK_LOCK=100 | **0.0000** | -0.0117 | 0.109 |

### Decision — codebook freeze destroys B-1 peak

**Adding `codebook_lock_after=100` on top of the Sprint 10 hard-
gate configuration wipes the +0.0125 peak back to 0.0000.** The
codebook freeze does NOT repeat the noise-filter story from the
transducer gate. Architecturally, the T1/T2 asymmetry relied on
the codebook being able to **re-distribute** information across
its 64-entry shared PSK alphabet during the pre-lock phase. When
both the mux constellation AND the codebook freeze at step 100,
the T1 network (which starts from a post-P1 codebook in v0.5
even though its own mux is fresh — confusing, but that's how
the Phase-1 checkpoint flow works) has no degree of freedom left
to accumulate asymmetry vs T2.

B-2 magnitude relaxes from -0.019 to -0.012 (-38 %), consistent
with "frozen codebook = less interference", but B-1 collapses
altogether. Thus the three components differ in their role:

| Component | Freezing effect | Empirical signature |
|-----------|-----------------|---------------------|
| mux constellation | preserves T1/T2 asymmetry | +0.0125 B-1 peak (Sprint 10) |
| transducer gate | filters gate noise | hard > soft (Sprint 11/12) |
| codebook | degrades T1/T2 asymmetry | destroys peak (this ADR) |

The codebook needs to **stay plastic** for the critical-period
signal to survive. This is the opposite of the nerve-wml#5 prior
hypothesis.

## B-3 stays invariant

Across all 5 new grids, B-3 Me6 is 0.109-0.125. Architectural
invariant confirmed once more.

## Paper v0.2 integration

§5.8 update:

> A finer gumbel_tau scan {0.2, 0.25, 0.35, 0.4} around the
> Sprint 12 B-2 anomaly reveals a **bimodal** positive-B-2
> structure: tau=0.20 and tau=0.30 both give B-2 ≈ +0.01-0.02,
> but tau=0.25 between them collapses to -0.011. The sweet spot
> is not a plateau — it's two separated phases.
>
> A compound experiment combining the Sprint 10 peak lock with
> `codebook_lock_after=100` **destroys** the B-1 peak
> (+0.0125 → 0.0000), demonstrating that the three plasticity
> components have distinct load-bearing roles: mux lock
> **preserves** T1/T2 asymmetry, hard transducer gate
> **filters** gate noise, codebook plasticity is **essential**
> — freezing it collapses the Amedi signal.

## Pre-registration fidelity

- No threshold changes.
- No metric-math changes.
- All prior ADRs 0010-0015 remain canonical.
- This ADR adds one hyperparameter (tau) resolution + one new
  hyperparameter (codebook_lock_after). OSF amendment v0.5
  "follow-up" covers both.

## Future directions

- Characterise whether the bimodal B-2 pattern is seed-stable
  (replicate tau=0.20 and tau=0.30 with 3-5 distinct seed bases).
- Sprint 14: explore whether the two B-2 peaks coincide with
  distinct internal gate trajectories via PlasticityGate logging.
- Paper v0.3: combine tau=0.30 (B-2 positive) with LOCK=100
  hard baseline to test if the two mechanisms interact non-
  destructively — may unlock simultaneous positive B-1 AND B-2.
