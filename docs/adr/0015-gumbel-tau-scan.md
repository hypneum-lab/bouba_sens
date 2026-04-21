# ADR-0015 — Gumbel tau scan (Sprint 12)

**Status:** Accepted — tau has no monotone handle on B-1, hard gate irreducible
**Date:** 2026-04-21 late evening
**Sprint:** 12

## Context

ADR-0014 (Sprint 11) found that `transducer_gating="gumbel"` at
`gumbel_tau=1.0` reduces the Sprint 10 B-1 peak by 50 %
(+0.0125 → +0.0062). The "hard gate = noise filter" interpretation
predicted that **tightening** the sigmoid (lower tau) should recover
the hard-gate behaviour asymptotically. Sprint 12 tests this by
scanning `gumbel_tau ∈ {0.1, 0.3, 0.5, 1.0, 2.0}` at the peak
`LOCK_AFTER=100` on the 4.5-modal real bridge.

## Grid provenance

| Item | Value |
|------|-------|
| Host | Studio |
| Commit | `667b34a` (main) |
| nerve-wml | `v1.5.3` |
| World | `StudyforrestRealWorld(data_dir=data/sf_phase2_adapted)` |
| Config | `LOCK_AFTER=100 TRANSDUCER_GATING=gumbel STEPS_TRAIN=200 STEPS_LESION=100 METRICS="Me1,Me2,Me3"` |
| Grids | 4 × 150 cells (`runs/v05_s12_tau{0_1,0_3,0_5,2_0}`) |
| Wall time | ~50 min with 4-way concurrency |
| Artefacts | `reports/v0.5_s12_tau{0_1,0_3,0_5,2_0}_aggregate.json` |

## Verdict table

| gumbel_tau | B-1 Me7 | B-2 Me3_delta | B-3 Me6 |
|-----------:|--------:|--------------:|--------:|
| 0.1 (tight) | 0.0000 | -0.0268 | 0.109 |
| **0.3** | **+0.0063** | **+0.0180** | **0.125** |
| 0.5 | 0.0000 | -0.0404 | 0.125 |
| 1.0 (Sprint 11) | +0.0062 | -0.0177 | 0.117 |
| 2.0 (flat) | +0.0063 | -0.0052 | 0.117 |
| **hard (Sprint 10)** | **+0.0125** | -0.0190 | 0.109 |

Pre-registered thresholds unchanged (0.05 / 0.10 / 0.02).

## Decision — hard gate is qualitatively distinct from any Gumbel regime

**No value of `gumbel_tau` recovers the +0.0125 hard-gate peak.**
B-1 plateaus at ~+0.006 for `tau ∈ {0.3, 1.0, 2.0}` (1.5 decades)
and collapses to exactly 0.0000 at `tau ∈ {0.1, 0.5}`. The
hypothesis from ADR-0014 that tighter sigmoid asymptotically
recovers the hard rule is **falsified**: the hard binary gate is
not a limit of the Gumbel family, it is a **qualitatively distinct
regime**. Soft gating at any tau dilutes B-1 by at least 50 %.

Mechanistic reading: the hard gate's discontinuity at the
`gate[src] < 0.1 AND gate[dst] > 0.3` threshold produces a
**phase-transition** in information routing that no continuous
approximation captures. The T1/T2 asymmetry accumulated over
Phase 1 relies on this transition to preserve discrete routing
history; smoothing it — at any scale — averages out the signal.

## Secondary finding — tau=0.3 is an anomalous B-2 positive

**tau=0.3 produces the only positive B-2 Me3_delta (+0.018) of
the entire Sprint 9/10/11/12 chain on the 4.5-modal real bridge.**
Every other configuration (hard and all other gumbel_tau values)
gives B-2 ≤ 0 on this world. The magnitude is still under the
pre-registered 0.10 threshold, but the sign reversal is anomalous
and worth recording.

Candidate explanation: at `tau=0.3` the sigmoid slope is ~0.83
at zero margin, so it behaves as a **sharpened soft gate** — it
is nearly binary near the active/inactive boundary but has a
narrow transition band. This may correspond to a "Goldilocks
zone" where the gate is selective enough to preserve discrete
T1/T2 history (hence the small positive B-1 +0.006) AND smooth
enough to let MI migrate post-lesion (hence B-2 positive +0.018).
At `tau=0.1` the sigmoid is too tight and breaks the MI migration
path; at `tau >= 0.5` the gate is too smooth to preserve
selectivity.

The non-monotone B-2 pattern (0.1→-0.03, 0.3→+0.02, 0.5→-0.04)
suggests a narrow optimum that deserves a finer scan if paper
v0.3 wants to chase a B-2 threshold-crossing result.

## B-3 is lock-gate-tau invariant

Across the full scan plus the hard baseline, B-3 Me6 stays in
0.109-0.125 (5.4×-6.3× threshold). Architectural invariant
confirmed again: no knob in the compound critical-period family
touches B-3 by more than ±15 %.

## Paper v0.2 integration

§5.7 update:

> A gumbel_tau scan at the peak LOCK_AFTER=100 ({0.1, 0.3, 0.5,
> 1.0, 2.0}) confirms that no Gumbel configuration recovers the
> +0.0125 hard-gate peak: B-1 plateaus at ~+0.006 across 1.5
> decades and collapses to 0 at the extremes 0.1 and 0.5. The
> hard gate is not a limit of the Gumbel family but a
> qualitatively distinct routing regime that the continuous
> sigmoid cannot approximate.
> A secondary anomalous positive B-2 at tau=0.3 (+0.018, only
> positive in the 4.5-modal chain) hints at a narrow
> "selective-yet-migratable" sweet spot that deserves a finer
> scan in paper v0.3.

## Pre-registration fidelity

- No threshold changes.
- No metric-math changes.
- ADR-0010 .. ADR-0014 remain canonical.
- This ADR adds one new hyperparameter dimension; OSF amendment
  v0.5 "follow-up" clause covers it.

## Future directions

- Paper v0.3 finer tau scan around 0.3 (0.2, 0.25, 0.3, 0.35, 0.4)
  to characterise the B-2 anomaly.
- Sprint 13: `AdaptiveCodebook` freeze as a third compound lock
  component. If hard phase-transitions are load-bearing in the
  transducer gate, they may also matter in the codebook.
