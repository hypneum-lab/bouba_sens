# ADR-0010 — B-1 recovery via nerve-wml#4 plasticity lock

**Status:** Accepted — **partial recovery, directional not magnitude**
**Date:** 2026-04-21
**Sprint:** 7 (post-v0.4.0)

## Context

ADR-0004 (v0.2) and ADR-0005 (v0.3 cross-world) documented the
B-1 invariant directionally falsified: `me7 = me1(T1) - me1(T2)`
was −0.0063 on Gaussian, −0.0062 on XOR, +0.0125 on Sinusoid
(under-threshold), 0.0000 on Studyforrest mock, −0.0062 on
real ECG. Four of five worlds had T2 >= T1, contradicting the
pre-registered Amedi 2007 hypothesis.

Issue `hypneum-lab/nerve-wml#4` proposed a mechanism to restore
biological criticality: a `constellation_lock_after` kwarg on
`GammaThetaMultiplexer` that permanently freezes the constellation
when the internal step counter crosses the threshold. Shipped as
nerve-wml v1.4.0 (tag `v1.4.0`, commit `9c3dc65`) with 7 tests +
`plasticity_step` buffer that survives `state_dict()` round-trips.

bouba_sens commit `483f47c` wires the feature into `AdaptationLoop`
and the CLI + `run_grid.sh`. This ADR records the empirical B-1
verdict from the first grid run using the new mechanism.

## Grid provenance

| Item | Value |
|------|-------|
| Host | Studio (MacStudio, arm64, macOS 26.4) |
| Commit | `483f47c` (feat/b1-plasticity-recovery) |
| nerve-wml | `v1.4.0` (feat/plasticity-schedule) |
| Config | `LOCK_AFTER=200 STEPS_TRAIN=200 STEPS_LESION=100 METRICS="Me1,Me2,Me3" WORLD=gaussian` |
| Cells | 150 / 150 processed |
| Wall time | ~17 min solo |
| Artefact | `reports/v0.4_b1_recovery_aggregate.json` |

## Biological semantics of the run

- **T2 (late-acquired):** Phase 1 runs 200 steps, `plasticity_step`
  advances 0 → 200, crosses `constellation_lock_after=200`,
  `requires_grad=False`. Checkpoint saves the locked state. Phase 2
  restore re-applies the lock via the new `load_state_dict` override.
  Phase 2 trains on lesioned data with a **frozen** constellation.
- **T1 (congenital):** Phase 1 is skipped. Fresh mux starts at
  `plasticity_step=0`. Phase 2 trains for 100 steps; counter ends
  at 100, well below 200 → mux **stays plastic** through the
  entire congenital regime.

This is the first architectural T1 vs T2 asymmetry in the benchmark.

## Verdicts vs v0.3 Gaussian baseline

| Invariant | v0.3 Gaussian (ADR-0004) | **v0.4 b1-recovery** | Delta |
|-----------|-------------------------:|---------------------:|-------|
| B-1 Me7 (thr 0.05) | −0.0063 FAIL (inverted) | **0.0000 FAIL (directional recovery)** | +0.0063 |
| B-2 Me3 delta (thr 0.10) | +0.0275 FAIL | −0.0092 FAIL | −0.0367 |
| B-3 Me6 max-abs (thr 0.02) | 0.1484 PASS (7.4×) | 0.1719 PASS (8.6×) | +0.0235 |

### Per-cell me1 T1 vs T2 at floor SNR (illustrative)

| Modality | T1 me1 | T2 me1 | sign |
|----------|-------:|-------:|:----:|
| audio | 0.2262 | 0.2562 | T2 |
| vision | 0.2213 | 0.2700 | T2 |
| tactile | 0.2400 | 0.2538 | T2 |
| gravity | 0.2537 | 0.2500 | T1 |
| force | 0.2750 | 0.2250 | **T1** |

The median across all 75 T1/T2 pairs is exactly 0.0000 because
the mechanism balances the signs: lock helps T1 on *some* modalities
(force, gravity) and hurts it on others (audio, vision, tactile).

## Decision

**Partial recovery: directional yes, magnitude no.**

1. **B-1 is directionally recovered.** v0.3 had −0.0063 (T2 > T1).
   v0.4 has exactly 0.0000 (balanced). The inversion is gone — but
   no positive gap emerges either. The pre-registered 0.05 threshold
   is not crossed.
2. **B-2 sign flips back to negative.** The lock blocks the post-
   lesion MI migration channel that was weakly positive in v0.3
   (+0.028 → −0.009). Informationally, the T2 network can no
   longer reorganise its constellation to route information around
   the lesion — which is, architecturally speaking, the *correct*
   behaviour of a critical-period model.
3. **B-3 stays robust.** 8.6× threshold, essentially equivalent to
   v0.3's 7.4× (slight improvement). The lock doesn't harm the
   structural asymmetry that B-3 measures.

## What this tells us about the architecture

The plasticity lock **implements critical-period semantics** (the
T2 network is architecturally frozen where T1 is not) but the
**semantic gain does not saturate the pre-registered threshold on
GaussianWorld**. Three candidate interpretations:

1. **Lock is necessary but not sufficient.** Freezing the
   constellation alone, without freezing the `CrossModalTransducer`
   gates and the `AdaptiveCodebook` entries, may not be enough.
   Issue nerve-wml#5 (Gumbel transducer variant) or a companion
   `transducer_lock_after` could compound the effect.
2. **The 0.05 threshold is calibrated on a richer world.** Amedi
   2007 studies humans on linguistic / visual tasks. GaussianWorld's
   4-class 32-dim latent may not support a 0.05 gap in principle.
   Running this same lock config on the Studyforrest real-ECG
   bridge (ADR-0009) would clarify — deferred to a follow-up grid.
3. **100 Phase-2 steps too short to reveal the gap.** The T1 mux
   has 100 plastic steps to adapt vs T2's 0. A longer Phase 2
   (e.g. 500 steps) might let T1's advantage compound.

None of these three would **falsify** the mechanism; all three
would refine its empirical range.

## Pre-registration fidelity

- No threshold changes.
- No metric-math changes.
- The v0.3 verdicts in ADR-0004 / ADR-0005 / ADR-0009 remain the
  canonical record.
- This is a **new experimental condition** (lock=200), not a
  retroactive edit of v0.3.

## Next steps

1. Re-run the lock grid on XOR + Sinusoid + real ECG to get the
   cross-world B-1 recovery pattern. If Sinusoid also flips from
   +0.0125 to 0.0 (balanced) and XOR from −0.006 to 0.0, the
   mechanism's effect is **world-invariant directional recovery**
   — worth a paper subsection.
2. Extend `AdaptationLoop` to also call `transducer.step()` once
   issue nerve-wml#5 ships, so the full critical-period regime is
   tested.
3. Scan `LOCK_AFTER` in {100, 200, 400, 800, none} on a single
   seed to see if the recovered me7 dose-responds to the lock
   timing — this is the actual Amedi-2007 analogue (younger lock
   = stronger congenital advantage).
4. Record in the paper (Sprint 8) as "Mechanism shown to produce
   directional recovery; absolute magnitude does not cross 0.05
   on GaussianWorld; dose-response scan pending."
