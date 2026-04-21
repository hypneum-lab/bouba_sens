# ADR-0014 — Compound lock + gumbel transducer (Sprint 11)

**Status:** Accepted — compound hypothesis **falsified**, soft gating dilutes B-1 peak
**Date:** 2026-04-21 late evening
**Sprint:** 11

## Context

ADR-0013 established a B-1 peak at `LOCK_AFTER=100` on the 4.5-modal
real bridge with the v0.5 default hard transducer gating. The
nerve-wml#5 (closed) and bouba_sens#5 issue chains proposed that
compounding the constellation lock with a soft (Gumbel-style)
transducer gate could push B-1 above the pre-registered 0.05
threshold, by letting gradients flow continuously through the
`gate[src] < 0.1 AND gate[dst] > 0.3` rule instead of hard
thresholding it.

Sprint 11 tests this by adding `transducer_gating` kwarg to
`CrossModalNerve` with two options — `"hard"` (v0.5 default,
byte-identical behaviour) and `"gumbel"` (sigmoid-soft on the
`(gate_dst - 0.3) - (gate_src - 0.1)` margin, modulated by
`gumbel_tau`). CLI + run_grid.sh threading added. One new grid
run: `LOCK_AFTER=100 TRANSDUCER_GATING=gumbel` on the existing
4.5-modal `data/sf_phase2_adapted` tensors from Sprint 9.5.

## Grid provenance

| Item | Value |
|------|-------|
| Host | Studio |
| Worktree | `~/Projets/bouba_sens_b1` |
| Commit | `06e6efb` (main) |
| nerve-wml | `v1.5.3` (upstream #5 closed) |
| World | `StudyforrestRealWorld(data_dir=data/sf_phase2_adapted)` |
| Config | `LOCK_AFTER=100 TRANSDUCER_GATING=gumbel STEPS_TRAIN=200 STEPS_LESION=100 METRICS="Me1,Me2,Me3"` |
| Grid | 150 cells (`runs/v05_s11_compound`) |
| Wall time | ~20 min solo |
| Artefact | `reports/v0.5_s11_compound_aggregate.json` |

## Verdict table vs Sprint 10 baseline

| Config | B-1 Me7 | B-2 Me3_delta | B-3 Me6 |
|--------|--------:|--------------:|--------:|
| LOCK=100 + HARD (Sprint 10 peak) | **+0.0125** | -0.0190 | 0.1094 |
| LOCK=100 + GUMBEL (this ADR) | **+0.0062** | -0.0177 | 0.1172 |
| Delta | **-50 %** | +7 % (less negative) | +7 % |

Pre-registered thresholds unchanged (0.05 / 0.10 / 0.02).

## Decision — compound hypothesis is honestly falsified

**Gumbel transducer gating reduces the B-1 peak by half.** The
naïve reading "soft differentiation = better gradient flow =
larger T1/T2 asymmetry" is not supported by the data. The
Sprint 10 peak was load-bearing on the **hard binary gate** —
when the gate flips from inactive to active, the transducer
switches abruptly, preserving discrete T1 vs T2 phase-1 history.
The sigmoid-soft interpolation **smears** that history across
the training trajectory, reducing the effective critical-period
signal.

Three candidate mechanisms, any of which could be refined in
a future sprint:

1. **Hard gate acts as a noise-filter**: by rejecting sub-
   threshold gate margins, the v0.5 rule suppresses spurious
   gradient updates that don't meet the biological "enough
   activity to route" threshold. Gumbel lets every pair route
   proportionally, which dilutes the signal-to-noise on the
   T1/T2 asymmetry axis.
2. **Margin scaling is miscalibrated**: `gumbel_tau=1.0` gives
   a mild sigmoid (slope ~0.25 at 0). A smaller `tau` would
   tighten the sigmoid toward the hard rule; a larger `tau`
   would flatten it further. Sprint 11 did not scan this
   hyperparameter.
3. **Temporal interference compounds**: Sprint 10 already
   noted that B-2 is most negative at the B-1 peak (i.e.
   lock=100 is a local attractor for interference).
   Gumbel may further spread that interference across more
   transducer pairs.

Of the three, (1) is the most compelling mechanistic reading:
the benchmark is telling us that **some part of the B-1
asymmetry mechanism requires a hard phase-transition**, not a
continuous path. This is a scientifically interesting finding
in its own right — it constrains the space of architectures
that could reproduce Amedi 2007 critical-period effects.

## Paper v0.2 integration

§5.7 Compound critical-period (Sprint 11):

> A compound experiment combining the Sprint 10 peak lock
> (`LOCK_AFTER=100`) with sigmoid-soft transducer gating reduces
> the B-1 peak from +0.0125 to +0.0062 on the 4.5-modal real
> bridge, a 50 % attenuation. Contrary to the naïve hypothesis
> that soft differentiability improves critical-period signal,
> the hard binary `CrossModalTransducer` gate appears to act as a
> noise-filter that preserves the T1/T2 asymmetry. Soft gating
> dilutes the signal across all 20 cross-modal pairs
> proportionally, reducing the effective signal-to-noise on the
> Me7 axis. This is an **architectural constraint** on what
> mechanisms can reproduce Amedi 2007 in this setup: at least
> one component of the plasticity router needs a hard phase-
> transition, not a continuous gate.

## Pre-registration fidelity

- No threshold changes.
- No metric-math changes.
- ADR-0013 dose-response peak remains canonical.
- This ADR strictly extends the matrix with one new
  `transducer_gating` variable at a single `LOCK_AFTER=100`
  point. OSF amendment v0.5 "follow-up" clause covers
  lock-combined experiments.

## Future directions

- **Sprint 12**: scan `gumbel_tau ∈ {0.1, 0.3, 0.5, 1.0, 2.0}`
  at the peak lock=100 to test whether a tighter sigmoid
  recovers the hard-gate peak or produces a new maximum
  in-between.
- **Alternative**: compound the peak lock with
  `AdaptiveCodebook` freezing (not yet an open issue, but
  natural next gate to test).
- Keep the default `transducer_gating="hard"` everywhere —
  v0.5 byte-identical behaviour preserved.
