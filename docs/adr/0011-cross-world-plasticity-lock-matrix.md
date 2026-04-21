# ADR-0011 — Cross-world plasticity-lock matrix

**Status:** Accepted — **homogenising, not magnitude-recovering**
**Date:** 2026-04-21
**Sprint:** 7 (paper-v0.1-blocking)

## Context

ADR-0010 recorded a partial B-1 directional recovery on GaussianWorld
with `constellation_lock_after=200` (nerve-wml v1.4.0) — the
inversion `me7 = -0.0063` flipped to exactly `0.0000`. Next step
from that ADR was to re-run the lock grid on XOR, Sinusoid, and
real MIT-BIH ECG to test whether the recovery is world-invariant.
This ADR records those three grids.

## Grid provenance

| Item | Value |
|------|-------|
| Host | Studio (MacStudio, arm64, macOS 26.4) |
| Worktree | `~/Projets/bouba_sens_b1` (isolated from the other agent's sprint7/critical-validation checkout) |
| Commit | `483f47c` (feat/b1-plasticity-recovery, CLI wiring unchanged from ADR-0010) |
| nerve-wml | `v1.4.0` (feat/plasticity-schedule, worktree resolves file:// to the new API) |
| Config | `LOCK_AFTER=200 STEPS_TRAIN=200 STEPS_LESION=100 METRICS="Me1,Me2,Me3"` |
| Grids | 3 × 150 cells : `WORLD=xor`, `WORLD=sinusoid`, `WORLD=studyforrest` with `BOUBA_SENS_STUDYFORREST_DATA=data/studyforrest_real` (MIT-BIH ECG) |
| Wall time | ~40 min in parallel (4-way concurrency with another agent's `null_b3_robust`) |
| Artefacts | `reports/v0.4_b1_{xor,sinusoid,ecg}_lock_aggregate.json` |

## Cross-world matrix

| World | B-1 no-lock (v0.3 / ADR-0005/9) | B-1 lock=200 | Delta |
|-------|-------------------------------:|-------------:|-------:|
| Gaussian | −0.0063 (inverted) | **0.0000** | +0.0063 |
| XOR | −0.0062 (inverted) | **0.0000** | +0.0062 |
| Sinusoid | **+0.0125** (correct sign) | **0.0000** | **−0.0125** |
| real ECG | −0.0062 (inverted) | −0.0062 | 0.0000 |

| World | B-3 no-lock | B-3 lock=200 | Delta |
|-------|------------:|-------------:|-------:|
| Gaussian | 0.1484 (7.4×) | 0.1719 (8.6×) | +16 % |
| XOR | 0.1406 (7.0×) | 0.1250 (6.2×) | −11 % |
| Sinusoid | 0.1406 (7.0×) | 0.1562 (7.8×) | +11 % |
| real ECG | 0.4453 (22.3×) | 0.4453 (22.3×) | 0 % |

B-3 stays PASS in 4/4 worlds with the lock (all >> 0.02).

## Decision

**The plasticity lock is homogenising, not magnitude-recovering.**

Three distinct regimes emerge from the matrix :

1. **Synthetic + inverted (Gaussian, XOR).** Lock flips
   `me7 = -0.006` to exactly `0.000` — the inversion is removed.
2. **Synthetic + correct-sign (Sinusoid).** Lock flips
   `me7 = +0.0125` to exactly `0.000` — **the positive gap is
   destroyed**. The lock does not selectively help congenital-T1.
3. **Real biological signal (ECG).** Lock has **no measurable
   effect** (me7 stays at −0.0062, me6 unchanged to the 4th
   decimal). ECG's 3 zeroed modalities (tactile / gravity /
   force) dominate the architecture's response : the lock
   cannot operate on the 2 live modalities in a way that
   produces an observable T1/T2 differential.

### Why this is honest falsification of the simple "lock = Amedi" story

The naïve reading of nerve-wml#4 was : critical-period lock
should let T1 accumulate plasticity advantage that T2 lacks,
producing a positive me7 gap on at least one world. Zero of
four worlds shows that pattern. Instead the lock acts like a
**low-pass filter** on the T1/T2 difference — it pushes
everything toward zero regardless of original sign.

Architecturally interpretable : locking the constellation
removes the single mechanism by which T1 and T2 could develop
differentially during Phase 2. Both Ph2 training loops now
converge on the same already-frozen constellation ; only the
non-mux modules (`PlasticityGate`, `AdaptiveCodebook`,
`CrossModalTransducer`, sensories, head) can differ between
T1 and T2, and they don't differ enough to cross the
pre-registered 0.05 threshold.

## Headline for the paper (Sprint 8)

B-3 :
- PASS in 4/4 worlds, under both regimes, 6×-22× threshold.
- Monotone growth with input complexity survives the lock.
- **Strongest architectural invariant confirmed by the benchmark.**

B-1 :
- Unconstrained : directionally falsified (T2 >= T1) in 3/4 worlds.
- Under lock : homogenised to zero in 3/4 worlds.
- No world-condition pair achieves the pre-registered 0.05 magnitude.
- **Pre-registered hypothesis is not recovered by the simple lock.**

B-2 :
- Unconstrained : positive sign 4/5 worlds, magnitude 3× to 50× under-threshold.
- Under lock (Gaussian) : flips to −0.0092.
- Estimator-limited (Kraskov) — to be revisited via nerve-wml#7 + MINE.

## Pre-registration fidelity

- No threshold changes.
- No metric-math changes.
- v0.3 verdicts (ADR-0004 / 0005 / 0008 / 0009) remain canonical.
- ADR-0010 (single-world lock verdict) remains canonical.
- This ADR strictly extends the experimental matrix ; nothing
  retroactive.

## Next steps

1. **Update paper §5.2** with this 4x2 matrix. Main claim updates
   from "directional recovery on Gaussian" (ADR-0010) to "lock is
   homogenising across 3/4 synthetic worlds and ineffective on the
   real signal" (ADR-0011).
2. **Close issue nerve-wml#4** : the mechanism is shipped,
   empirically tested on 4 worlds, and the verdict is
   publishable even if negative. The issue's open-until-cross-
   world-replication gate is satisfied. Follow-ups tracked on
   issue nerve-wml#5 (transducer lock compound) and a new
   issue for dose-response scanning.
3. **Paper claims** : do NOT promise Amedi recovery ; frame the
   lock as "a diagnostic that falsifies the single-parameter
   critical-period hypothesis for this architecture on these
   worlds."
4. **v0.2 paper work** : compound lock with
   `CrossModalTransducer` gating (nerve-wml#5) and
   `AdaptiveCodebook` freeze could produce a real B-1 recovery
   — declared as scope for paper v0.2, not v0.1.

## Implementation note : the isolated-worktree pattern

This grid ran from `~/Projets/bouba_sens_b1` (a `git worktree`
on branch `feat/b1-plasticity-recovery`) with a symlinked
`data/` dir. The parent clone `~/Projets/bouba_sens` was used
concurrently by another agent on `sprint7/critical-validation`.
Earlier attempts in this session had three grid instances
killed within ~5 min when the other agent's `uv sync`
reinstalled the CLI without the `--constellation-lock-after`
flag. The worktree pattern gives each agent its own `.venv`
resolution while sharing the underlying object store. This is
now the documented protocol for multi-agent Studio work ; it
should be lifted into a `docs/concurrency.md` guide in a
follow-up PR.
