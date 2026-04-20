# ADR-0007 — Biological-adjacent bridge stub (StudyforrestWorld)

**Status:** Accepted
**Date:** 2026-04-20
**Sprint:** 7 (Task 7.6)

## Context

ADR-0005 closed Sprint 6 with B-3 PASS 3/3 on synthetic worlds (Gaussian,
XOR, Sinusoid). A reviewer — and the maintainer themselves — immediately
asked: these three worlds may be samples of the same cluster in world-
space; extrapolating B-3 to biological plasticity requires data that
lives outside that cluster. Task 7.6 lands the minimal infrastructure
needed to address this critique in a future sprint.

## Scope — what this is and what it is NOT

**IS:**
- A `StudyforrestWorld(WorldSimulator)` class that implements the same
  `sample()` / `modality_dims()` contract as the 3 synthetic worlds.
- A `mock=True` mode (default when no `data_dir` is passed) that
  generates data with **deliberately biological-adjacent statistics**:
  AR(1) scene latent, cross-modal audio/vision correlation, temporal
  autocorrelation on labels. The mock is self-contained (no network).
- A `data_dir` mode that loads pre-fetched tensors from disk — the
  on-ramp to real Studyforrest film + fMRI data when those tensors
  are extracted in a future sprint.

**IS NOT:**
- A scientific replication of biological cross-modal plasticity.
- A full integration of Studyforrest fMRI + audio + visual features.
- A claim that the mock mode captures all dimensions of biological
  data — it captures just enough to move the world-complexity audit
  profile qualitatively (see numeric gap in "Evidence" below).

## Evidence that the bridge changes the audit profile

Run `compute_world_profile` on a 512-sample batch from each world:

| Metric | gaussian | xor | sinusoid | studyforrest_mock | Gap to closest synthetic |
|--------|---------:|----:|---------:|------------------:|------------------------:|
| `intrinsic_dim_pca_audio` | 30 | 30 | 29 | **4** | 7x compression |
| `intrinsic_dim_pca_vision` | 30 | 30 | 29 | **4** | 7x compression |
| `mi_pairwise` | 0.015 | 0.023 | 0.018 | **0.392** | 17x stronger cross-modal MI |
| `temporal_autocorr` | 0.00 | 0.07 | -0.02 | **0.82** | structure absent from synthetics |
| `label_conditional_entropy` | 1.93 | 0.98 | 1.82 | 1.73 | within cluster |
| `linear_separability` | 0.89 | 0.49 | 0.87 | 0.47 | near XOR |

Three metrics (audio rank, vision rank, cross-modal MI, temporal
autocorr) place the mock clearly **outside** the synthetic cluster.
Two metrics (conditional entropy, linear separability) place it
**within** the synthetic spread. Partial external-validity gain.

## Limitations recorded on the bridge itself

1. Tactile, gravity, force modalities are **zeroed** — no embodied
   signal yet. A future sprint can derive tactile/proprioceptive
   surrogates from film motion annotations if needed.
2. Mock mode is a **statistical surrogate**, not biological data.
   Running `run_grid.sh WORLD=studyforrest` on mock is useful for
   smoke-testing the pipeline but cannot replace a real data ingest.
3. No fMRI signal is consumed in this bridge. The flagship Amedi 2007
   claim (occipital cortex takes over when visual input is absent)
   requires brain data; we only consume behavioural-level features.

## Decision

Ship the stub as-is. The mock statistics are calibrated to pass the
unit tests that assert non-trivial cross-modal MI and temporal
autocorrelation, which is the **minimum bar** for the bridge to be
honest about its contract. Any data-wiring improvements belong to
Sprint 9+.

## Non-decisions (scope discipline)

- We do NOT re-run the 150-cell grid on StudyforrestWorld in this
  sprint. That belongs to Sprint 8 after the pre-registration amendment
  is drafted — dropping a new world through the verdict pipeline without
  an OSF amendment would be p-hacking.
- We do NOT touch the B-1 / B-2 / B-3 thresholds. The bridge exists to
  enable future out-of-cluster tests, not to shift the goalposts on
  existing ones.

## Next steps (Sprint 8+)

1. Sprint 8 paper draft consumes this ADR in its "Limitations: external
   validity" paragraph, referencing both ADR-0005 (synthetic cluster
   verdicts) and ADR-0007 (bridge stub).
2. Sprint 9+ extracts real Studyforrest tensors (1 hour of film,
   ~60 GB pre-processed → ~50 MB of features) and runs a 5-seed grid
   with `WORLD=studyforrest data_dir=...`. Pre-registration amendment
   must be filed on OSF first.
3. Optional: extend the mock's temporal coherence to include T1/T2
   lesion scenarios so the bridge becomes a real stand-in for
   biological validation on CI (zero-network, seconds to run).
