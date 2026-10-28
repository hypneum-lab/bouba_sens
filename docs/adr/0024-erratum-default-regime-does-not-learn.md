# ADR-0024 — Erratum: with the default settings the network does not learn the task

**Status:** Accepted — PI decision 2026-10-26 (22:28). Erratum; no value,
metric, threshold or past verdict number is deleted or changed.
**Date:** 2026-10-26
**Evidence:** `docs/milestones/2026-10-26-code-collapse-diagnosis.md`
(PR #42), `reports/diag_code_collapse_*.json`, ADR-0022, ADR-0023,
`reports/v0.5_s14c_joint_aggregate.json`.

## Fact

With the default settings (argmax emission with the 1e-3 bridge, Adam
at lr 1e-3, `run_grid.sh` schedule: 200 Phase-1 steps and 100 Phase-2
steps, batch 16), **the network does not learn the classification
task**. Every number below is at the 4-class chance level (0.25) or at
ln 4:

- Phase-1 training loss stays at ln 4 = 1.386 for all 200 steps: mean
  1.385 (Gaussian) and 1.390 (mock) over the last 20 steps, seeds 0–1.
- On the intact networks, the full-network accuracy with all five
  modalities live is 0.270 (Gaussian) and 0.274 (mock) on the PR #40
  intact networks (ADR-0023).
- Mean Me1 is 0.271 / 0.288 on the PR #40 grids, 0.257 / 0.260 on the
  ADR-0023 confirmatory grids, and 0.251 (range 0.211–0.275) on the
  historical Sprint 14c joint grid.

The cause is in the diagnosis note. The argmax blocks the task gradient,
the only remaining path is a label-agnostic bridge of about 1e-6, and
Adam rescales it into drift that collapses each modality onto one or a
few codes.

## Scope: every published result of the repository

Every grid of the repository (Sprints 0–17, ADR-0003 to ADR-0023,
synthetic and real-bridge worlds, the Amedi dose-response curve, lock,
gating, τ and codebook-freeze scans, N12 and Q3/Q3+) was produced with
this emission (the only one the code has had) and with the
`STEPS_TRAIN=200` / `STEPS_LESION=100` schedule recorded in the ADRs.
The ADR-0001 spike, a separate script, is not covered. All of them describe a system that
**does not perform the task**:

- **B-1** (Me7 congenital gap): a difference of Me1 values that are all
  at chance. The Retract (ADR-0019) stands; the effect it retracted
  could not have arisen from task learning.
- **B-2** (Me3 MI migration): mutual information between fused codes and
  labels in a network whose code channel is collapsed and carries no
  label information. The Null stands; it is a null of an untrained
  regime, not evidence against MI migration in a learning agent.
- **B-3** (Me6): the v1 matrix is chance accuracies (ADR-0022); the v2
  asymmetry is the unequal informativeness of the inputs as read from
  encoder states that changed only by optimiser drift (ADR-0023).
- **Plasticity mechanisms** (lock, gating, codebook freeze, transducers):
  their reported effects are effects on the dynamics of this
  non-learning regime.

**Requalification.** All B-1 / B-2 / B-3 results are requalified as
*measured in a regime where the network does not learn the task*. They
are not evidence for or against cross-modal plasticity in an agent that
performs the task. Every number stays in place in its ADR, report and
paper section. Each status statement now carries this qualification.

## What follows

- `--emission straight-through` is added as an option, default
  unchanged (separate PR). Its pilot criterion is fixed before the run.
- Any new reading of B-1 / B-2 / B-3 needs a regime that learns, shown by
  a preregistered criterion.
- The TMLR submission stays blocked (it already was, ADR-0019).

## References

ADR-0019, ADR-0022, ADR-0023; PR #42; issue #22; PI decision 2026-10-26.
