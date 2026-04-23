# ADR-0014 — B-3 partition-control verdict (Sprint 9 gateway)

**Status:** Accepted
**Date:** 2026-04-24
**Sprint:** 9 (gateway before ADR-0012)
**Decision authority:** Issue #4 dichotomous interpretation framework, agreed in #3.

## Context

Issue #3 (closed 2026-04-22) established that the v0.4 real-ECG
grid fails the null-model partition control at the 22nd percentile
(n=9 random 3+2 partitions of the same 5 modalities). ADR-0009 was
retracted (commit `7f72d96`) accordingly. The §Next steps of the
retraction mandated re-running the same pipeline on two more grids
before Sprint 9 ADR-0012 acceptance, with a pre-committed
dichotomous interpretation framework recorded in issue #4.

This ADR records the verdict.

## Pipeline + grids

| Item | Value |
|------|-------|
| Pipeline | `scripts/critical_validation_pipeline.sh` (commit `858ce51`, branch `sprint9/critical-pipeline`) |
| Test | n=9 random 3+2 modality partitions vs the pre-registered perceptive/proprioceptive 3+2 split, on Me6 median |
| Acceptance | `passes_95pct` = `true` iff the pre-reg partition ranks > 95th percentile of the n=9 null distribution |
| Grids run | `runs/v04_studyforrest_real_grid` (#3), `runs/v04_studyforrest_grid` (mock), `runs/v03_xor_grid` (XOR cluster representative) |
| Host | Studio (MacStudio, arm64), commit `4869dcd`, 2026-04-23 23:34Z |

## Results

| Grid | Source ADR | B-3 raw median | Pre-reg rank | Percentile | `passes_95pct` |
|---|---|---:|---:|---:|---|
| ECG (real) | ADR-0009 (retracted) | 0.4453 | 2 / 9 | 22.2 % | **false** |
| Mock (Studyforrest AR(1)) | ADR-0008 | 0.2891 | 4 / 9 | 44.4 % | **false** |
| XOR (cluster repr.) | ADR-0005 | 0.1172 | 2 / 9 | 22.2 % | **false** |

**Verdict per the issue #4 dichotomous framework :** **3 / 3 grids FAIL the partition control.**

## Decision

The B-3 architectural-property narrative is **empirically dead** in
its current form. Across three signal classes spanning a 3.8×
range in raw B-3 magnitude — synthetic-cluster (XOR, 0.117),
AR(1)-mock (Studyforrest mock, 0.289), real biological (ECG,
0.445) — the pre-registered perceptive/proprioceptive 3+2
partition is **statistically indistinguishable from random 3+2
partitions of the same modalities** (rank 2-4 / 9, 22-44
percentile, never above the 95% threshold).

Concretely : whatever Me6 is measuring on these grids, it is **a
property of the modality-set's per-modality entropy distribution,
not of the cognitive partition structure** the framework claimed
to have validated.

## Implications for paper §8

1. **Withdraw the "B-3 monotone growth across substrates → architectural property" narrative.** This was the centrepiece of the cross-world claim in §8.2-§8.4 ; the partition-control evidence does not support the architectural reading.
2. **Reframe what B-3 actually measures.** Plausible candidates for the new framing :
   - *Entropy proxy*: B-3 magnitude tracks per-modality entropy ; useful as a dataset-richness diagnostic but not a cognitive-architecture invariant.
   - *Modality-count proxy*: B-3 may grow with the cardinality and dispersion of the modality set, independent of any partition.
   - *Pipeline-internal smoothing*: a feature of the Me6 estimator on small partitions, surfaced by the n=9 null study.
   The ADR-0014 verdict does not select among these — it only forecloses the "architectural property" reading.
3. **B-1 and B-2 status is unchanged from ADR-0009 retraction.** Bootstrap CI on B-1 still straddles 0 ; B-2 multi-estimator agreement still at noise floor on small probe batches. No new evidence from the partition control changes those.
4. **Sprint 9 ADR-0012 acceptance is unblocked but constrained.** ADR-0012 (real-5modal-Studyforrest-verdicts) can proceed, but its B-3 reporting MUST include a partition-control column as part of its acceptance criteria. The 5-modal real Studyforrest is the next opportunity to test whether *biological multi-modality* (rather than physiological single-channel ECG) recovers a partition-distinguishable signal — but the **prior** is now that it will not.

## What this does NOT change

- The raw Me6 magnitude observations across the 5 worlds remain valid and reproducible (SHA-pinned aggregates exist).
- The Me1/Me2/Me3 estimators are independent of the partition-control critique.
- The pre-registered 3+2 partition (perceptive vs proprioceptive) remains a meaningful **conceptual** distinction in the cognitive-architecture vocabulary ; it just isn't an invariant we can validate on these grids with this estimator.

## Artefacts

Local copy of the three pipeline output trees (mock + XOR ; ECG was already on Studio per #3) :

- `reports/v0.4-mock-partition-control_critical_validation/`
- `reports/v0.3-xor-partition-control_critical_validation/`

Each contains `null_b3_analysis.json` (verdict), `prereg_aggregate.json`, `me7_bootstrap.json`, `mi_estimator_comparison.json`, and a `MANIFEST.md` with SHA-256s for reproducibility.

## Cross-references

- Issue #3 — original null-model finding on real ECG (closed 2026-04-22)
- Issue #4 — gateway tracking issue, dichotomous framework (this ADR closes it)
- ADR-0009 — retracted ECG verdict (commit `7f72d96`)
- ADR-0008 — mock verdict (now superseded by this ADR for B-3 claims)
- ADR-0005 — cross-world cluster verdict (now superseded by this ADR for B-3 claims)
- Branch `sprint9/critical-pipeline` commit `858ce51` — the canonical pipeline
- Pipeline run host : Studio (`MacStudio-de-MonsieurB.local`), bouba_sens commit `4869dcd`, 2026-04-23 23:34Z
