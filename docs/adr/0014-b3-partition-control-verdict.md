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

### v1 — original n=9 with-replacement sampling (commit `c2870d8`)

| Grid | Source ADR | B-3 raw median | Pre-reg rank | Percentile | `passes_95pct` |
|---|---|---:|---:|---:|---|
| ECG (real) | ADR-0009 (retracted) | 0.4453 | 2 / 9 | 22.2 % | false |
| Mock (Studyforrest AR(1)) | ADR-0008 | 0.2891 | 4 / 9 | 44.4 % | false |
| XOR (cluster repr.) | ADR-0005 | 0.1172 | 2 / 9 | 22.2 % | false |

**Verdict per the issue #4 dichotomous framework :** **3 / 3 grids FAIL.**

### v2 — exhaustive n=9 unique partitions (2026-04-24, this commit)

The v1 sampling used `generate_random_3_2_partitions(n=k+1, seed=0)` with the default `unique=False`, which samples **with replacement** from the 9 distinct 3+2 alternatives. Only **5 of 9 unique partitions** were actually tested (the other 4 never appeared in any of the 9 indices). This was a methodological hole. Patched to `unique=True`, re-ran on the 3 original grids plus a 4th (Sinusoid cluster representative).

| Grid | Source ADR | B-3 raw median | Pre-reg rank | Percentile | `passes_95pct` |
|---|---|---:|---:|---:|---|
| ECG (real) | ADR-0009 (retracted) | 0.4141 | 5 / 9 | 55.6 % | false |
| Mock (Studyforrest AR(1)) | ADR-0008 | 0.2891 | 5 / 9 | 55.6 % | false |
| XOR (cluster repr.) | ADR-0005 | 0.1172 | 3 / 9 | 33.3 % | false |
| **Sinusoid** (cluster) | ADR-0005 | 0.1328 | **1 / 9** | **11.1 %** | false |

**Verdict v2 :** **4 / 4 grids FAIL.** Pre-reg never exceeds the median on any grid ; on Sinusoid, pre-reg ranks **dead last** among the 9 alternatives.

### Interpretation update — methodological caveat (Axe 2 positive-control finding)

A constructed positive-control matrix where the partition structure is maximally salient (signal concentrated on all 6 pre-reg cross pairs, identical magnitude) gives `pre-reg ≤ all 9 random alternatives` because every random 3+2 partition's cross block intersects the pre-reg's. A second design (single hot pair audio↔gravity) gives pre-reg at 44.4 %, tied with 5 of 9 alternatives that catch the same pair in their cross block.

**No matrix on 5 modalities, with the current max-statistic, can put pre-reg strictly above the 95th percentile of the 9 unique alternatives.** The combinatorics of 3+2 partitions of 5 modalities guarantee shared cross pairs. The `passes_95pct` flag is **theoretically unreachable** under this design.

This means the v2 verdict has a layered reading :

1. **Empirical (robust)** — pre-reg never beats the median across 4 structurally divergent grids spanning a 4× B-3 magnitude range. Even relaxing the threshold to "median or above", pre-reg only meets it on ECG and Mock (ties at 55.6 %), never on cluster grids. **The pre-registered partition does not capture more partition signal than a random alternative would.**
2. **Methodological (caveat)** — the `passes_95pct` flag was the wrong success criterion for this design. A more diagnostic statistic on this small partition space is the **rank distribution across grids** : pre-reg ranks {5, 5, 3, 1} on {ECG, Mock, XOR, Sinu}, mean rank 3.5, median rank 4. A truly architectural partition would give ranks ≥ 8 across grids. The observed pattern is closer to "pre-reg is statistically equivalent to a random partition" than to "pre-reg captures partition structure but is below the binary detection threshold".

Both readings agree that the architectural-property claim is unsupported. The methodological caveat is documented to prevent a future reader (or reviewer) from incorrectly inferring that the test was ever capable of "passing" in the binary sense.

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
2. **Reframe what B-3 actually measures — Axe 3 numerical experiment (2026-04-24).** Three hypotheses were on the table : entropy proxy, modality-count proxy, pipeline-internal smoothing. A controlled numerical sweep distinguishes them :

   | Sweep | Setup | Me6 raw range observed |
   |---|---|---|
   | A — entropy | k=5, sigma ∈ {0.01, 0.05, 0.1, 0.3, 1.0}, 50 trials each | Me6 ≈ 2.6 × sigma : `[0.027, 0.138, 0.266, 0.814, 2.549]` |
   | B — modality-count | sigma=0.1 fixed, k ∈ {3, 5, 7, 9, 11, 15}, 50 trials each | Me6 grows logarithmically : `[0.192, 0.266, 0.309, 0.362, 0.370, 0.389]` |

   **Decisive observation** : all 4 grids in this verdict have **k=5 modalities (fixed)**. The empirical B-3 range observed across them — 0.117 (XOR) → 0.133 (Sinu) → 0.289 (Mock) → 0.414 (ECG), spanning 3.5× — therefore **cannot** come from modality-count (constant) and **must** come from per-cell entropy variation across data sources. **Hypothesis A wins for the empirical pattern.** B-3 magnitude is dominated by per-cell entropy/variance, not by partition structure or modality count.

   Concretely : the original ADR-0009 7× → 15.6× → 22.3× narrative is correctly reframed as *"per-modality entropy on real ECG > AR(1) mock > synthetic cluster, by a factor that exactly matches what a noise-level sweep predicts"*. This is a positive scientific finding (entropy is a real property of the data), but it is **not** an architectural invariant of the cognitive system the framework claims to model.

3. **B-1 and B-2 status is unchanged from ADR-0009 retraction.** Bootstrap CI on B-1 still straddles 0 ; B-2 multi-estimator agreement still at noise floor on small probe batches. No new evidence from the partition control changes those.
4. **Sprint 9 ADR-0012 acceptance is unblocked but constrained.** ADR-0012 (real-5modal-Studyforrest-verdicts) can proceed, but its B-3 reporting MUST include : (a) partition-control column per the dichotomous framework, (b) per-modality entropy column (so the entropy-proxy hypothesis can be quantified for that grid), (c) explicit caveat that `passes_95pct` is theoretically unreachable on 5 modalities.
5. **Methodology fix (Axe 5)** : `scripts/run_grid_with_partition_control.sh` (commit `87cc48c`) now wraps `run_grid.sh` + `critical_validation_pipeline.sh` so future grids cannot ship without partition control attached.
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
