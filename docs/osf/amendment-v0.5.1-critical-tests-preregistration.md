# OSF pre-registration amendment — bouba_sens v0.5.1 (critical-tests acceptance criteria)

**Status:** Draft, ready to file
**Parent pre-registration:** `dream-of-kiki` OSF DOI `10.17605/OSF.IO/Q6JYN`
**Parent amendment:** `amendment-v0.5-studyforrest-5modal.md` (filed 2026-04-21, Task 9.5)
**Amendment tag on file:** `bouba_sens/v0.5.1-critical-tests`
**Amendment date:** 2026-04-21
**Filing order:** must land on OSF **before** Task 9.6 grid runs start, so the critical-test acceptance criteria are pre-registered rather than post-hoc justifications.

## What is being amended

The v0.5 amendment (2026-04-21) added a `StudyforrestRealWorld` bridge with 5 real biological modalities and pre-registered the grid schedule (no-lock + lock=200 at `STEPS_TRAIN=200`, `STEPS_LESION=100`). Its decision rule for paper v0.2 is:

| B-3 on 5-modal real | Paper v0.2 claim |
|---------------------|------------------|
| PASS at ≥ 10× threshold | "B-3 is an architectural invariant..." (strong) |
| PASS at 1–10× threshold | "...attenuated under biological input complexity." (moderate) |
| FAIL | "synthetic-cluster artefact..." (retraction) |

The Sprint 8 retrospective (`docs/adr/0006-critical-validation.md`, 2026-04-21, `v0.4.0` tag) showed that the v0.2 Gaussian grid **passed B-3 at 7.4× threshold** and was subsequently **downgraded to the 12.5th percentile of an n=8 null-model partition distribution**. The "multiple-of-threshold" criterion was demonstrably insufficient to rule out a partition tautology; it would have promoted a tautological finding to "strong" under the v0.5 decision rule.

This v0.5.1 amendment **tightens the acceptance criteria** of the three invariants B-1, B-2, B-3 for the Sprint 9 Task 9.6 grids (and all subsequent grids) so that a "strong" or "moderate" publication claim requires *both* the pre-registered effect-size threshold *and* the matching critical-validation check. Thresholds, metric implementations, grid structure, and world definitions are **unchanged**.

## What is amended

### B-1 — Congenital gap (Me7)

**v0.5 acceptance (inherited).** `median(me7) > 0.05`.

**v0.5.1 acceptance (new, required for "PASS"):**

1. `median(me7) > 0.05`, AND
2. the 95 % CI on the median (scipy `bootstrap`, percentile method, `n_resamples = 10_000`, `random_state = 0`) does **not** contain 0.

Rationale: ADR-0006 §5.2 showed that the v0.2 `median(me7)` values (Gaussian −0.006, XOR −0.006, Sinusoid +0.013) all produced 95 % CIs straddling zero — an effect size 5–10× below the threshold is indistinguishable from sampling noise without an explicit interval.

Implementation: `scripts/bootstrap_me7.py` (committed `b452f48`, fixed to read `invariants.b1.raw_me7_pairs` nested path). The pipeline `scripts/critical_validation_pipeline.sh` (commit `858ce51`) runs this test automatically.

### B-2 — MI migration (Me3 delta)

**v0.5 acceptance (inherited).** `median(me3_delta) > 0.10` bit, computed with the sklearn Kraskov $k$-NN estimator at `k=3`, `batch=16`.

**v0.5.1 acceptance (new, required for "PASS"):**

1. `median(me3_delta) > 0.10` under Kraskov, AND
2. `median(me3_delta) > 0.05` (half the pre-reg threshold to absorb estimator variance) under **at least one** of the two alternative estimators:
   - quantile-binning with Gaussian-NB fallback for $d > 1$ (`me3_delta_binning`, 16 bins / dim, committed `ba02431`)
   - DV-bound MINE neural estimator (`me3_delta_mine`, 300 epochs, seed 0, committed `ba02431`)

Rationale: ADR-0006 §5.3 showed that at `batch=16` the Kraskov estimator's bias dominates any real effect of order $10^{-1}$; all three estimators returned median $\approx 0$ on the v0.2 grid. The amendment v0.5 inherits the n=16 probe batch; v0.5.1 does not change that, but it requires a second estimator to confirm the Kraskov reading is not spurious.

**Note on probe batch.** bouba_sens `v0.5.0` (commit `4869dcd`) introduced a `probe_batch_size` kwarg on `AdaptationLoop.lesion_phase` with default 128. If Task 9.6 grids are run with the default (recommended) rather than the legacy 16, the Kraskov bias is materially reduced and the two-estimator check acts as a standard robustness gate.

### B-3 — Perceptive / proprioceptive asymmetry (Me6)

**v0.5 acceptance (inherited).** `median(me6_max_abs_off_diag) > 0.02`, with the pre-registered perceptive/proprioceptive 3+2 partition $\{\text{audio}, \text{vision}, \text{tactile}\}$ vs $\{\text{gravity}, \text{force}\}$.

**v0.5.1 acceptance (new, required for "PASS"):**

1. `median(me6_max_abs_off_diag) > 0.02` under the pre-registered partition, AND
2. the pre-registered partition's median ranks at the **≥ 95th percentile** of the empirical null distribution produced by applying the partition-aware statistic `me6_max_abs_off_diag_partitioned` to the nine distinct non-pre-reg 3+2 partitions of the five modalities, computed by `scripts/run_null_b3.sh` / `scripts/critical_validation_pipeline.sh` with `PARTITION_SEED=0, PARTITION_INDEX ∈ {0, …, 8}`.

Rationale: ADR-0006 §5.1 and §7.1 showed that the v0.2 Gaussian pre-reg partition passed Me6 at 7.4× threshold but ranked at the **12.5th percentile** of the n=8 non-pre-reg null distribution (n=9 in a subsequent full run, pre-reg rank 0/9). Without this rank criterion, B-3 "PASS at 10× threshold" is indistinguishable from a partition-size tautology.

## Revised decision rule for paper v0.2

| B-3 PASS at ≥ 10× threshold + null-rank ≥ 95th pctl | "B-3 is a cross-partition architectural invariant..." (strong) |
|---|---|
| B-3 PASS at ≥ 10× threshold but null-rank < 95th pctl | "B-3 magnitude holds but pre-reg partition is not distinguishable from random 3+2 partitions — see §Limitations." (moderate, mandatory caveat) |
| B-3 PASS at 1–10× threshold + null-rank ≥ 95th pctl | "B-3 attenuated but distinguishable..." (moderate-plus) |
| B-3 PASS at 1–10× threshold + null-rank < 95th pctl | "B-3 attenuated and indistinguishable from null — see §Limitations." (moderate-minus) |
| B-3 FAIL | "synthetic-cluster artefact; v0.1 headline retracted." |

B-1 and B-2 follow the two-part acceptance above: PASS iff both the magnitude and the robustness check pass; FAIL otherwise.

## What is NOT being amended

- Thresholds `0.05 / 0.10 / 0.02` (frozen since spec §1.2).
- Metric implementations `me1_accuracy`, `me2_recovery_auc`, `me3_delta` (Kraskov), `me6_asymmetry`, `me6_max_abs_off_diag`, `me7_congenital_gap`, `me9_bootstrap`.
- 5-seed × 5-modality × 2-timing × 3-SNR = 150-cell grid structure.
- Modality mapping in the v0.5 amendment (audio / vision / tactile / gravity / force → Studyforrest sources).
- Grid schedule (no-lock + lock=200) and 40 min parallel timeline.

## Release note

Task 9.8 in the Sprint 9 plan says *"v0.5.0 tag + release"*. `v0.5.0` is already live on the repository (`chore(release): v0.5.0 Sprint 8 close`, commit `93133b9`, 2026-04-21 morning). The Studyforrest 5-modal release therefore bumps to **`v0.6.0`** (minor, narrative continues — no breaking API change).

## Timeline

| Step | ETA |
|------|-----|
| File this document on OSF | same-day, **before** Task 9.6 |
| Task 9.6 grid runs | same-day (~40 min parallel, after filing) |
| `scripts/critical_validation_pipeline.sh` on each of the two aggregates | same-day (~5 min each) |
| ADR-0012 (task 9.7) consumes both grids + the two pipeline outputs | same-day |
| `v0.6.0` tag + release | same-day |

Grid runs strictly post-filing, critical-test outputs integrated into ADR-0012.

## Cross-references

- Sprint 8 retrospective: `docs/adr/0006-critical-validation.md`
- Pipeline script: `scripts/critical_validation_pipeline.sh` (commit `858ce51`, branch `sprint9/critical-pipeline`)
- Paper methodology section: `papers/paper1/main.md` §6 (branch `sprint8/paper-v0.1`)
- Coordination issue: [hypneum-lab/bouba_sens#2](https://github.com/hypneum-lab/bouba_sens/issues/2)
- Parent amendment: `docs/osf/amendment-v0.5-studyforrest-5modal.md` (commit `012ee7d`)

## Filing instructions

1. Upload this document to the OSF project page for `dream-of-kiki`, alongside the parent v0.5 amendment, under the `bouba_sens` namespace.
2. Tag the upload `bouba_sens/v0.5.1-critical-tests`.
3. Commit the filed timestamp back to `docs/osf/amendment-v0.5.1-critical-tests-preregistration.md` header before Task 9.6 launches.
