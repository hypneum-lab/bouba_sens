# bouba_sens — Sprint 7 Implementation Plan (critical validation of v0.3 findings)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`.

**Goal.** Stress-test the three headline findings from v0.2 / v0.3 against the four critical objections a rigorous reviewer would raise before the paper draft (Sprint 8) is written. No threshold changes, no metric-math changes — only null-model controls, bootstrap IC, and estimator-robustness checks.

**Tech Stack:** unchanged (Python 3.14, uv, PyTorch ≥ 2.5, scipy, sklearn, hydra, plotly, jinja2, pyarrow).

**Parent spec:** `docs/superpowers/specs/2026-04-20-bouba-sens-design.md`. ADR-0003, ADR-0004, ADR-0005 are the inherited context.

**Sprint 7 scope:** Tasks 7.1 → 7.4.

**Compute target:** Studio for Tasks 7.1 + 7.2 (grid reruns). GrosMac for Tasks 7.3 + 7.4 (code, estimators, ADR).

---

## Why this sprint exists

The v0.3 release claims three empirical findings :

- **F1 — B-3 world-agnostic PASS** on 3 synthetic worlds at ~7-8× threshold.
- **F2 — B-1 directionality flips on Sinusoid** vs Gaussian / XOR.
- **F3 — B-2 MI migration decays** Gaussian > XOR > Sinusoid.

Each has a structural objection that has not been closed :

- F1 may measure "size-3 vs size-2 partition dynamics" (tautology), not perceptive/proprioceptive asymmetry.
- F2 effect sizes (|0.006-0.013|) are below detection noise; the sign flip could be pure sampling.
- F3 decay may be Kraskov-k-NN-specific rather than a true MI migration pattern.

Without these controls, the paper is unreviewable at NeurIPS / TMLR / D&B standards.

---

## File structure touched in Sprint 7

```
bouba_sens/
├── scripts/
│   ├── run_null_b3.sh                   [Task 7.1]  new: random-partition grids
│   └── bootstrap_me7.py                 [Task 7.2]  new: bootstrap IC on Me7 median
├── src/bouba_sens/metrics/
│   ├── mi_migration.py                  [Task 7.3]  add binning + MINE estimators
│   └── asymmetry.py                     [Task 7.1]  support arbitrary partition spec
├── reports/
│   └── v0.3_critical_validation/        [all tasks] new: aggregated test artefacts
├── docs/adr/
│   └── 0006-critical-validation.md      [Task 7.4]  pass/fail verdicts per test
└── CHANGELOG.md                         [Task 7.4]  v0.4.0 entry
```

---

## Tasks

### Task 7.1 — B-3 null-model control (random-partition grid on GaussianWorld)

**Objection.** Me6 max-abs off-diag is computed over a 5×5 query×lesion matrix. The pre-registered partition (perceptive={audio,vision,tactile} vs proprioceptive={gravity,force}) is a 3+2 split. Any 3+2 split of 5 modalities might show comparable asymmetry — in which case B-3 passes for a structural reason (partition size) rather than a cognitive one.

**Test.**
- [ ] Generate **10 random 3+2 permutations** of the modality set {audio, vision, tactile, gravity, force}, excluding the pre-registered partition.
- [ ] Re-run the 150-cell grid on GaussianWorld for each permutation (same STEPS_TRAIN=200, STEPS_LESION=100, 5 seeds, reusing the same phase1 pretrain caches where possible).
- [ ] Aggregate Me6 max-abs off-diag for each permutation's perf matrix.
- [ ] Compute the empirical distribution of Me6 over 10 random partitions.

**Acceptance.**
- The pre-registered partition's Me6 median must be at the **≥ 95th percentile** of the random-partition distribution.
- If it falls within the bulk of the distribution, **B-3 is downgraded** to "partition-size effect" and the paper narrative is reframed as "a measurable 5-modality lesion asymmetry, direction-agnostic" rather than "perceptive/proprioceptive asymmetry".

**Wall-clock estimate.** 10 × ~17 min = ~3 h on Studio. Phase1 pretrain can be reused across permutations, reducing to ~1 h 45 min.

### Task 7.2 — B-1 bootstrap IC on Me7 median (per-world)

**Objection.** The reported Me7 medians (-0.006, -0.006, +0.013) are all 5-10× below the 0.05 threshold. With 5 seeds × 15 (modality × SNR) pairs = 75 points per world, seed noise may dominate.

**Test.**
- [ ] Load `reports/v0.2_aggregate_{gaussian,xor,sinusoid}.json`.
- [ ] For each world, extract the 75 raw Me7 paired-values (congenital – late_acquired).
- [ ] Bootstrap 10_000 resamples per world, take the median of each.
- [ ] Derive 95 % CI from the 2.5 / 97.5 percentiles.

**Acceptance.**
- For the sign-flip claim to survive, Sinusoid 95 % CI must not overlap the Gaussian 95 % CI, and at least one CI must not cross 0.
- If all 3 CIs straddle 0, **F2 is downgraded** to "no significant topology-dependent B-1 signal at v0.2 grid scale". ADR-0006 records this as an honest null result, not a finding.

**Wall-clock estimate.** ~5 min Python on GrosMac.

### Task 7.3 — Me3 MI estimator robustness (cross-check Gaussian > XOR > Sinusoid decay)

**Objection.** `me3_delta` currently uses `sklearn`'s Kraskov k-NN MI estimator, known to be noisy and biased at high ambient dimension. The observed Gaussian 0.028 > XOR 0.004 > Sinusoid 0.002 decay may be estimator-specific.

**Test.**
- [ ] Extend `src/bouba_sens/metrics/mi_migration.py` with two alternative estimators :
  - `me3_delta_binning` : discretise codes into 32 bins per dim, use plug-in MI.
  - `me3_delta_mine` : Mutual Information Neural Estimator (Belghazi 2018), small 3-layer critic, 1000 training steps on (codes, labels) pairs.
- [ ] Re-aggregate the v0.2 grids for all 3 worlds using each alternative estimator.
- [ ] Compare the qualitative ordering across the three estimators.

**Acceptance.**
- The Gaussian > XOR > Sinusoid ordering must hold under **at least one alternative estimator** in addition to Kraskov.
- If the ordering flips under both alternative estimators, **F3 is downgraded** to "Kraskov-specific artefact". ADR-0006 records the disagreement and recommends that Paper 1 §Methods describe the estimator sensitivity as a known limitation.

**Wall-clock estimate.** ~2 h implementation + ~30 min re-aggregation.

### Task 7.4 — ADR-0006 + paper-ready narrative reframing

- [ ] `docs/adr/0006-critical-validation.md` : for each of F1, F2, F3, record the test outcome and the revised narrative.
- [ ] Update `README.md` Findings section — tighten or downgrade claims per the test outcomes.
- [ ] Update `docs/superpowers/specs/2026-04-20-bouba-sens-design.md` §Results to match.
- [ ] Paper 1 v0.1 draft (Sprint 8) builds figures on the **validated subset only**. Downgraded findings are discussed in §Limitations, not headlined.
- [ ] Bump `src/bouba_sens/_version.py` to `0.4.0` (or `0.3.1` if no findings are downgraded) and tag.

---

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| R-sprint7-1: 10 random partitions on Studio (~3 h) clashes with other training workloads | Schedule during off-hours; reuse phase1 pretrain caches. |
| R-sprint7-2: MINE critic diverges at small batch size | Fallback to binning estimator; document if MINE unstable. |
| R-sprint7-3: bootstrap IC disagrees across seeds (multi-modal posterior) | Report full distribution in ADR-0006, not just CI. |
| R-sprint7-4: all three tests downgrade findings | Honest null results **are** the scientific contribution. Paper still gets written, narrative pivots. |

---

## Exit criteria

Sprint 7 closes when :

1. All three critical tests completed + aggregated artefacts in `reports/v0.3_critical_validation/`.
2. ADR-0006 committed with pass / fail / downgrade verdicts per test.
3. README narrative updated to match the validated subset.
4. CHANGELOG v0.4.0 entry.
5. Tag `v0.4.0` pushed (or `v0.3.1` if no downgrade).

**Honest posture.** If Tasks 7.1-7.3 downgrade any finding, Sprint 7 succeeds — the paper becomes stronger, not weaker, because reviewers can no longer object to what we have already tested and reported.
