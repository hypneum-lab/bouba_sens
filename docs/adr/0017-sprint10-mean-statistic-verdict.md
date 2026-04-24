# ADR-0017 — Sprint 10 mean-statistic verdict (negative on real data)

**Status:** Accepted — **negative result**
**Date:** 2026-04-24
**Sprint:** 10
**Companion to:** ADR-0014 (verdict), ADR-0015 (lemma), ADR-0016 (design).

## Context

ADR-0016 prescribed Candidate B (mean statistic over cross pairs)
as the recommended replacement for the structurally-broken
max-statistic partition test (ADR-0015 lemma). Sprint 10
implementation:

- `me6_mean_off_diag_partitioned` added to
  `src/bouba_sens/metrics/asymmetry.py` (Studio working tree;
  upstream merge to main pending the broader sprint9/critical-pipeline
  branch landing — see "Caveats" below).
- 6 unit tests in `tests/unit/test_me6_mean.py`, all green.
- Positive-control validation against three synthetic matrix
  designs (ADR-0016 acceptance criteria).
- Empirical re-aggregation of 4 v0.X exhaustive grids under the
  new statistic (no cell re-runs needed; per_query_me1.json
  re-read with the new aggregation).

## Positive-control validation (ADR-0016 §Acceptance step 5)

| Design | Description | mean-stat midrank pctl | passes > 95 % ? |
|---|---|---:|---|
| Uniform cross (6/6 pre-reg cross pairs at value 1) | dense partition signal | **100.0 %** | ✅ |
| Single hot pair (audio↔gravity at 1) | sparse signal | 72.2 % | ❌ |
| Sparse block (3 of 6 pre-reg cross pairs at 1) | mid-density signal | 94.4 % | ❌ |

Mean-stat **strictly improves on max-stat** under dense signals
(Design 1: 100 % vs max-stat ceiling 72.2 %), is **equivalent**
under fully sparse signals (Design 2: same 72.2 %), and
**under-shoots 95 % marginally** under mid-density signals
(Design 3: 94.4 %).

Conclusion at this stage: mean-stat is a **conditional**
improvement — useful when the partition signal is dense, no help
when sparse. ADR-0016 §Acceptance was provisionally satisfied (the
test "can reach > 95 %") but the verdict on real data was deferred
to the empirical re-aggregation below.

## Empirical re-aggregation on the 4 v0.X grids

Per (seed, timing, snr) trio, the 5×5 perf matrix `M[i, j]` is
built from `per_query_me1.json` — row i = lesioned modality, col
j = queried modality. Me6 is computed per matrix (median across
30 trios per grid), then mid-rank percentile vs the 9 unique
3+2 alternatives.

| Grid | n_trios | mean pre-reg | max pre-reg | **mean midrank** | max midrank | Δ (mean − max) |
|---|---:|---:|---:|---:|---:|---:|
| ECG  | 30 | 0.2174 | 0.4141 | **22.2 %** | 61.1 % | **−38.9 %** |
| Mock | 30 | 0.1354 | 0.2891 | **11.1 %** | 72.2 % | **−61.1 %** |
| XOR  | 30 | 0.0625 | 0.1172 | **61.1 %** | 55.6 % | +5.6 % |
| Sinu | 30 | 0.0677 | 0.1328 | **22.2 %** | 44.4 % | −22.2 % |
| **Grand mean** | | | | **29.2 %** | **58.3 %** | **−29.2 %** |

**The mean statistic performs substantially WORSE than the max
statistic on real data, not better.** This contradicts the
positive-control optimism and is the central finding of Sprint 10.

### Interpretation — why mean fails on real data

The positive-control "uniform cross" design has the partition signal
spread evenly across all 6 pre-reg cross pairs, which is the
mean's best case. The 4 empirical grids do **not** look like that:
the relative collapse of pre-reg under mean-stat (esp. Mock −61 %)
suggests the actual partition signal — to the extent it exists —
is concentrated in a small subset of cross pairs (sparse-signal
regime). In this regime, the mean dilutes the signal across the
6 cross pairs, while the max picks up the dominant pair directly.

This pattern itself is informative :

- *Synthetic XOR* (the only grid where mean > max, +5.6 %) likely
  has a more uniform per-pair asymmetry distribution, consistent
  with the XOR world's symmetric construction.
- *Biological / mock* grids (ECG, Mock, Sinu) appear to have
  sparse partition signals — at most 1–2 cross pairs carry the
  asymmetry magnitude, not 6.

The asymmetry between synthetic and biological worlds in this
regard is itself a finding worth documenting in §3.3 of the paper.

## Decision

**Candidate B (mean statistic) is REJECTED as the primary
replacement for the partition control test on real bouba_sens
data.** The theoretical advantage in dense-signal regime does not
transfer to the empirical sparse-signal regime of biological
grids. Continuing with mean-stat as the headline test would
*worsen* pre-reg's apparent rank, not improve it.

Per ADR-0016's escalation clause, **promote Candidate C
(permutation test on modality labels) to Sprint 11 priority**.
Candidate C does not depend on the partition signal being dense
or sparse — it tests exchangeability of the modality labels
themselves, which is a more fundamental hypothesis.

## Implications for paper §3.3 / §4.4

The reframing in commit `6599c99` (entropy-proxy reading of B-3)
**stays correct** — this ADR doesn't restore the architectural
claim. What it adds is a finer methodological observation :

- Adding mean-stat as a secondary diagnostic remains useful (it
  gives a pure positive control that the pipeline machinery
  works), but the **primary test** for partition discrimination
  must come from a fundamentally different statistic family
  (permutation, not partition-aware-aggregate).

The paper should mention the negative result on mean-stat as a
"we tried this and it doesn't work for our data regime"
methodological note, with a forward pointer to the Sprint 11
permutation test results (when those land).

## Caveats / debt

1. **Upstream merge to main pending.** The
   `me6_mean_off_diag_partitioned` helper, `partitions.py`,
   `aggregate_grid.py --partition-prereg`, and
   `critical_validation_pipeline.sh` all live on either
   sprint9/critical-pipeline (commit `858ce51` and ancestors)
   or in Studio's working tree. This ADR records the *empirical
   verdict* from running them, but landing them on main is
   prerequisite to anyone else reproducing. Recommended: a
   minimal-PR cherry-pick of just (a) `partitions.py`,
   (b) `me6_max_abs_off_diag_partitioned` +
   `me6_mean_off_diag_partitioned`, (c) the
   `aggregate_grid.py --partition-prereg --statistic` CLI flags,
   (d) `critical_validation_pipeline.sh`. Skip the mass deletions
   on the divergent branch.

2. **OSF amendment NOT filed.** ADR-0016 §Acceptance required an
   OSF amendment `docs/osf/amendment-v0.6-mean-statistic.md` before
   Sprint 10 release. Since Sprint 10 *rejects* the mean statistic
   as primary, the amendment should instead document the
   *attempt and its outcome* rather than the proposed adoption.
   Draft template stub created at
   `docs/osf/amendment-v0.6-mean-statistic-attempt.md` (TODO: fill
   in pre-submission narrative + the verdict from this ADR).

3. **No `--statistic` CLI flag landed.** The empirical re-aggregation
   was done via `/tmp/me6_mean_4grids_v2.py` — a one-shot analyzer
   that bypasses the pipeline. The "extend `aggregate_grid.py`
   with `--statistic {max,mean}` flag" deliverable is partially
   done (helper exists, CLI flag plumbing not). If Sprint 11
   permutation test work lands, the cleanest path is to add both
   `--statistic mean` and `--statistic permutation` flags in the
   same PR.

## Cross-references

- ADR-0014 — original verdict (4/4 fail under max-stat).
- ADR-0015 — Lemma proving max-stat ceiling 72.2 %, motivating
  the search for a replacement statistic.
- ADR-0016 — design doc that prescribed Candidate B for
  Sprint 10.
- This ADR (0017) — empirical refutation of B, escalation to C.
- Issue #4 (closed) — gateway tracking for the broader investigation.
- Branch `sprint9/critical-pipeline` commit `858ce51` — pipeline
  + tests + machinery this work depends on.
