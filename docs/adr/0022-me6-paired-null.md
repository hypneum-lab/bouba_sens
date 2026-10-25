# ADR-0022 — Me6 paired null: the B-3 median against a sign-flip null on the saved outcomes

**Status:** Proposed — awaiting PI decision. This ADR modifies no metric,
no verdict, no past ADR and no locked threshold.
**Date:** 2026-10-25
**Sprint:** post-v0.5.9 audit (follows ADR-0021)
**Artefacts:** `src/bouba_sens/loop.py` (`query_successes`,
`successes_to_accuracy`), `src/bouba_sens/cli.py`
(`per_query_successes.json`), `scripts/me6_paired_null.py`,
`tests/unit/test_query_successes.py`, `tests/unit/test_me6_paired_null.py`,
`reports/me6_paired_null.json`

## Context

ADR-0021 compared the historical B-3 medians with an **idealized iid
binomial reference**: 25 independent `Binomial(64, p) / 64` entries with a
common p. It listed three departures from the pipeline it could not model
— the shared probe batch within a column, a single common p, and the
contiguous Studyforrest windows — and concluded that a dependence-aware
null was required before any statement stronger than "relative to the
iid reference". The round-2 cross-review of 2026-10-25 made the same
point as a gate: **"aucun verdict de compatibilité ou de rejet avant
calibration appariée"** (no compatibility or rejection verdict before a
paired calibration). The PI decided the same day to build that paired
null.

### Why the iid reference is not the null of the pipeline

The B-3 statistic is computed as follows (verified in the code, see
ADR-0021 §"The pipeline, as executed"):

1. In each lesion cell, `bouba-sens lesion` calls
   `AdaptationLoop.query_accuracy(q, seed=seed + 777)` for the five query
   modalities q (`src/bouba_sens/cli.py`). The five calls draw **the same
   probe batch** of n = 64 items (`world.sample(batch_size=64,
   seed=seed + 777)`), mask the four other modalities to zero, and return
   the mean of 64 per-item 0/1 outcomes. The lesion seed is
   `seed * 10000 + count` (`scripts/run_grid.sh`), so the five cells of a
   trio (one per lesioned modality m) use **different** probe batches.
2. `scripts/aggregate_grid.py` groups cells by (seed, timing, SNR) and
   stacks the five `per_query_me1.json` vectors into a 5×5 matrix
   `P[q, m]` (rows = query, columns = lesioned).
3. `Me6 = max_{q ≠ m} |P[q, m] − P[m, q]|` per trio
   (`src/bouba_sens/metrics/asymmetry.py`), then the **median over the
   K = 30 trios**, compared with the fixed 0.02.

Until this ADR the per-item outcomes were discarded after the mean; only
the 25 accuracies survived. A null that respects the design (same probe
items within a column, per-item outcomes, empirical accuracies, between-
item dependence in the data) needs them.

## Decision (proposed)

### Code

- `AdaptationLoop.query_successes(query_modality, *, batch_size=64,
  seed=987_654) -> list[int]` returns the per-item outcomes (1 = correct)
  with exactly the masking of `query_accuracy`. `query_accuracy` is now
  `successes_to_accuracy(query_successes(...))`: the float32 mean of the
  outcomes, which is **bit-identical** to the previous
  `(preds == label).float().mean().item()` for every batch size (the
  partial sums of 0/1 values are exact in float32; only the final
  division rounds, identically in both paths). Non-regression test
  against the pre-refactor body on batches 64 / 48 / 7.
- `bouba-sens lesion` writes `per_query_successes.json`
  (`{modality: [0/1] × 64}`) next to `per_query_me1.json`, both derived
  from the same five forward passes, in the same order as before. The
  docstring states that the five queries share one probe batch.
- Cells produced before this change have no successes file; the paired
  null cannot be computed on them and reports them as incomplete.

### Null hypothesis

H0: **symmetry of the expected performances**, `E[P[q, m]] = E[P[m, q]]`
for every pair (q, m). Under H0 the per-item paired difference
`d_i = s_qm(i) − s_mq(i)` is symmetric about zero, so its sign carries no
information about asymmetry.

### Procedure (`scripts/me6_paired_null.py`, numpy only)

For every complete trio, every unordered pair (m, q) with m < q, and
every item i: swap `s_mq(i) ↔ s_qm(i)` with probability 1/2 — i.e. flip
the sign of `d_i`. Recompute P, Me6 per trio exactly as steps 2–3, then
the median over trios exactly as the verdict. B = 20 000 replicates, seed
25102026 (fixed chunks of 250 replicates, each with its own
`default_rng([seed, start])`, so the draws do not depend on memory
settings). Reported, for the **median over trios** and for **each trio's
own maximum**: observed value, null mean / median / P95 / P99
(`inverted_cdf`, attained values), `P(null ≥ observed)` and
`p = (count + 1) / (B + 1)`, and `P(B-3 passes | H0)` with the fixed 0.02
rule. For comparison the script recomputes the ADR-0021 iid binomial
reference (`me6_null_calibration.simulate_null`) at the same K, n and the
grand-mean accuracy of the grid.

What the paired null preserves: the observed per-item outcomes (|d_i|,
hence the discordant items and the empirical accuracies of every entry),
the within-column probe sharing (the five queries of a cell keep their
common items), and whatever between-item structure the data carry — only
the **direction** of each paired difference is randomised. Discovery of
cells uses the same directory regex and the same (seed, timing, SNR)
grouping as `aggregate_grid.py` (unit-tested for equality), and the
observed Me6 is tested equal to `aggregate_grid._compute_me6` on the same
cells.

### Reading rule

- `median.frac_ge_observed` small (say < 0.05): the observed B-3 median
  sits in the upper tail of the paired null — not what sign-symmetry of
  the paired differences produces. Large: compatible with H0 at this n.
- `median.null.frac_above_threshold` is the frequency with which the
  fixed rule "median > 0.02" fires under H0; a value near 1 means the
  rule cannot separate signal from noise at this n (ADR-0021 found 1.0
  for the iid reference at n = 64).
- The per-trio rows show whether an excess, if any, is carried by a few
  trios or by all.

None of these is a verdict on B-3: the PI reads the JSON and decides.

## Results (`reports/me6_paired_null.json`)

Grid run on electron-server (Ubuntu 24.04, 8-core Xeon X5680 without
AVX, no GPU, Python 3.14.4, numpy 2.4.4, torch 2.11.0 on CPU), commit
`6ddf536` of `feat/me6-paired-null`, Gaussian world, `scripts/run_grid.sh`
defaults (STEPS_TRAIN 200, STEPS_LESION 100, `Me1,Me2,Me3`, gating hard),
150 cells run four at a time with the per-cell seeds of `run_grid.sh`
(`seed * 10000 + count`), 2026-10-25 09:51–10:43 (52 min under heavy
competing load; ~30 s per cell when the host is idle). 150/150 cells, 30
complete trios, no failure. Paired null: B = 20 000, seed 25102026,
5.6 s. Cross-check: `aggregate_grid.py` on the same grid gives
`invariants.b3.median_me6_max_abs = 0.15625`, `cells_counted = 30`
(PASS under the fixed rule) — identical to `median.observed`.

### Verdict statistic (median over the 30 trios)

| | observed | null mean | null median | P95 | P99 | P(null ≥ obs) | P(B-3 passes \| H0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| paired sign-flip null | **0.1562** (20/128) | 0.1409 | 0.1406 | 0.1562 | 0.1562 | **0.1735** (p = 0.1735) | 1.0000 |
| iid binomial reference (ADR-0021, p = 0.2541) | 0.1562 | 0.1409 | 0.1406 | 0.1562 | 0.1562 | 0.1452 | 1.0000 |

The grand-mean accuracy of the 750 matrix entries is 0.254 (4-class
chance 0.25). The pooled per-trio null has mean 0.1444, P95 0.2188,
P99 0.2500.

### Per-trio maxima

Observed Me6 per trio ranges from 0.0312 (seed1_T1_floor) to 0.3750
(seed1_T2_plus10); 20 of 30 lie in 0.109–0.1875. Each trio's own null
has mean 0.138–0.157 and P99 0.2344–0.2812. Four trios have
P(null ≥ obs) < 0.05: seed1_T1_minus10 (0.2188, 0.0487),
seed1_T2_minus10 (0.2812, 0.0111), seed1_T2_plus10 (0.3750, 0.0001) and
seed2_T1_plus10 (0.2344, 0.0462); seed4_T2_plus10 is at 0.0585. Under H0
about 1.5 of 30 would be expected below 0.05. The pairs attaining the
maximum differ from trio to trio (vision|gravity, tactile|force,
audio|vision, …): no single pair carries the excess.

### Reading (not a verdict — PI decision pending)

- The grid-level B-3 median (0.1562) is **inside the body of the paired
  null**, not in its upper tail: 17 % of null grids reach or exceed it.
  Because the statistic moves in steps of 1/128, 0.1562 is also the
  attained P95 and P99 value of the null; the observed value is one step
  above the null median (0.1406).
- The paired null and the iid reference are **numerically almost the
  same** on this grid (means 0.1409 / 0.1409, P99 0.1562 / 0.1562, tail
  0.17 / 0.15). This is what the design predicts for an iid-sampled
  world where the two members of every pair come from different probes
  (limit 3 below): the paired structure that survives is the shared
  probe within a column, which the sign flip preserves and the iid
  reference ignores, and its effect on the maximum is small here. The
  equality does not transfer to the Studyforrest worlds (limit 1).
- The fixed rule "median > 0.02" fires in **100 %** of paired-null
  grids, as it did under the iid reference: on this grid the B-3 PASS
  is what the rule returns under H0.
- A minority of trios (three of the six seed-1 trios in particular) have
  maxima in their own null's upper tail. Whether this is seed-1
  structure or multiplicity (30 trios × 10 pairs) is not decided here.

## Known limits (to be weighed by the PI)

1. **Exchangeability of items within a probe.** The per-item sign flip
   is exact (conditional on |d_i|) when the outcomes of a probe are
   exchangeable, which holds for the iid-sampled Gaussian, XOR and
   Sinusoid worlds. The Studyforrest world draws contiguous temporal
   windows (ADR-0021 (iii)); for it a **block** sign flip (contiguous
   runs of items flipped together) is needed. Not implemented here.
2. **Independent signs across the 10 pairs.** Pairs sharing a cell
   (e.g. (a, b) and (a, c) both use cell a's probe) have correlated
   differences; the null keeps their observed magnitudes but flips their
   signs independently, so the distribution of the maximum is only
   approximately that of the pipeline under H0. A stricter variant would
   flip the five entries of a cell jointly; its justification is not
   obvious when the two members of a pair come from different probes,
   and it is left open.
3. **Items are paired by index across two different probes.** Since the
   two cells of a pair use different lesion seeds, `s_mq(i)` and
   `s_qm(i)` are outcomes on different world items. Under H0 with
   exchangeable items the pairing is arbitrary but harmless; it is not a
   within-item comparison. A design where all five cells of a trio share
   one probe seed would make it one (follow-up).
4. **H0 is symmetry of expectations, not "no structure".** A grid can
   have very different accuracies per modality (ADR-0014's entropy
   reading) and still satisfy H0; conversely, rejecting H0 says nothing
   about the pre-registered 3+2 partition.
5. **One grid, one world.** The computation covers a fresh Gaussian grid
   produced for this ADR; the historical grids have no per-item files
   and cannot be re-analysed without re-running them (ADR-0021 (c)).

## What this does and does not change

- Changes nothing in the metrics, the aggregator, the thresholds or any
  recorded verdict. `query_accuracy` keeps its value bit for bit.
- Adds one file per lesion cell (`per_query_successes.json`, ~1 KB).
- Adds a numpy-only script, its tests, and the paired-null JSON.

## Proposed follow-ups (none done here)

1. PI decision on this ADR (accept / amend / reject) after reading the
   JSON.
2. Block sign flips for the Studyforrest world; a `--block` option.
3. Re-run the XOR and Sinusoid grids with the per-item files and apply
   the same null (iid worlds, so limit 1 does not apply).
4. Decide whether future grids should share one probe seed across the
   five cells of a trio (limit 3), and whether to raise n (ADR-0021 (c)).
5. If the PI adopts a null-calibrated reading of B-3: OSF amendment and
   paper erratum (ADR-0021 follow-up 5), since 0.02 is locked.

## References

ADR-0014-b3, ADR-0020, ADR-0021; `src/bouba_sens/loop.py`,
`src/bouba_sens/cli.py`, `src/bouba_sens/metrics/asymmetry.py`,
`scripts/aggregate_grid.py`, `scripts/run_grid.sh`,
`scripts/me6_null_calibration.py`; cross-review of 2026-10-25.
