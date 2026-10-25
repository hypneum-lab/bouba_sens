# ADR-0021 — Me6 null calibration: the B-3 statistic under sampling noise alone

**Status:** Accepted — validated by the PI (2026-10-25). This ADR modifies no
metric, no verdict, no past ADR and no locked threshold.
**Date:** 2026-10-25
**Sprint:** post-v0.5.9 audit (follows ADR-0020)
**Artefacts:** `scripts/me6_null_calibration.py`,
`tests/unit/test_me6_null_calibration.py`,
`reports/me6_null_calibration.json`

## Context

The B-3 verdict ("perceptive / proprioceptive asymmetry") has been
reported as PASS on every grid since ADR-0004, with medians 0.1484
(Gaussian, ADR-0004), 0.1406 / 0.1406 (XOR, Sinusoid, ADR-0005-v03;
ADR-0005-cross-world lists 0.1562 for Sinusoid), 0.3125 (Studyforrest
mock, ADR-0008), 0.4453 (real ECG, ADR-0009, headline retracted) and
0.109375 (Sprint 14c joint grid, `reports/v0.5_s14c_joint_aggregate.json`),
each described as "7×–22× the 0.02 threshold". ADR-0014 then showed that
the pre-registered 3+2 partition is not distinguishable from random
partitions of the same modalities, and reframed the magnitude as an
entropy proxy. One question was never asked: **what value does the B-3
statistic take when there is no asymmetry at all**, i.e. when the 25
accuracies of the matrix share the same true value and differ only by
sampling noise? The 0.02 threshold was fixed before the campaign and
was never compared with that null.

### The pipeline, as executed

1. Each matrix entry is a `query_accuracy` on a probe batch of
   **64 samples**: `AdaptationLoop.query_accuracy`
   (`src/bouba_sens/loop.py:306-337`, default `batch_size: int = 64` at
   `:310`, `self.world.sample(batch_size=batch_size, seed=seed)` at
   `:321`, accuracy = mean of 64 correct/incorrect at `:337`). The
   `lesion` command calls it with the default batch for the five query
   modalities (`src/bouba_sens/cli.py:374`,
   `loop.query_accuracy(m, seed=seed + 777)`), and the `eval` Me1 probe
   is the same call on the audio query (`cli.py:501`).
2. The aggregator stacks, per (seed, timing, SNR) trio, the five
   `per_query_me1.json` vectors into a 5×5 matrix (rows = query
   modality, columns = lesioned modality; `scripts/aggregate_grid.py:127-162`).
3. Me6 per trio = `me6_max_abs_off_diag(me6_asymmetry(M))`
   (`src/bouba_sens/metrics/asymmetry.py:9-22` and `:25-28`): the
   maximum of the 10 absolute off-diagonal differences
   `|M[i, j] − M[j, i]|`.
4. B-3 "passes" iff the **median over the K = 30 trios** (5 seeds × 2
   timings × 3 SNR cells, `scripts/run_grid.sh`; `cells_counted = 30`
   in every 150-cell aggregate) exceeds `B3_ME6_THRESHOLD = 0.02`
   (`aggregate_grid.py:28`, `:204`, `:219-224`; `_median` averages the
   two middle values for even K, `:228-236`).

The reported medians are all multiples of 1/128 (0.1484 = 19/128,
0.1406 = 18/128, 0.3125 = 40/128, 0.4453 = 57/128, 0.109375 = 14/128):
the fingerprint of accuracies measured on 64 samples and medianed over
an even number of trios (already noticed in ADR-0014, Axe 6).

### Base accuracies actually measured

The per-cell aggregates of the v0.2 / v0.3 / v0.4 grids were never
committed (`reports/` and `runs/` are gitignored; ADR-0020 §Limits), so
the per-query accuracies behind the historical matrices are not
available in this repository. The only tracked aggregate with per-cell
accuracies is `reports/v0.5_s14c_joint_aggregate.json` (Sprint 14c joint
grid, Gaussian world, ADR-0017-sprint14): 30 cells, `me1` (audio query,
n = 64, mean over 5 seeds) between **0.211 and 0.275, grand mean 0.251**
— the chance level of the 4-class Gaussian and Sinusoid labels
(`world/gaussian.py:62`, `world/sinusoid.py:62-64`). XOR carries a
2-class label (`world/xor.py:57`), so its chance level is 0.5. The
calibration therefore sweeps p ∈ {0.2, 0.25, 0.3, 0.5, 0.7, 0.9} plus
the three measured values {0.2112, 0.2514, 0.2750}.

## Method

`scripts/me6_null_calibration.py` (numpy only, no model, no checkpoint):

- H0 (the "iid reference"): the 25 entries of `M` are independent
  `Binomial(n, p) / n` draws, each from n iid Bernoulli outcomes,
  with a common `p`. For each (n, p) it simulates R = 20 000 grids of
  K = 30 trios (600 000 trios), computes Me6 per trio exactly as steps
  2–3 above, then the median over the 30 trios exactly as step 4.
- Reported: mean, median, P95, P99 of **Me6 per trio** and of the
  **median-over-30** (the verdict statistic), and the fraction of null
  grids with median > 0.02, i.e. **P(B-3 passes | H0)**. Quantiles use
  `inverted_cdf`, so P99 is an attained value of the discrete
  distribution.
- Extra n ∈ {96, 128, 192, 384, 512, 768, 1536, 2048, 3072, 8192} are
  simulated with R = 5 000 to locate, for every observed median, the
  smallest n at which it would exceed the null P99.
- Run on electron-server (Python 3.14.4, numpy 2.4.4, seed 0,
  `nice -n 19`, 112 s):
  `uv run python scripts/me6_null_calibration.py --replicates 20000
  --out reports/me6_null_calibration.json`. The unit test runs the same
  code with R = 200.

**Simplifications and what they leave open.** The reference built here
is an **idealized iid binomial reference**, not the exact null of the
production pipeline. Three departures are known and none is modelled:

- (i) **Shared probe batch within a column.** Within one cell the five
  queries share one probe batch (`seed + 777`), so entries of the same
  column are correlated through the samples. The two entries of each
  compared pair (i, j) / (j, i) come from different cells and different
  lesion seeds (`run_grid.sh`: `seed * 10000 + count`), so each single
  difference is a difference of independent estimates, as simulated;
  but the 10 differences entering the maximum are not mutually
  independent, and that dependence changes the distribution of the
  maximum (in either direction, depending on its sign and strength).
  Equal true accuracies alone do not imply independent entries.
- (ii) **One common p.** H0 takes one common p. The p-sweep varies the
  marginal binomial variance only; it does not vary the covariance
  between entries and therefore does **not** bracket the effect of (i)
  or (iii).
- (iii) **Temporally contiguous probes (Studyforrest mock and real
  ECG only).** `StudyforrestWorld.sample` draws a contiguous temporal
  window of `batch_size` rows from a random start
  (`src/bouba_sens/world/studyforrest.py:63-68`), and the mock labels
  are a quantised slow drift of an AR(1) scene latent, autocorrelated
  by construction (`_build_mock_cache`, `:164-207`); the real ECG run
  uses the same world class and sampler on a recorded time series
  (ADR-0009). The 64 correct/incorrect outcomes of one probe are then
  not independent Bernoulli draws, the accuracy need not follow
  `Binomial(64, p) / 64`, and with positively autocorrelated outcomes
  its variance is larger than the iid value. Different seeds change the window start but do
  not remove this within-window dependence. The Gaussian, XOR and
  Sinusoid worlds draw iid samples, so (iii) does not apply to them.

A dependence-aware null (e.g. a permutation or block-bootstrap null
built from the saved per-sample outcomes, or simulation with the
measured within-column and within-window correlation) is required
before any statement stronger than "relative to the iid reference".

## Results (`reports/me6_null_calibration.json`)

### Null distribution of the verdict statistic (median over 30 trios)

| n | p | Me6/trio mean | Me6/trio P99 | median-of-30 mean | median-of-30 P95 | median-of-30 P99 | P(B-3 passes \| H0) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.20 | 0.1328 | 0.2344 | 0.1294 | 0.1406 | 0.1484 | 1.0000 |
| 64 | 0.25 | 0.1437 | 0.2500 | 0.1402 | 0.1562 | 0.1562 | 1.0000 |
| 64 | 0.30 | 0.1520 | 0.2656 | 0.1483 | 0.1641 | 0.1719 | 1.0000 |
| 64 | 0.50 | 0.1659 | 0.2812 | 0.1619 | 0.1797 | 0.1875 | 1.0000 |
| 64 | 0.70 | 0.1521 | 0.2656 | 0.1484 | 0.1641 | 0.1719 | 1.0000 |
| 64 | 0.90 | 0.0996 | 0.1719 | 0.0967 | 0.1094 | 0.1094 | 1.0000 |
| 256 | 0.20 | 0.0664 | 0.1172 | 0.0648 | 0.0703 | 0.0742 | 1.0000 |
| 256 | 0.25 | 0.0719 | 0.1250 | 0.0702 | 0.0781 | 0.0801 | 1.0000 |
| 256 | 0.30 | 0.0762 | 0.1328 | 0.0743 | 0.0820 | 0.0859 | 1.0000 |
| 256 | 0.50 | 0.0831 | 0.1445 | 0.0811 | 0.0898 | 0.0938 | 1.0000 |
| 256 | 0.70 | 0.0762 | 0.1328 | 0.0743 | 0.0820 | 0.0859 | 1.0000 |
| 256 | 0.90 | 0.0498 | 0.0859 | 0.0486 | 0.0547 | 0.0547 | 1.0000 |
| 1024 | 0.20 | 0.0333 | 0.0586 | 0.0324 | 0.0356 | 0.0371 | 1.0000 |
| 1024 | 0.25 | 0.0360 | 0.0625 | 0.0351 | 0.0386 | 0.0400 | 1.0000 |
| 1024 | 0.30 | 0.0381 | 0.0664 | 0.0372 | 0.0410 | 0.0425 | 1.0000 |
| 1024 | 0.50 | 0.0416 | 0.0723 | 0.0406 | 0.0449 | 0.0464 | 1.0000 |
| 1024 | 0.70 | 0.0381 | 0.0664 | 0.0372 | 0.0410 | 0.0425 | 1.0000 |
| 1024 | 0.90 | 0.0249 | 0.0439 | 0.0243 | 0.0269 | 0.0278 | 0.9993 |
| 4096 | 0.20 | 0.0166 | 0.0291 | 0.0162 | 0.0178 | 0.0187 | 0.0001 |
| 4096 | 0.25 | 0.0180 | 0.0315 | 0.0175 | 0.0193 | 0.0201 | 0.0161 |
| 4096 | 0.30 | 0.0191 | 0.0334 | 0.0186 | 0.0205 | 0.0214 | 0.1149 |
| 4096 | 0.50 | 0.0208 | 0.0364 | 0.0203 | 0.0223 | 0.0233 | 0.5916 |
| 4096 | 0.70 | 0.0190 | 0.0332 | 0.0186 | 0.0205 | 0.0214 | 0.1165 |
| 4096 | 0.90 | 0.0125 | 0.0217 | 0.0122 | 0.0134 | 0.0139 | 0.0000 |

At the measured base accuracies (n = 64): p = 0.2112 → median-of-30
mean 0.1320, P99 0.1562; p = 0.2514 → 0.1404, P99 0.1562; p = 0.2750 →
0.1446, P99 0.1719.

Where the fixed 0.02 threshold meets the null P99 of the verdict
statistic (p = 0.25): P99 = 0.0400 at n = 1024, 0.0286 at 2048, 0.0233
at 3072, **0.0201 at 4096**, 0.0142 at 8192. At p = 0.25 every
simulated reference grid clears 0.02 up to n = 1536, and the simulated
pass frequency only drops below 5 % at n ≈ 4096 (at p = 0.5 it is
still 0.59 at n = 4096 and 0.0 at 8192; at p = 0.9 it is already
0.9993 at n = 1024 and 0.44 at n = 1536). These are Monte Carlo
frequencies (R = 20 000 on the main grid, 5 000 on the fine grid),
not exact probabilities.

### Observed medians against the iid reference at n = 64

| Grid | Observed median | Null (median-of-30) at the relevant p | Reading |
|---|---:|---|---|
| Gaussian (ADR-0004) | 0.1484 | p ≈ 0.25: mean 0.1402, median 0.1406, P95 0.1562, P99 0.1562 | between the null median and P95 |
| XOR (ADR-0005-v03) | 0.1406 | p = 0.5 (2-class): mean 0.1619, median 0.1562, P99 0.1875 | **below** the null median |
| Sinusoid (ADR-0005-v03) | 0.1406 | p ≈ 0.25: median 0.1406 | **equals** the null median |
| Sprint 14c joint (`v0.5_s14c_joint_aggregate.json`) | 0.1094 | p ≈ 0.25: median 0.1406; simulated P(median ≤ 0.109375) = 0.0013 | **lower tail** of the reference (see below) |
| Studyforrest mock (ADR-0008) | 0.3125 | max P99 over all p: 0.1875 (median-of-30), 0.2812 (per trio) | above every iid-reference P99; reference not valid for contiguous probes, (iii) |
| Real ECG (ADR-0009) | 0.4453 | idem | idem |

The Sprint 14c lower-tail frequency is not stored in the JSON; it was
obtained from the same deterministic stream as the report
(`simulate_null(64, 0.25, replicates=20000, cells=30, seed=0)` from
`scripts/me6_null_calibration.py`, fraction of the 20 000 medians
≤ 0.109375: 0.0013; 0.0015 at p = 0.2514). A value that far below the
reference median is itself unexpected under the iid reference and is
more consistent with the reference being mis-specified for this grid
(e.g. dependence between entries, or an accuracy variance that is not
binomial) than with a clean match to it.

The "7.4× / 7× the threshold" multiples reported for the synthetic
worlds (ADR-0004, ADR-0005) are the multiples the iid reference itself
produces at n = 64: its P99 of the verdict statistic is 7.4×–9.4× the
threshold for p between 0.2 and 0.5.

### Smallest n at which each observed median would exceed the null P99

Read as: "if the asymmetry were real and kept this magnitude, how many
probe samples per entry would make it exceed what noise alone produces?"

| Observed median | p = 0.20 | p = 0.25 | p = 0.30 | p = 0.50 |
|---|---:|---:|---:|---:|
| 0.1484 (Gaussian) — verdict statistic / per trio | 96 / 192 | 96 / 192 | 96 / 256 | 128 / 256 |
| 0.1406 (XOR, Sinusoid) | 96 / 192 | 96 / 256 | 96 / 256 | 128 / 384 |
| 0.1094 (Sprint 14c) | 128 / 384 | 192 / 384 | 192 / 512 | 192 / 512 |
| 0.3125 (mock) | 64 / 64 | 64 / 64 | 64 / 64 | 64 / 64 |
| 0.4453 (ECG) | 64 / 64 | 64 / 64 | 64 / 64 | 64 / 64 |

None of the synthetic values exceeds the iid-reference P99 at the
n = 64 actually used. These n are computed under the same iid
reference and inherit its simplifications (i)–(iii); they are orders
of magnitude, not sample-size guarantees. The full table (all p, both
references) is in the JSON under `n_min`.

## Conclusion (prudent)

All statements below are relative to the **idealized iid binomial
reference** of §Method, not to the exact null of the pipeline
(simplifications (i)–(iii)).

(a) **At n = 64 the synthetic-world medians show no upper-tail excess
over the iid reference.** Against p ≈ 0.25 (p = 0.5 for XOR):
XOR 0.1406 is below the reference median (0.1562); Sinusoid 0.1406
equals it (0.1406); Gaussian 0.1484 lies above it, between the median
and the P95 (0.1562); none reaches the reference P99. Sprint 14c
0.109375 lies in the reference's **lower** tail (simulated frequency
0.0013 at p = 0.25), which is a sign that the reference is imperfect
for that grid rather than evidence of a good fit. The fixed rule
"median > 0.02" is cleared by all 20 000 simulated reference grids at
every p of the main sweep for n = 64 and n = 256, and at n = 1024 for
every p except p = 0.9 (0.9993). The B-3 PASS on these grids is
therefore what the rule also returns under the iid reference; these
grids do not by themselves provide evidence for an asymmetry. They do
not prove its absence either: an asymmetry of ~0.14 would be hidden at
this n (see (c)), and the dependence in (i) shifts the reference in an
unknown direction.

(b) **The Studyforrest mock (0.3125) and the real ECG (0.4453) exceed
the iid-reference P99 at every p and both references.** This does
**not** establish heterogeneity beyond sampling noise: both grids use
temporally contiguous probe windows over autocorrelated data
(simplification (iii)), for which the accuracy variance can exceed the
binomial value and the iid reference is not a valid null. Attributing
the excess to differences between the true accuracies of the matrix
requires a dependence-aware calibration (block bootstrap or permutation
over the saved per-sample outcomes, or simulation with the measured
within-window autocorrelation). Independently of that, ADR-0014 showed
that the magnitude does not align with the pre-registered partition and
tracks per-modality entropy, and ADR-0009 notes that three of the five
ECG modalities are zeroed. The excess remains to be explained.

(c) **A more informative test is to re-evaluate the existing cells at
larger n.** `query_accuracy` already takes `batch_size`
(`loop.py:310`); re-probing the saved lesion-adapted states of the
historical grids (checkpoints in the `runs/` trees, not in this
repository) at n = 1024 would lower the iid-reference P99 of the
verdict statistic to 0.040 (p = 0.25) and 0.046 (p = 0.5). A median
that stays near 0.14 would then be strong evidence against the iid
reference; one that falls to the 0.03–0.04 range would be consistent
with (a). Either reading must be confirmed against a dependence-aware
reference, since (i) persists at any n and (iii) persists for the
Studyforrest world unless its probes are drawn non-contiguously. Not
done here (no checkpoint access, no compute budget).

(d) **Proposal: replace the fixed 0.02 threshold by one calibrated on
a null at the n actually used.** The iid-reference values above are
indicative only — P95 / P99 of the median-of-K of 0.156 / 0.156 at
n = 64, p = 0.25 and 0.180 / 0.1875 at p = 0.5 — and should not be
adopted as thresholds as they stand: because of (i) and (iii) they may
under- or over-state the true null quantiles. A threshold for adoption
must come from a dependence-aware null matching the probe design of
each world (or be reported as the null percentile of the observed
median under that null). Under the iid reference, 0.02 corresponds to
the P99 only around n ≈ 4096 at p ≈ 0.25. Because 0.05 / 0.10 / 0.02
are locked by the OSF pre-registration, this is a proposal for an
amendment and a paper erratum, not a change made here.

## What this does and does not change

- Changes nothing in `src/`, `scripts/aggregate_grid.py`, the metrics,
  the thresholds or any recorded verdict. ADR-0004 to ADR-0020 stand as
  written; this ADR adds an idealized iid reference distribution they
  lacked, not a validated null for every world.
- Does not re-run any cell and uses no checkpoint.
- Adds a numpy-only script, its test, and the calibration JSON
  (SHA-256 recorded in `reports/MANIFEST.json`).

## Proposed follow-ups (none done here)

1. PI decision on this ADR (accept / amend / reject).
2. Build a dependence-aware null — block bootstrap or permutation over
   saved per-sample probe outcomes, or simulation with the measured
   within-column correlation (i) and within-window autocorrelation
   (iii) — before using any threshold from (d) or interpreting (b).
3. Re-probe the saved historical cells at n ∈ {1024, 4096} (c) on a
   machine holding the `runs/` trees; compare with the table above.
4. For future grids, raise the probe batch (n ≥ 1024: five forward
   passes of 1024 samples per cell, no training) and report the observed
   B-3 median together with its percentile under a null matching the
   probe design (this iid calibration only for iid-sampled worlds, and
   only as a reference until (2) is done).
5. OSF amendment + paper erratum for the "7×–22× threshold" wording
   (`docs/paper/main.tex:328, 402-405`, README, CHANGELOG) once the PI
   decides (d).
6. Add a `--batch-size` flag to `bouba-sens lesion` / `eval` so the
   probe size is recorded in `metadata.json` (currently implicit).

## References

ADR-0004, ADR-0005 (both files), ADR-0008, ADR-0009, ADR-0014-b3,
ADR-0017-sprint14, ADR-0020; `src/bouba_sens/loop.py`,
`src/bouba_sens/cli.py`, `src/bouba_sens/metrics/asymmetry.py`,
`scripts/aggregate_grid.py`, `scripts/run_grid.sh`;
`reports/v0.5_s14c_joint_aggregate.json`.
