# ADR-0021 — Me6 null calibration: the B-3 statistic under sampling noise alone

**Status:** Proposed — to be validated by the PI. This ADR modifies no
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

- H0: the 25 entries of `M` are independent `Binomial(n, p) / n` draws
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

**Simplifications.** (i) Within one cell the five queries share one
probe batch (`seed + 777`), so entries of the same column are correlated
through the samples; the two entries of each compared pair (i, j) / (j, i)
come from different cells and different lesion seeds
(`run_grid.sh`: `seed * 10000 + count`), so each difference is a
difference of independent estimates, as simulated. The residual
within-column dependence between the 10 differences is ignored. (ii) H0
takes one common p; a symmetric null with pair-specific accuracies has
the same binomial variance to first order, which the p-sweep brackets.

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
at 3072, **0.0201 at 4096**, 0.0142 at 8192; P(B-3 passes | H0) stays
at 1.000 up to n = 1536 and only drops below 5 % at n ≈ 4096 (at
p = 0.5 it is still 0.59 at n = 4096 and 0.0 at 8192).

### Observed medians against the null at n = 64

| Grid | Observed median | Null (median-of-30) at the relevant p | Reading |
|---|---:|---|---|
| Gaussian (ADR-0004) | 0.1484 | p ≈ 0.25: mean 0.1402, median 0.1406, P95 0.1562, P99 0.1562 | between the null median and P95 |
| XOR (ADR-0005-v03) | 0.1406 | p = 0.5 (2-class): mean 0.1619, median 0.1562, P99 0.1875 | **below** the null median |
| Sinusoid (ADR-0005-v03) | 0.1406 | p ≈ 0.25: median 0.1406 | **equals** the null median |
| Sprint 14c joint (`v0.5_s14c_joint_aggregate.json`) | 0.1094 | p ≈ 0.25: median 0.1406 | below the null median |
| Studyforrest mock (ADR-0008) | 0.3125 | max P99 over all p: 0.1875 (median-of-30), 0.2812 (per trio) | above every null P99 |
| Real ECG (ADR-0009) | 0.4453 | idem | above every null P99 |

The "7.4× / 7× the threshold" multiples reported for the synthetic
worlds (ADR-0004, ADR-0005) are the multiples the null itself produces
at n = 64: the null P99 of the verdict statistic is 7.4×–9.4× the
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

None of the synthetic values exceeds the null at the n = 64 actually
used. The full table (all p, both references) is in the JSON under
`n_min`.

## Conclusion (prudent)

(a) **At n = 64 the three synthetic worlds are compatible with the
null.** Gaussian 0.1484, XOR 0.1406 and Sinusoid 0.1406 sit at or below
the median of what 25 equal accuracies produce by sampling noise; the
Sprint 14c value 0.109375 is below it. The B-3 PASS on these grids is
what the rule returns on noise: under H0 the verdict passes with
probability 1.000 at n = 64, 256 and 1024. These grids therefore provide
no evidence for an asymmetry; they also do not prove its absence, since
a true asymmetry of ~0.14 would be hidden at this n (see (c)).

(b) **The Studyforrest mock (0.3125) and the real ECG (0.4453) exceed
the null P99 at every p and both references.** Their matrices contain
heterogeneity beyond binomial noise. This ADR does not explain it;
ADR-0014 showed it does not align with the pre-registered partition and
tracks per-modality entropy, and ADR-0009 notes that three of the five
ECG modalities are zeroed. It remains to be explained.

(c) **The decisive test is to re-evaluate the existing cells at
n ≥ 1024.** `query_accuracy` already takes `batch_size`
(`loop.py:310`); re-probing the saved lesion-adapted states of the
historical grids (checkpoints in the `runs/` trees, not in this
repository) at n = 1024 would lower the null P99 of the verdict
statistic to 0.040 (p = 0.25) and 0.046 (p = 0.5). A median that stays
near 0.14 would then be unambiguous; one that falls to ≈ 0.035 would
confirm (a). Not done here (no checkpoint access, no compute budget).

(d) **The fixed 0.02 threshold should be replaced by a threshold
calibrated on the null at the n actually used** — the P95 or P99 of
the median-of-K distribution above (0.156 / 0.156 at n = 64, p = 0.25;
0.180 / 0.1875 at p = 0.5), or equivalently a reported null percentile
of the observed median. The value 0.02 corresponds to the null P99 only
around n ≈ 4096 at p ≈ 0.25. Because 0.05 / 0.10 / 0.02 are locked by
the OSF pre-registration, this is a proposal for an amendment and a
paper erratum, not a change made here.

## What this does and does not change

- Changes nothing in `src/`, `scripts/aggregate_grid.py`, the metrics,
  the thresholds or any recorded verdict. ADR-0004 to ADR-0020 stand as
  written; this ADR adds the reference distribution they lacked.
- Does not re-run any cell and uses no checkpoint.
- Adds a numpy-only script, its test, and the calibration JSON
  (SHA-256 recorded in `reports/MANIFEST.json`).

## Proposed follow-ups (none done here)

1. PI decision on this ADR (accept / amend / reject).
2. Re-probe the saved historical cells at n ∈ {1024, 4096} (c) on a
   machine holding the `runs/` trees; compare with the table above.
3. For future grids, raise the probe batch (n ≥ 1024: five forward
   passes of 1024 samples per cell, no training) and report the observed
   B-3 median together with its null percentile from this calibration.
4. OSF amendment + paper erratum for the "7×–22× threshold" wording
   (`docs/paper/main.tex:328, 402-405`, README, CHANGELOG) once the PI
   decides (d).
5. Add a `--batch-size` flag to `bouba-sens lesion` / `eval` so the
   probe size is recorded in `metadata.json` (currently implicit).

## References

ADR-0004, ADR-0005 (both files), ADR-0008, ADR-0009, ADR-0014-b3,
ADR-0017-sprint14, ADR-0020; `src/bouba_sens/loop.py`,
`src/bouba_sens/cli.py`, `src/bouba_sens/metrics/asymmetry.py`,
`scripts/aggregate_grid.py`, `scripts/run_grid.sh`;
`reports/v0.5_s14c_joint_aggregate.json`.
