# ADR-0026 — Me6 at large probe sizes: preregistered paired nulls, n ∈ {64, 256, 1024, 4096}

**Status:** Proposed — awaiting PI sign-off. Preregistration (L1 once
tagged). Modifies no metric, verdict, past ADR or locked threshold.
**Date:** 2026-10-30
**Decision source:** PI decision of 2026-10-30, 14:10 (fresh Gaussian and
mock grids, argmax-bridge, B-3 probe at n ∈ {64, 256, 1024, 4096},
per-item outcomes, paired null, power curve against n; for the
methodological note « nuls calibrés et errata »).
**Tag:** `prereg-me6-large-n-<YYYYMMDD>` (annotated, on the merge commit
of this ADR, date of the tag; expected `prereg-me6-large-n-20261030`).
No pilot or confirmatory cell may start before it.
**Builds on:** ADR-0021 (iid reference, follow-ups (c), 3, 4, 6),
ADR-0022 (paired null), ADR-0023 (trio-shared probe seed, block rule),
ADR-0024 (argmax-bridge does not learn). ADR-0025 is reserved for the
ActionSense verdicts (`docs/superpowers/plans/2026-10-28-actionsense-world.md`).
**Artefacts:** `src/bouba_sens/cli.py` (`lesion --probe-sizes`,
`--probe-sizes-seed`), `scripts/run_grid.sh` and
`scripts/run_grid_probe_v2.sh` (`PROBE_SIZES`), `scripts/me6_large_n.py`,
`scripts/b3_probe_v2.py` (`block_rule(successes_file=…)`),
`tests/unit/test_probe_sizes.py`, `tests/unit/test_me6_large_n.py`.

## Context

ADR-0021 showed that, under an iid binomial reference, the frozen B-3
rule "median Me6 > 0.02" fires on every null grid up to n ≈ 1536 and
drops below 5 % only near n = 4096. ADR-0022 replaced the reference by a
paired sign-flip null on saved per-item outcomes, at n = 64 only. With
the argmax-bridge emission the network does not learn (ADR-0024): every
v1 query accuracy is at chance, so these grids are a regime where the
v1 matrix has no structure by construction. That makes them a test bed
for the statistic, its null and the frozen rule as n grows. It is
**not** a B-3 reading: ADR-0022 made a probe above chance a
precondition of any B-3 reading, and that stays.

## What has already been seen (declared)

All grids of ADR-0021 to ADR-0024 (seeds 0–4, 100–101, 200–204),
their paired nulls and the ADR-0021 iid table at n ∈ {64, 256, 1024,
4096}. A timing smoke on electron-server before the tag (seed 999,
outside every set below, one phase 1 and three lesion cells per world;
§Cost) checked the new files and their bit-identity with the old ones;
**no Me6 statistic or null was computed on it**. The power curve of
§Cost is a simulation without data.

## Decision

### Grids

- **Worlds:** `gaussian` and `studyforrest` without data cache (mock).
- **Emission:** `argmax-bridge` (default). STEPS_TRAIN 200,
  STEPS_LESION 100, batch 16, metrics Me1,Me2,Me3, gating hard, τ 1.0,
  lesion seed `seed · 10000 + count`, as in ADR-0023.
- **Script:** `scripts/run_grid_probe_v2.sh` at the tagged commit, with
  `PROBE_SIZES=64,256,1024,4096` (v2 probe left on, its default).
- **Seeds:** confirmatory **400–404** (150 cells + 5 phase 1 per world,
  30 trios). Pilot **100–101**, mock only (60 cells); no Gaussian pilot,
  since its block is 1 by design.
- **One training grid per world.** n only acts on the probe: each lesion
  cell is probed at the four sizes on the same frozen state.

### Probes

- **Primary: v1 zero-masking probe at each n** (`query_successes`,
  files `per_query_successes_n{N}.json`). The five cells of a trio share
  one probe seed, `seed · 10000 + 6000 + trio` (trio = 3 · timing index +
  SNR index), the same for the four sizes; each size is run in a forked,
  reseeded torch RNG. P[q, m] and P[m, q] are then scored on the same
  items, as in ADR-0023 (ADR-0022 limit 3 removed).
- **Secondary, descriptive:** the historical v1 file at n = 64 with its
  per-cell seed (`per_query_successes.json`), the bridge to the values
  of ADR-0004, ADR-0008 and ADR-0022; and the v2 probe at its fixed
  size 256 (ADR-0023 design, unchanged).
- **Why v1 only varies with n.** The question is the behaviour of the
  ADR-0021 / ADR-0022 statistic and of the 0.02 rule as n grows, and
  0.02 was written for the v1 matrix. v2 is not varied: its H0 is
  ill-posed (query main effect, ADR-0023), so a larger test set would
  only shrink the null around a known rejection. Its pool of 512 items
  would also be swallowed by a 4096-item test set on the mock world
  (all items overlap and would be dropped).

### Statistic, null, block lengths

- **Statistic:** per world and per n, the median over the 30 trios of
  Me6 = max over the 10 pairs of |P[q, m] − P[m, q]|.
- **Null:** paired sign flip of `scripts/me6_paired_null.py`, B = 20 000
  replicates, seed 25102026, p = (#{null ≥ observed} + 1) / (B + 1).
- **Block rule** (ADR-0023 rule, per n): Gaussian b = 1 at every n.
  Mock: on the pilot grid, lag-k ACF of the paired differences for
  k = 1 … n/4; k* = smallest k with ACF < 0.1; b = 1 if k* = 1, else the
  power of two ≥ k*; b = n/4 if the ACF never falls below 0.1. Applied
  to the four per-size files and to the historical file
  (`scripts/me6_large_n.py blocks`). The pilot record
  (`docs/prereg/2026-10-30-me6-large-n-pilot.md`,
  `reports/me6_large_n_pilot_mock.json`) is committed and pushed
  **before** the confirmatory grids start. No Me6 statistic is computed
  on the pilot.

### Reading (per world, per n; `scripts/me6_large_n.py report`)

1. Fewer than 30 complete trios → « non concluant » (incomplete grid).
2. p < 0.01 → « excès sur le nul apparié ».
3. p ≥ 0.05 → « compatible avec le nul apparié ».
4. 0.01 ≤ p < 0.05 → « non concluant ».

Eight primary rows (2 worlds × 4 n), read separately, unadjusted p
reported together with the Bonferroni level 0.01 / 8.

**Frozen 0.02 rule, comparison only:** for each row, whether
median > 0.02 (strict) and how often the rule fires under the paired
null. It decides nothing.

### Power curve and expected report

- `scripts/me6_large_n.py power`: iid binomial entries (ADR-0021), one
  pair (audio | vision) with a true difference δ ∈ {0.02, 0.05, 0.10,
  0.15} in every trio, 30 trios, α = 0.01, null 20 000 and alternative
  2 000 replicates, seed 25102026, at p = 0.25 and at each world's
  observed grand-mean accuracy. Reported: power of the paired test,
  firing rate of the 0.02 rule under H0 and under each δ.
- Report: `reports/me6_large_n_{gaussian,mock,power}.json` and a note
  `docs/milestones/<date>-me6-large-n-results.md`, with the observed
  medians and null quantiles next to the ADR-0021 iid prediction at the
  same n. Expected, not decisive: Gaussian rows compatible with the
  paired null at every n, paired null close to the iid reference, and
  the 0.02 rule firing under H0 at n ≤ 1024 but rarely at 4096. No
  prediction for the mock rows.

### Run

- **Machine:** KXKM-AI CPUs only (`CUDA_VISIBLE_DEVICES=""`), nice 19,
  `PAR=10`, `OMP_NUM_THREADS=2` (20 cores), environment recorded
  (commit, uv lock, torch / numpy versions).
- **Output directory, new:** `/ai/saisail/bouba_sens-runs/me6-large-n-<tag date>/`
  with `pilot_mock/`, `gaussian/`, `mock/` and `*_logs/`.
- **Order:** merge on green CI → tag + Software Heritage save → pilot
  mock → commit block lengths → Gaussian and mock confirmatory grids →
  analysis. A cell that fails is rerun by the idempotent script; any
  other change is an amendment written before it is applied.

## Known limits

1. The v1 probe reads a network that does not learn (ADR-0024). The
   rows calibrate the statistic; they say nothing about B-3.
2. Mock at n = 4096 is a census: the cache has 4096 rows and
   `StudyforrestWorld.sample` then always returns the whole stream
   (start 0), so every cell and trio probes the same items. At n = 1024
   windows of different trios overlap heavily.
3. The four sizes share one seed per trio, so their items overlap; rows
   at different n are not independent and are not pooled.
4. The v1 forward pass is not RNG-free (modules in training mode,
   `query_successes` docstring). In the smoke the new n = 64 file at the
   historical seed matched `per_query_successes.json` byte for byte on
   the mock but differed on 1–3 of 64 items per query on the Gaussian
   world. The per-size probes are therefore a fresh draw, not a copy of
   the historical probe, even at n = 64.
5. KXKM is not the electron-server of ADR-0022 / ADR-0023: results are
   reproducible in distribution, not byte for byte.

## What this ADR does not change

No metric, aggregator, threshold (0.02 stays frozen and is only
compared), past verdict, B-1 / B-2 / B-3 status, or default of the CLI
and grid scripts (without `--probe-sizes` / `PROBE_SIZES` every output is
bit-identical, `tests/unit/test_probe_sizes.py`). It does not reopen
B-3, change the emission, or use the straight-through grids.

## Cost

**Smoke** (electron-server, 8-core Xeon X5680, nice 19, `OMP_NUM_THREADS=2`,
seed 999, full STEPS, load average 11–19 from other jobs, so the times are
pessimistic; 2026-10-30 14:30–15:08):

| step | Gaussian | mock |
|---|---:|---:|
| phase 1 (200 steps, v2 control) | 217 s | 241 s |
| T2 lesion cell, no `--probe-sizes` | 191 s | 175 s |
| T2 lesion cell, `--probe-sizes 64,256,1024,4096` | 192 s | 203 s |
| T1 lesion cell, same | 217 s | 179 s |
| `eval` | 15 s | 16 s |

The four extra probes are a few forward passes (≈ 27 000 items per cell,
no training); the mock difference is within the load noise. Idle
reference, ADR-0023 (same machine, PAR = 4): 155 jobs in 54 min (Gaussian)
and 36 min (mock), i.e. ≈ 85 s and ≈ 55 s per job per slot.

**Total.** n only acts on the probe, so one training grid per world is
enough: 2 × (150 cells + 5 phase 1) = 310 jobs, plus the mock pilot
(60 cells + 2 phase 1) = 372 jobs. Wall time on KXKM at PAR = 10
(20 cores): ≈ 372 × 55–85 s / 10 ≈ **35–55 min** at the idle per-job
cost, ≈ **2.2 h** at the loaded cost of the smoke (≈ 210 s per job).
Four separate grids per n would cost 4 × 310 + 62 = 1 302 jobs, ≈ 2–7.6 h,
for the same data. Analysis: the paired null at n = 4096 (30 trios,
20 000 replicates) took 255 s at block 1 and 41 s at block 64 under the
same load; with the iid reference, ≈ 10 min per world. The power curve
(below) took 15 s.

**Design-stage power curve** (`me6_large_n.py power`, p = 0.25, iid
reference, no data):

| n | null mean | null P99 | 0.02 rule fires under H0 | power δ = 0.02 / 0.05 / 0.10 / 0.15 |
|---:|---:|---:|---:|---|
| 64 | 0.1402 | 0.1562 | 1.000 | 0.01 / 0.02 / 0.18 / 0.84 |
| 256 | 0.0702 | 0.0801 | 1.000 | 0.02 / 0.20 / 1.00 / 1.00 |
| 1024 | 0.0351 | 0.0400 | 1.000 | 0.08 / 1.00 / 1.00 / 1.00 |
| 4096 | 0.0176 | 0.0203 | 0.018 | 0.92 / 1.00 / 1.00 / 1.00 |

## References

ADR-0021, ADR-0022, ADR-0023, ADR-0024;
`docs/prereg/2026-10-25-b3-probe-v2.md`; PI decision 2026-10-30 14:10.
