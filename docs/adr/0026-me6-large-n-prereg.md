# ADR-0026 — Me6 at large probe sizes: preregistered paired nulls, n ∈ {64, 256, 1024, 4096}

**Status:** Proposed — awaiting PI sign-off. Preregistration (L1 once
tagged). Modifies no metric, verdict, past ADR or locked threshold.
**Date:** 2026-10-30
**Decision source:** PI decision of 2026-10-30, 14:10 (fresh Gaussian and
mock grids, argmax-bridge, B-3 probe at n ∈ {64, 256, 1024, 4096},
per-item outcomes, paired null, power curve against n; for the
methodological note « nuls calibrés et errata »). Amended before any
run by the PI decisions of 2026-10-30 ≈ 16:10: larger mock stream,
fresh v1 draw accepted, no Gaussian pilot, iid **and** empirical power,
run on electron-server (not KXKM).
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
`src/bouba_sens/world/studyforrest.py` (`stream_sha256`, `cache_sha256`,
`n_rows`), `BOUBA_SENS_STUDYFORREST_MOCK_SAMPLES` (`cli._build_world`),
`tests/unit/test_probe_sizes.py`, `tests/unit/test_me6_large_n.py`,
`tests/unit/test_mock_samples.py`.

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
4096}. Two timing smokes on electron-server before the tag (seed 999,
outside every set below; §Cost) checked the new files, their
bit-identity with the old ones, the item disagreements of limit 4, the
mock stream hashes and label counts, and the run time of the empirical
power on a **synthetic** grid; **no Me6 statistic or null was computed
on any smoke cell**. The iid power curve of §Cost is a simulation
without data.

## Decision

### Grids

- **Worlds:** `gaussian`, and `studyforrest` without data cache (mock)
  with the **longer mock stream** of §Mock stream below
  (`BOUBA_SENS_STUDYFORREST_MOCK_SAMPLES=65536`, pilot and confirmatory).
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

### Mock stream (PI decision 2026-10-30 ≈ 16:10)

- **How the historical mock is built.** No file and no script: it is
  generated in memory at every world construction by
  `_build_mock_cache(n, seed)` (`src/bouba_sens/world/studyforrest.py`),
  with `n = 4096` (`StudyforrestWorld(n_mock_samples=4096)` default) and
  `seed` = the world seed passed by the CLI. It is an AR(1) 4-d scene
  latent (α = 0.9), audio / vision projections plus noise, labels = the
  latent's first coordinate (lag 2) cut at its quartiles. The source is
  a generator, not a finite recording, so it can be lengthened cleanly.
- **Opt-in length.** `BOUBA_SENS_STUDYFORREST_MOCK_SAMPLES=N` makes
  `_build_world` call `StudyforrestWorld(seed, n_mock_samples=N)`, in
  `train`, `lesion` and `eval` alike; `train` / `lesion` then record
  `mock_samples` and `mock_cache_sha256` in `metadata.json`. Unset, the
  mock is built exactly as before (`tests/unit/test_mock_samples.py`).
- **Size: N = 65 536 (16 × 4096).** A 4096-item probe then covers 6.25 %
  of the stream; window starts are uniform on [0, 61 440), so two
  independent 4096-windows overlap with probability ≈ 2 · 4096 / 61 440
  ≈ 13 %, and the v2 test / pool windows rarely overlap. The stream is
  ≈ 6 500 correlation lengths of the AR(1) latent (≈ 1 / (1 − 0.9) = 10
  steps). Memory ≈ 100 MB per process, build ≈ 10 s (Python AR loop),
  both acceptable at PAR = 4; 64 × would cost 400 MB and ≈ 40 s per
  process for no gain in the question asked.
- **Hashes** (`stream_sha256`: audio float32, vision float32, labels
  int64 with shapes; electron-server, torch 2.11.0, numpy 2.4.4):
  - historical 4096 rows, seed 0:
    `67c6466906044f2a311c451dcf14f604964576e1a6efd2b5b36db5a7d3d4ed2f`
  - 65 536 rows, seed 0 (the `lesion` world, i.e. every probe):
    `4709392ef5853b6e391ba6a0346e26e3914e35872cf8cf3d2aeb839673493b85`
  - each phase-1 stream (seed = training seed) and each `eval` stream is
    recorded in its `metadata.json` at run time.
  Labels stay balanced (16 384 per class).
- **A modified mock world.** The longer stream is not an extension of the
  historical one: the projections are drawn after the n × 4 noise and the
  label quartiles are taken over the whole stream, so every row differs.
  The mock rows of this ADR are **not comparable item by item** with any
  earlier mock grid (ADR-0008, ADR-0022, ADR-0023); only distribution-level
  comparisons are made, and they are labelled as such.

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

### Power curves and expected report

- **iid** (`me6_large_n.py power --mode iid`): iid binomial entries
  (ADR-0021), one pair (audio | vision) with a true difference δ ∈ {0.02,
  0.05, 0.10, 0.15} in every trio, 30 trios, α = 0.01, null 20 000 and
  alternative 2 000 replicates, seed 25102026, at p = 0.25 and at each
  world's observed grand-mean accuracy. Reported: power of the paired
  test, firing rate of the 0.02 rule under H0 and under each δ.
- **Empirical** (`me6_large_n.py power --mode empirical`, on each
  confirmatory grid and each size, with the committed blocks): in every
  trio, k = round(δ · n / 2) items of P[a, b] equal to 0 are set to 1 and
  k items of P[b, a] equal to 1 are set to 0, chosen at random (fewer if
  not available; the effective δ = 2k / n is reported). Pairs
  **audio | vision** and **gravity | force** (one per side of the
  3 + 2 partition), δ ∈ {0.02, 0.05, 0.10}, **50 plantings** each, each
  planting with its own generator `default_rng([25102026, n, pair, δ
  index, planting])` and its own paired sign-flip null (1 000 replicates,
  seed 25102026 + planting, committed block). Reported per (world, n,
  pair, δ): power = fraction of plantings with p < 0.01, firing rate of
  the 0.02 rule, mean planted median. The planting ignores the item
  order, so on the mock it plants an asymmetry without temporal
  structure; this is declared, not corrected.
- Report: `reports/me6_large_n_{gaussian,mock,power}.json` and a note
  `docs/milestones/<date>-me6-large-n-results.md`, with the observed
  medians and null quantiles next to the ADR-0021 iid prediction at the
  same n. Expected, not decisive: Gaussian rows compatible with the
  paired null at every n, paired null close to the iid reference, and
  the 0.02 rule firing under H0 at n ≤ 1024 but rarely at 4096. No
  prediction for the mock rows.

### Run

- **Machine:** electron-server (8-core Xeon X5680, CPU torch), nice 19,
  from 23:00 on the day of the tag, after NH-01 has finished; `PAR=4`,
  `OMP_NUM_THREADS=2` (8 threads on 8 cores, as in ADR-0022 / ADR-0023).
  Not KXKM (a CPU job there slows E1-small by 40–130 %). Environment
  recorded (commit, uv lock, torch / numpy versions).
- **Output directory, new:**
  `~/scratch/bouba_sens-runs/me6-large-n-<tag date>/` on electron-server,
  with `pilot_mock/`, `gaussian/`, `mock/` and `*_logs/`.
- **Order:** merge on green CI → tag + Software Heritage save → pilot
  mock → commit block lengths → Gaussian and mock confirmatory grids →
  analysis (paired nulls, then iid and empirical power). A cell that fails is rerun by the idempotent script; any
  other change is an amendment written before it is applied.

## Known limits

1. The v1 probe reads a network that does not learn (ADR-0024). The
   rows calibrate the statistic; they say nothing about B-3.
2. The mock is a modified world (§Mock stream): not comparable item by
   item with any earlier mock grid. With the historical 4096-row stream a
   4096-item probe would have been the whole stream for every cell.
3. The four sizes share one seed per trio, so their items overlap; rows
   at different n are not independent and are not pooled.
4. The v1 forward pass is not RNG-free (modules in training mode,
   `query_successes` docstring). In the smoke the new n = 64 file at the
   historical seed matched `per_query_successes.json` byte for byte on
   the mock but differed on 1–3 of 64 items per query on the Gaussian
   world. The per-size probes are therefore a fresh draw, not a copy of
   the historical probe, even at n = 64.
5. World seeds (pre-existing, not changed here): `train` builds its
   world with the training seed, `lesion` with seed 0 and `eval` with the
   lesion seed (`cli.py`). For the mock, phase 1 therefore trains on a
   different stream from the one every lesion cell adapts on and is
   probed on (for the Gaussian world, on different projections). This
   holds for every grid of the repository and is reported, not fixed.

## What this ADR does not change

No metric, aggregator, threshold (0.02 stays frozen and is only
compared), past verdict, B-1 / B-2 / B-3 status, or default of the CLI
and grid scripts (without `--probe-sizes` / `PROBE_SIZES` every output is
bit-identical, `tests/unit/test_probe_sizes.py`; without
`BOUBA_SENS_STUDYFORREST_MOCK_SAMPLES` the mock is the historical one,
`tests/unit/test_mock_samples.py`). It does not reopen
B-3, change the emission, or use the straight-through grids.

## Cost (electron-server, PAR = 4)

**Smoke 1** (8-core Xeon X5680, nice 19, `OMP_NUM_THREADS=2`, seed 999,
full STEPS, load average 11–19 from other jobs, so pessimistic;
2026-10-30 14:30–15:08), historical mock:

| step | Gaussian | mock (4096) |
|---|---:|---:|
| phase 1 (200 steps, v2 control) | 217 s | 241 s |
| T2 lesion cell, no `--probe-sizes` | 191 s | 175 s |
| T2 lesion cell, `--probe-sizes 64,256,1024,4096` | 192 s | 203 s |
| T1 lesion cell, same | 217 s | 179 s |
| `eval` | 15 s | 16 s |

**Smoke 2** (same machine, load 8–15, 16:19–16:29), mock 65 536: stream
build 10.2 s per process; phase 1 183 s; T2 lesion cell with the four
sizes 254 s; `eval` 33 s (it builds the stream too). The four extra
probes cost a few forward passes; the longer mock costs ≈ 20 s per job
(two stream builds). Idle reference, ADR-0023 (same machine, PAR = 4):
155 jobs in 54 min (Gaussian) and 36 min (mock), ≈ 85 s and ≈ 55 s per
job per slot.

**Grids.** n only acts on the probe, so one training grid per world:
Gaussian 155 jobs; mock 155 + pilot 62 = 217 jobs.

| | jobs | idle (≈ s / job) | wall, PAR 4, idle | loaded (≈ s / job) | wall, PAR 4, loaded |
|---|---:|---:|---:|---:|---:|
| Gaussian | 155 | 85 | ≈ 55 min | 210 | ≈ 2.3 h |
| mock 65 536 (pilot + confirmatory) | 217 | 75 | ≈ 68 min | 290 | ≈ 4.4 h |
| **total grids** | 372 | | **≈ 2.0 h** | | **≈ 6.7 h** |

The pilot (62 mock jobs, ≈ 20–75 min) must finish and its block lengths
be committed before the confirmatory grids start. Four separate grids
per n would cost ≈ 4 × the training jobs for the same data.

**Analysis** (smoke timings under load): paired null at n = 4096, 30
trios, 20 000 replicates, 255 s at block 1 (41 s at block 64), ≈ 10 min
per world over the five files with the iid reference. Empirical power:
one planting with its 1 000-replicate null takes ≈ 0.14 / 0.46 / 1.9 /
7.4 s at n = 64 / 256 / 1024 / 4096 (block 1, synthetic grid), so
2 pairs × 3 δ × 50 plantings ≈ 300 × 9.9 s ≈ **50 min per world**
(less on the mock if its blocks are > 1). iid power curve: 15 s.

**Total on electron-server from 23:00: ≈ 4 h idle, ≈ 9 h under the
smoke's load** (grids, then ≈ 2 h of analysis).

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
