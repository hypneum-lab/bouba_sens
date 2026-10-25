# Pilot report — amendments v0.7 / v0.8 / v0.9 (2026-10-25)

**Status:** pilot data only. Nothing here is confirmatory, and no
pre-registered rule, bound, threshold or N was changed because of these
results. Pilot data are excluded from every confirmatory test (v0.7 §4.3,
v0.8 §7, v0.9 §7). No confirmatory seed (1001–1099, 2001–2044,
3001–3044) was run, sampled or inspected.

**Authorisation:** PI go, 2026-10-25 11:50 (run the pilots of the
draft PR, branch `docs/osf-prereg-2026-10`).

## Provenance

| Item | Value |
|------|-------|
| bouba_sens commit (code that ran) | v0.7 round 1 and v0.9 check: `b09070bc6af4a78cd1ea8f57d20918f2a443f84d` plus the scripts in `scripts/pilots/` (commit `164a18c`); v0.7 round 2: `de68e990a4822a040be30bc4c0fa5ef8e7df9405` |
| nerve-wml | tag `v1.7.0`, `c36822aac38dc83e2c9dd14f504948d0d17267eb` (the CI pin, `.github/workflows/ci.yml`) |
| Host | macM1: Apple M1 Max, 32 GB, macOS 27.0.1 (26A434), CPU only, `nice -n 10`, one job at a time |
| Python / torch / numpy | 3.14.4 / 2.11.0 / 2.4.4 (`uv run --frozen`, `uv.lock` of the branch; local uv rewrote only the lock-file revision header, not committed) |
| World cache | `StudyforrestRealWorld`, 9,000 frames, label counts [180, 3480, 4280, 1060], 10 label runs (median run 530 frames) |
| Date (UTC) | 2026-10-25 10:53–11:01 (round 1, v0.9 check), 11:23–11:24 (v0.7 round 2) |

Cache SHA-256 (`data/studyforrest_real/`):

```
37e12a14dcb39950b3442e243141ca8b05f74eb1130a81c2756b66ac6908f049  audio.pt
924ee146c9669250a128387ed85fad047cbd91a183a63f4c58dbc529a4cef9d5  vision.pt
a553c11be9ee76eb4340ad5f9bba959449d34b000e0e2520a8f9f53654fb1921  tactile.pt
38b6bc7ed4c125c605010a28b20680b24925af403d5665832d9c7b7e8f508964  gravity.pt
890b90aea31adb93f160609a74ce5a2da50942fa82ff8802a7645f9ef15cd773  force.pt
5de7e69b6a7b96406fbec839b83b073a6924fde34702ebe00ee154edaf67c3b2  labels.pt
637f974bfefbe0cdf8bafe92e87fa6f60a54310e3216741b7e00776f61c8c665  MANIFEST.md
```

The JSON outputs are listed with SHA-256 in `reports/MANIFEST.json`
(`pilots/…`); the `.jsonl` diagnostics and the logs are committed
alongside (the manifest tool only covers `.json`, `.csv`, `.png`).

## Summary

| Pilot | Seeds | Status | Wall time | Output |
|-------|-------|--------|-----------|--------|
| v0.7 B-2 round 1 (§4.3, protocol of 2026-10-24) | 1101–1105 | **run**; primary readout degenerate (findings 1–2) | 57 s total; 10–13 s per seed | `reports/pilots/v07_b2_pilot_seeds1101-1105.json`, `…_derived.json`, `v07_b2_probe_diagnostic.jsonl` |
| v0.7 B-2 round 2 (§4.2 after the PI decisions of 2026-10-25) | 1101–1105 (same seeds) | **run** | 54 s total; 10–12 s per seed | `reports/pilots/v07_b2_pilot_r2_seeds1101-1105.json`, `…_r2_derived.json` |
| v0.8 (§7) | 2101–2110 | **blocked**: driver missing (draft filing note, §9) | — | none; structural check on seed 0 only (`v08_hard_rank_diagnostic_seed0.jsonl`) |
| v0.9 manipulation check (§3.3) | 3101–3105 | **run, gate passed** | 35 s | `reports/pilots/v09_manipulation_check_seeds3101-3105.json` |
| v0.9 variance components of D, runtime (§7) | 3101–3105 | **blocked**: driver for arms / grids missing (draft filing note, §9) | — | none |

## v0.7 — B-2 pilot, round 1 (protocol as drafted on 2026-10-24)

**Command** (`reports/pilots/logs/run_v07.sh`, log `logs/v07_b2_pilot.log`):

```
BOUBA_SENS_STUDYFORREST_DATA=<cache> nice -n 10 uv run --frozen python -u \
  scripts/spikes/e1_b2_dimensional_mine.py --preset v07-e2 \
  --seed-list 1101,1102,1103,1104,1105 \
  --out reports/pilots/v07_b2_pilot_seeds1101-1105.json
```

The preset fixes the frozen §4.2 configuration (world studyforrest,
LOCK_AFTER 100, 200 / 100 steps, batches 16 / 16 / 512, `k_steps` 50,
`--probe-demod both`) and its world check passed
(`StudyforrestRealWorld`, also recorded per seed in
`v07_b2_probe_diagnostic.jsonl`).

**Per-seed primary ΔMI-hat (Ross, hard readout), bits:** 0, 0, 0, 0, 0
(exactly). MI-hat pre = post on every seed; 512 of 512 probe points
have a zero k-NN distance (`ross_n_zero_distance` = 512, pre and post).

**Per-seed E-2g ΔMI-hat (Ross, Gumbel readout), bits:** 0, +0.01406, 0,
+0.03548, 0 (mean +0.00991, SD 0.01554).

**Finding 1 — the hard (primary) readout is input-independent.** On
every pilot seed, the hard-demodulated full fused probe
(`nerve.fuse(hard_demod=True)`) has **one distinct row** across the
512 probe rows, before and after the lesion
(`v07_b2_probe_diagnostic.jsonl`): every probe input is mapped to the
same code. Pre and post codes differ from each other, but within each
matrix all rows are identical, so every k-NN MI estimate is determined
by the labels alone and ΔMI-hat = 0 for every estimator (exactly 0
for Ross, full KSG, 1-D KSG and binning; |Δ| < 1e-6 bit for MINE). The Gumbel readout has 512 distinct rows.
Consequence for the registered E-2: with this readout, the per-seed
statistic is identically 0, so E-2 would pass TOST with a zero-width
CI whatever the network does; the test would carry no information.

**Finding 2 — the probe often contains a single label.** The probe is
one contiguous window of 512 frames (`world.sample(512, seed = s −
99_999)`), and the cache has only 10 label runs (median 530 frames).
Probe label counts: 1101 {1: 512}, 1102 {1: 346, 2: 140, 3: 26},
1103 {1: 512}, 1104 {1: 144, 2: 140, 3: 228}, 1105 {2: 512}. With one
label, MI(Z; y) = 0 for any code, so ΔMI-hat = 0 by construction (the
Gumbel value is exactly 0 on 1101, 1103 and 1105 for this reason).
Class 0 is absent from all five probes. §4.2 says "a class with fewer
than k + 1 probe points makes the seed a technical failure"; the code
applies this only to classes present in the probe. Read over all four
classes, every pilot seed would be a technical failure.

**Pilot-derived quantities (§7, `scripts/pilots/derive_v07_pilot.py`,
`reports/pilots/v07_b2_pilot_derived.json`).**

| Readout | per-seed SD | UCL80 (df = 4) | exact TOST power at N = 95, μ = 0 / −0.02 | N for 0.80 | label |
|---------|------------:|---------------:|------------------------------------------:|-----------:|-------|
| hard (primary in round 1) | 0 | 0 | undefined (degenerate: all values identical) | — | none assignable (see finding 1) |
| Gumbel (E-2g in round 1, descriptive only) | 0.01554 | 0.02420 | 1.000 / 1.000 | 4 / 6 | "adequately powered" (descriptive) |

N stays 95 (§7). The Gumbel row is not the registered computation
(§6: "the E-2 power is computed on the hard readout"); it is reported
because the hard row is undefined.

**Runtime (§7).** 57 s for the five seeds on macM1 (seed 1101: 13 s
including start-up and the world check; 1102–1105: 10–11 s each).
At 11 s per seed, the 95 confirmatory B-2 runs take about 18 min on
this host.

The derived file was regenerated with the round-2 version of
`derive_v07_pilot.py`, which reads the primary readout from the pilot
file; the numbers are unchanged (keys `primary_hard_ross`,
`gumbel_ross_descriptive`).

## v0.7 — B-2 pilot, round 2 (PI decisions of 2026-10-25)

After seeing round 1, the PI decided (2026-10-25, about 12:10; disclosed
in the draft header as post-pilot changes of the unfiled draft): (1) the
Gumbel readout becomes the primary E-2 readout and the hard readout
becomes descriptive (E-2h); a deterministic soft readout may be added
later as a declared secondary comparison; (2) the E-2 probe becomes a
class-balanced multi-window probe, fixed at the source, specified in
§4.2 (4 classes × 8 windows × 16 frames, blocks tiled from label runs,
`torch.randperm` with a generator seeded s − 99_999; implemented in
`bouba_sens.probe`, commit `de68e99`). The pilot was re-run on the same
seeds.

**Command** (`reports/pilots/logs/run_v07_r2.sh`, log
`logs/v07_b2_pilot_r2.log`, commit `de68e99`):

```
BOUBA_SENS_STUDYFORREST_DATA=<cache> nice -n 10 uv run --frozen python -u \
  scripts/spikes/e1_b2_dimensional_mine.py --preset v07-e2 \
  --seed-list 1101,1102,1103,1104,1105 \
  --out reports/pilots/v07_b2_pilot_r2_seeds1101-1105.json
```

Probe label counts: 128 / 128 / 128 / 128 on every seed. Zero k-NN
distances in the Gumbel codes: 0 on every seed (pre and post); in the
hard codes all 512 points still have a zero k-NN distance on every
seed (one distinct row, except the post-lesion code of 1105, which has
a few repeated rows).

| Seed | Primary ΔMI-hat (Gumbel, Ross), bit | E-2h (hard, Ross), bit | D_PR (post, Gumbel) |
|-----:|-----------:|-----------:|------:|
| 1101 | +0.0466 | 0 | 13.08 |
| 1102 | +0.0347 | 0 | 12.04 |
| 1103 | +0.0694 | 0 | 12.38 |
| 1104 | +0.0789 | 0 | 13.12 |
| 1105 | +0.6461 | +0.1859 | 3.52 |

**Pilot-derived quantities (§7, `reports/pilots/v07_b2_pilot_r2_derived.json`).**

| Readout | mean | per-seed SD | UCL80 (df = 4) | exact TOST power at N = 95, μ = 0 / −0.02 | N for 0.80, μ = 0 / −0.02 | label |
|---------|-----:|------------:|---------------:|------------------------------------------:|--------------------------:|-------|
| Gumbel (primary, registered) | +0.1751 | 0.2639 | 0.4110 | 3.8e-7 / 3.4e-7 | 581 / 1,162 | **"low informativeness"** |
| hard (E-2h, descriptive) | +0.0372 | 0.0831 | 0.1295 | 0.964 / 0.725 | 59 / 117 | "under-powered" (descriptive) |

N stays 95 (§7). The SD is dominated by seed 1105 (+0.646 bit, with a
post-lesion participation ratio of 3.5 against 12–13 on the other
seeds); the rule uses the pilot SD as defined, without exclusion. The
five pilot values are all positive (descriptive; pilot data are not
tested).

**Runtime (round 2).** 54 s for the five seeds (1101: 12 s including
start-up; 1102–1105: 10–11 s each).

### Seed 1105 diagnosis (read-only, PI request 2026-10-25 12:45)

Question: is the 1105 value (+0.646 bit, post-lesion participation
ratio 3.5 against 12–13 elsewhere) a genuine outcome, a numerical or
implementation artefact, or a probe / readout edge case? Script
`scripts/pilots/diagnose_v07_seed1105.py` (re-runs the frozen preset
step by step with the same calls as the runner; no rule or statistic
changed); outputs `reports/pilots/v07_r2_seed1105_diagnostic.jsonl`
and `reports/pilots/v07_r2_gumbel_redraw.jsonl`, macM1, same commit
and environment as round 2.

| Seed | Ross Δ (reproduced) | loss first / last 10 steps (max) | post PR | post top-1 eigenvalue share | post between-class share of variance | post total variance | NN label agreement post |
|-----:|--------:|---------------------:|------:|------:|------:|------:|------:|
| 1101 | +0.0466 | 1.675 / 1.797 (3.57) | 13.08 | 0.106 | 0.006 | 0.190 | 0.287 |
| 1102 | +0.0347 | 1.666 / 1.804 (3.64) | 12.04 | 0.118 | 0.010 | 0.227 | 0.291 |
| 1103 | +0.0694 | 1.769 / 1.845 (3.72) | 12.38 | 0.109 | 0.008 | 0.201 | 0.262 |
| 1104 | +0.0789 | 1.667 / 1.882 (3.57) | 13.12 | 0.104 | 0.006 | 0.236 | 0.273 |
| 1105 | +0.6461 | 1.669 / 2.075 (4.41) | 3.52 | 0.513 | 0.346 | 0.469 | 0.418 |

Pre-lesion, 1105 looks like the other seeds (PR 12.5, top-1 share
0.113, between-class share 0.007, Ross MI-hat +0.043 bit).

Evidence:

1. *Not a numerical artefact.* Every loss value is finite, the run is
   deterministic (the diagnostic reproduces the committed Δ to all
   digits), and five estimators agree on a large increase (Ross
   +0.646, full KSG +0.658, MINE full +0.557, binning +0.586, 1-D KSG
   +0.597 bit; round-2 output).
2. *Not one Gumbel draw.* Re-drawing the post-lesion Gumbel readout
   five times on the same trained network (RNG seeds 10,000–10,004)
   gives Ross MI-hat post 0.634–0.677 bit (mean 0.662, SD 0.017) and
   PR 3.36–3.56; on 1101 the same check gives −0.018 to +0.051 (mean
   +0.010, SD 0.028).
3. *Not a probe edge case.* The probe is the balanced 4 × 128 probe on
   every seed; pre-lesion statistics of 1105 are typical.
4. *A change in the trained network.* After the lesion phase the 1105
   code is dominated by one direction (51 % of the variance; total
   variance doubled) that carries the label (between-class share 0.35
   against ≤ 0.01 elsewhere). The hard readout, input-independent on
   the other seeds, splits the 1105 probe into two codes: {class 0:
   0, 1: 32, 2: 128, 3: 112} and {0: 128, 1: 96, 2: 0, 3: 16}, i.e. a
   learned switch separating classes 0/1 from 2/3. The lesion-phase
   loss rose more than on the other seeds (last-10 mean 2.08 against
   1.80–1.88; maximum 4.41).

Diagnosis: a **genuine outcome of this seed's lesion-phase training
trajectory** (a reorganisation of the fused code into a class-aligned
dominant direction), not an artefact of the estimator, the readout
draw or the probe. It is reported as is; no rule is changed and the
seed is not excluded (v0.7 §6: outliers are not removed). A side
observation for the PI: the within-seed Gumbel readout noise of the
post-lesion MI-hat (SD 0.017–0.028 bit over re-draws) is of the same
order as the ΔMI-hat of the other four seeds (+0.035 to +0.079).

**Not done in this pilot (v0.7).** (a) The §11 checklist item "effective
SNR checked in `metadata.json` of a pilot cell" refers to a B-1 grid
cell, but §4.3 says no B-1 pilot is needed and names no B-1 pilot seed;
not run. (b) The known-signal calibration (§4.3 item 5) is a pre-filing
task, not a pilot; not run here.

## v0.8 — blocked

The pilot needs the driver that trains the Phase-1 checkpoints and
calls `capture_block` for every (seed, lock) block (draft filing note;
§9 checklist, unchecked). It is not in the repository, so k*, the
within-seed SD of Y, the median S_split, the exclusion rate, the
runtime and N could not be produced. Seeds 2101–2110 were not used.

**Finding 3 — structural check (not a pilot run).** v0.8 uses the same
hard probe path (`lesion_phase(probe_demod="hard")` →
`fuse(hard_demod=True)`). On seed 0 (a historical default grid seed
outside every amendment pool), LOCK_AFTER 25 and 175, Z_pre under hard
demodulation has 1 distinct row and numerical rank 0 (its plug-in
participation ratio is 0 by the library convention for constant codes);
the Gumbel capture of the same checkpoint has rank 436–437
and PR 12.4–12.7 (`v08_hard_rank_diagnostic_seed0.jsonl`). If the pilot
reproduces this, `select_k_from_pilot` returns k* = 1 (clamp of a
median PR of 0), every Z_pre fails the rank check of §3.1, no block is
defined and the run is a protocol failure by construction.

## v0.9 — manipulation check run, rest blocked

**Command** (`reports/pilots/logs/run_v09.sh`, log
`logs/v09_manipulation_check.log`):

```
BOUBA_SENS_STUDYFORREST_DATA=<cache> nice -n 10 uv run --frozen python -u \
  scripts/pilots/run_v09_manipulation_check.py \
  --out reports/pilots/v09_manipulation_check_seeds3101-3105.json \
  3101 3102 3103 3104 3105
```

`run_v09_manipulation_check.py` is a thin entry point (new in this
commit) around the merged `decorrelation.run_manipulation_check` with
its frozen constants (32 windows × 16 anchors, 200 draws per seed,
10,000 label swaps, generator seed 20261023) on
`StudyforrestRealWorld(probe_fraction = 0.10)` (n_train 8,100,
n_probe 900). It refuses any seed list other than 3101–3105.

| Quantity | Value |
|----------|------:|
| Δ_K = mean(K_joint − K_decor) | 0.10775 |
| one-sided permutation p | 9.999e-05 (= 1 / 10,001, the smallest attainable) |
| mean K_joint / K_decor | 0.12091 / 0.01316 |
| mean K of the original contiguous anchors (reported, not used) | 0.16580 |
| Gate (§3.3) | **passed** |
| Wall time | 35 s |

The variance components of D, the planning SD (× 1.558) and N need the
arm / grid driver and the train-side fusion gain (draft filing note,
§9 checklist, unchecked); not run. The pilot fusion-gain probe uses
held-out frames, so finding 2 does not transfer directly, but the
probe window is again one contiguous window (inside the 900 probe
frames).

## Decisions needed from the PI

1. **v0.7 E-2 primary readout.** The hard readout makes E-2 vacuous
   (finding 1). Options include: keep the Gumbel readout as primary
   (as in ADR-0019), redefine the hard readout (for example the soft
   demodulation probabilities without sampling), or drop E-2. Any
   change is a change to the draft before filing.
2. **v0.7 probe label coverage.** A single contiguous 512-frame window
   is single-label on 3 of 5 pilot seeds (finding 2). The probe
   definition, and the reading of the "fewer than k + 1 points per
   class" technical-failure rule (classes present vs all four classes),
   need a decision before filing.
3. **v0.8 hard probe.** Finding 3 implies a protocol failure by
   construction under the hard probe rule (PI decision 2026-10-23).
   This needs a decision before the driver is written and the pilot
   run.
4. **Drivers.** The v0.8 driver and the v0.9 arm / grid driver (with
   the train-side fusion gain) must be written, reviewed and merged
   before those pilots can run.
5. **v0.7 B-1 pilot cell** for the SNR `metadata.json` check: which
   seed, if any.
