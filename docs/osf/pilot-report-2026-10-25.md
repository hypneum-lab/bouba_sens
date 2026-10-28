# Pilot report — amendments v0.7 / v0.8 / v0.9 (2026-10-25)

**Status:** pilot data only. Nothing here is confirmatory. Pilot data
are excluded from every confirmatory test (v0.7 §4.3, v0.8 §7, v0.9
§7). No confirmatory seed (v0.7 4001–4154, v0.8
2001–2044, v0.9 3001–3044) was run, sampled or inspected.

**Result-informed changes (disclosed).** The drafts are unfiled, and
the PI changed them on 2026-10-25 **after seeing these pilot results**:
the v0.7 readout, probe, number of Gumbel draws, primary statistic
(robust TOST for E-1 and E-2), a new secondary outcome (E-2s) and N
(95 → 150); the v0.8 probe readout (hard → Gumbel, K = 8 averaged
codes); the v0.9 held-out probe (tail split → class-stratified). Each
change is listed in the draft headers with its trigger. The sections
below are in chronological order; earlier sections record what was
true at the time and are kept as history.

**Current position (2026-10-25, 18:45).** All pilots are run: v0.7
B-2 rounds 1–3 (plus the N = 150 recomputation), v0.8 (seeds
2101–2110, after Codex review rounds 1–2 of the driver), v0.9
manipulation check (tail split, then stratified split) and the v0.9
main-grid D-variance pilot (seeds 3101–3105). No confirmatory seed was
run.

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
| v0.7 B-2 round 3 (K = 8 Gumbel draws, robust E-2, switch; PI decisions 2026-10-25 13:12) | 1101–1105 (same seeds) | **run** | 112 s run (about 22 s per seed) + 57 s derivation | `reports/pilots/v07_b2_pilot_r3_seeds1101-1105.json`, `…_r3_derived.json` |
| v0.8 (§7), at 11:00 | 2101–2110 | blocked then (driver missing); seed-0 hard-rank check | — | `v08_hard_rank_diagnostic_seed0.jsonl` |
| v0.8 pilot (§7), after the driver and the review | 2101–2110 | **run**: k* = 13, 10 / 10 complete, N = 20 | 61 min (about 365 s per seed) | `v08_pilot_capture_log.json`, `v08_pilot_quantities.json`, `v08_pilot_n_rule.json` |
| v0.9 manipulation check (§3.3) | 3101–3105 | **run, gate passed** | 35 s | `reports/pilots/v09_manipulation_check_seeds3101-3105.json` |
| v0.9 manipulation check, stratified split | 3101–3105 | **run, gate passed** | 38 s | `v09_manipulation_check_stratified_seeds3101-3105.json` |
| v0.9 main grid and D variance (§7) | 3101–3105 | **run**: σ̂_plan 0.0352, N = 20; FG mostly 0 | 404 s | `v09_pilot_main_grid.json`, `v09_pilot_n_rule.json` |

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

## v0.7 — B-2 pilot, round 3 (PI decisions of 2026-10-25, 13:12)

Decisions, taken after round 2 and the seed-1105 diagnosis below and
disclosed in the v0.7 header: K = 8 Gumbel draws per network with fixed
seeds (`bouba_sens.readout`), a robust E-2 (90 % Hodges–Lehmann CI;
mean-based t-TOST kept as a reported secondary), and a registered
secondary binary outcome E-2s "class-separating switch learned" (post
top-1 share ≥ 0.25 and post between-class share ≥ 0.10; thresholds
justified in v0.7 §4.2 from the no-structure value and the nine
non-switch pilot matrices). Commit `aaf8448`, macM1, log
`reports/pilots/logs/v07_b2_pilot_r3.log`; the training trajectories
and the legacy single-draw readouts are identical to round 2 (same
legacy values in the log).

| Seed | primary ΔMI-hat (K = 8), bit | per-draw SD | single draw (round 2) | post top-1 / between-class share | switch pre → post |
|-----:|------:|------:|------:|------:|:----:|
| 1101 | −0.0177 | 0.050 | +0.0466 | 0.109 / 0.006 | no → no |
| 1102 | −0.0184 | 0.040 | +0.0347 | 0.120 / 0.006 | no → no |
| 1103 | −0.0328 | 0.059 | +0.0694 | 0.114 / 0.006 | no → no |
| 1104 | +0.0047 | 0.033 | +0.0789 | 0.105 / 0.005 | no → no |
| 1105 | +0.6265 | 0.039 | +0.6461 | 0.511 / 0.344 | no → **yes** |

With 8 draws the four non-switch seeds move from +0.035 … +0.079 (one
draw) to −0.033 … +0.005: the round-2 positives were mostly the noise
of a single Gumbel draw. Seed 1105 stays at +0.63 bit.

**Pilot-derived quantities (`reports/pilots/v07_b2_pilot_r3_derived.json`).**

| Rule | per-seed summary | power at N = 95, μ = 0 / −0.02 | N for 0.80 | label |
|------|------|------:|------:|------|
| **robust (registered E-2)** | HL −0.0068, median −0.0177 | **0.744 / 0.746** | 150 (grid 95, 150, 200, …) | **"under-powered"** |
| mean (secondary) | mean +0.1125, SD 0.2877, UCL80 0.4481 | ≈ 0 / ≈ 0 (exact t-TOST) | 690 / 1,381 | "low informativeness" |

Robust planning model: the five pilot values resampled with
replacement, deviations from their pseudo-median inflated by the df = 4
UCL80 factor 1.558, shifted to pseudo-median μ; 20,000 draws at N = 95,
2,000 per grid N, generator 20261025. The model keeps one switch seed
in five, so 20 % of the simulated seeds sit at about +1 bit; that tail,
not the spread of the non-switch seeds, limits the power. N stays 95.

**E-2s in the pilot:** switch after the lesion on 1 of 5 seeds (1105),
exact 95 % CI 0.005–0.716; 0 of 5 before the lesion.

### N = 150 (PI decision 2026-10-25, 13:32) and the robust B-1 power

The PI fixed N = 150 (robust power target 0.80) and moved E-1, E-1b and
E-1c to the robust TOST too. Recomputed at commit `c40eeeb` on macM1
(`reports/pilots/logs/v07_n150_derive.log`):

| Quantity | Value | File |
|---|---:|---|
| B-2 robust power at N = 150, μ = 0 / −0.02 (20,000 draws) | 0.904 / 0.905, "adequately powered" | `v07_b2_pilot_r3_derived_n150.json` |
| B-2 robust rule, P(Equivalent) at θ = +0.05 / −0.05 (size at the bounds, same law) | 0.0035 / 0.0028 | same |
| B-2 mean rule (secondary), exact t-TOST power at N = 150 | 2e-5, "low informativeness" | same |
| B-1 robust, P(E-1) / P(E-1 and E-1b) at the observed per-lock pseudo-medians, N = 150 (2,000 draws) | 1.000 / 0.938 | `v07_b1_robust_power_n150.json` |
| B-1 robust, same with all centres 0 | 1.000 / 0.951 | same |

B-1 planning data are the Q3/Q3+ 10 seeds (no new pilot), resampled as
5-vectors with the df = 9 UCL80 inflation (1.293) and the 0.00625
quantum.

### v0.7 B-1 timing check (seed 0, outside every pool)

`scripts/run_grid.sh` with `SEEDS=0 WORLD=studyforrest LOCK_AFTER=100
K_STEPS=50` on macM1 (`reports/pilots/logs/run_b1_timing.sh`, log
`b1_timing_seed0.log`, commit `aaf8448`): one grid (Phase 1 + 30 lesion
cells) in 182 s. At N = 150 (900 grids) that is about 45.5 h; with B-2
(about 0.9 h) about 46.5 h sequential on macM1. Cell metadata
(`reports/pilots/b1_timing_seed0_T2_audio_floor_metadata.json`) records
`snr_init` 20, `snr_floor` −20, `k_steps` 50, so the floor is reached
at step 50 by the §4.1 formula; the grid outputs do not record the
world class (the CLI falls back silently to the stub without the
cache), which remains an open item for the v0.7 checklist. Seed 0 is a
historical grid seed, not a pilot seed; whether it can stand in for
the "pilot cell" of the checklist is for the PI to decide.

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

## v0.8 — status at 11:00 (historical; superseded by the driver and the Gumbel readout)

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

## v0.9 — manipulation check under the tail split (11:00; historical split)

**Command** (`reports/pilots/logs/run_v09.sh`, log
`logs/v09_manipulation_check.log`):

```
BOUBA_SENS_STUDYFORREST_DATA=<cache> nice -n 10 uv run --frozen python -u \
  scripts/pilots/run_v09_manipulation_check.py \
  --out reports/pilots/v09_manipulation_check_seeds3101-3105.json \
  3101 3102 3103 3104 3105
# to reproduce at a later commit, add --split tail (the default is now
# the class-stratified split of the PI decision of 2026-10-25)
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

## v0.8 pilot (seeds 2101–2110)

Command (`reports/pilots/logs/run_v08_pilot.sh`, log
`reports/pilots/logs/v08_pilot.log`), macM1, commit `aef0988` (driver
as reviewed in Codex round 2; later commits changed only the v0.7
analysis, tests and text):

```
uv run python scripts/run_v08_subspace.py capture \
  --seeds 2101,…,2110 --blocks-dir <runs>/v08_pilot_blocks
uv run python scripts/run_v08_subspace.py pilot \
  --blocks-dir <runs>/v08_pilot_blocks --out reports/pilots/v08_pilot_quantities.json
uv run python scripts/pilots/derive_v08_pilot.py \
  reports/pilots/v08_pilot_quantities.json reports/pilots/v08_pilot_n_rule.json
```

World `StudyforrestRealWorld`; Gumbel readout, each capture the mean
of K = 8 seeded draws; 70 blocks of 16 arms (51–59 s per block, about
365 s per seed, 61 min in all). The raw code matrices (2.2 GB) stay on
macM1 (`~/hypneum-lab/pilots-pr9/runs/v08_pilot_blocks`), not in the
repository.

| Quantity (v0.8 §3 item 3, §7) | Value |
|---|---:|
| k* (median plug-in PR of the 70 pilot Z_pre, rounded half up) | **13** (median 12.72; range 10.74–13.39) |
| invalid cells / undefined blocks / complete seeds | 0 / 1,050; 0 / 70; **10 / 10** |
| q̂ (Clopper–Pearson 80 % bound of 0 / 10) | 0.149 |
| exclusion model | level |
| σ̂ (two-way ANOVA residual SD, df 54) / UCL80 | 0.00293 / **0.00320** |
| median S_split (complete blocks, all 7 levels) | **0.0739** (blocks 0.052–0.077) |
| ĝ = median S_split / σ̂_UCL80 | **23.09** SD |
| N rule: safeguarded P(Supported) at N = 20, trend ĝ + 0.70 SD, 100,000 draws (lower bound) | 0.9070 (0.9052) → **N = 20** |
| runtime at N = 20 | about 2.0 h on macM1 |

Reported, no decision: the block outcomes Y range from −0.0074 to
+0.0159 (median +0.0008), more than ten times below the guard median
S_split; under the frozen planning rule the assumed trend (ĝ + 0.70 SD
≈ 0.076 in Y units) is far above anything seen in the pilot, so N = 20
is the minimum by rule rather than a sign of a large expected effect.
66 % of the Z_pre / Z_post matrices have an eigengap ratio
λ_k* / λ_(k*+1) below 1.1 (the section 5 instability flag). The
finite-sample-corrected participation ratio of Chun et al. is not
implemented and was not computed.

## v0.9 pilot (seeds 3101–3105), stratified split

Command (`reports/pilots/logs/run_v09_pilot.sh`, log
`reports/pilots/logs/v09_pilot.log`), macM1, commit `849d0ef`:
manipulation check with `--split stratified`, then
`scripts/run_v09_decorrelation.py --seeds 3101,…,3105`, then
`scripts/pilots/derive_v09_pilot.py`.

**Manipulation check (stratified split):** Δ_K = 0.1206, p = 1 / 10,001,
**passed** (K_joint 0.130, K_decor 0.010; original anchors 0.234). The
tail-split check of 11:00 (Δ_K 0.1077) is kept as history.

**Main grid and D:** 75 runs (decor / joint / plain × 5 onsets × 5
seeds), 404 s. Held-out accuracy acc_all is 0.373–0.506 in 74 of 75
runs (0.750 once), i.e. near the majority-class rates of the probe
(class 1: 192 / 512 = 0.375; class 2: 240 / 512 = 0.469), and FG is 0 in
most runs. D(s, o) (rows 3101–3105, onsets 0–200):

```
3101:  +0.0020 +0.0020 -0.0020  0.0000 -0.0039
3102:   0.0000 -0.0039 -0.0020  0.0000  0.0000
3103:   0.0000 -0.0020 +0.0078  0.0000 -0.1113
3104:   0.0000  0.0000  0.0000  0.0000  0.0000
3105:   0.0000 -0.0059 +0.0020  0.0000  0.0000
```

ANOVA without replication: MS_seed 0.000422, MS_residual 0.000511
(df 16), σ̂_u² = 0, σ̂ = 0.0226, **σ̂_plan = 0.0352** (× 1.558). N rule
(early profile, μ₀ 0.05, ρ 0, 100,000 draws): P(A+ and C+) = 0.9499 at
N = 20 (lower bound 0.9485) → **N = 20**. Runtime: about 81 s per seed
for the 15 main-grid runs; the full 31-run design per seed is
estimated at about 3 min per seed, about 1 h at N = 20 (the remaining
16 runs were not timed).

Reported, no decision: after 250–450 Phase-1 steps the networks mostly
predict one class on the held-out probe, so fusion gain has almost no
room and D is 0 in 14 of 25 pilot cells; σ̂ is driven by one cell
(−0.111). The rules are unchanged; whether the design can detect a
deficit of the SESOI size with such floor-level fusion gain is a
question for the PI.

## PI decisions of 19:04 (after all pilots) and the runs they asked for

- **Draft headers:** bouba_sens has no filed parent registration
  (Q6JYN covers dream-of-kiki H1–H4; v0.4–v0.6 never filed); each draft
  is now stated as a standalone pre-registration.
- **v0.7 confirmatory seeds:** new block 4001–4150, reserve 4151–4154
  (no other use as a base seed; derived batch seeds s + t overlap
  across pools, an open PI decision, v0.7 §8); the E1 runner
  (`--osf-filed`) and `run_grid.sh` (`OSF_FILED`) refuse them as base
  seeds without a filing reference (a user attestation, not a DOI
  check); the analysis accepts only that roster but has no filing
  gate.
- **World class:** every `train` / `lesion` output records it;
  `aggregate_grid.py --require-world` refuses a grid without it or on
  the stub.
- **SNR pilot cell (v0.7 B-1):** dedicated pilot seed **4201**, outside
  every pool, v0.7 B-1 parameters (Studyforrest, 200 / 100 steps,
  batch 16, HARD gating, `LOCK_AFTER=100`, `K_STEPS=50`), commit
  `1ec2a0a`, 3.6 min for the grid. In all 30 cells the recorded SNR
  starts at +20 dB, reaches −20.00 / −10.00 / +10.00 dB at step 50 and
  holds it to step 99 (means −9.80 / −2.35 / +12.55 dB), and every
  output records `StudyforrestRealWorld`; the aggregate with
  `--require-world StudyforrestRealWorld` passed
  (`reports/pilots/v07_snr_pilot_seed4201_cells.json`, `…_aggregate.json`,
  log `reports/pilots/logs/v07_snr_pilot.log`). The seed-0 timing run
  above is no longer used for this check.
- **Numbering confirmed:** N18–N20, ADR-0023–0025.
- **Exact sign test** (registered sensitivity analysis, no symmetry
  assumption), power at N = 150 under the same planning laws
  (`reports/pilots/v07_sign_power_n150.json`): B-2 P(equivalent) =
  1.000 at μ = 0 and −0.02; at the bounds 0.000 when the law's median
  is at ±0.05, but 0.778 when its pseudo-median is at +0.05 (the
  skewed law has its median well below its pseudo-median, so the two
  tests bound different quantities); B-1 P(E-1) = 1.000 but
  P(E-1 and E-1b) = 0.015 / 0.024 (order-statistic CIs of quantised
  single-lock values rarely fit strictly inside ±0.025).
- **v0.8 and v0.9 redesign:** proposals in
  `docs/osf/redesign-proposals-v08-v09-2026-10-25.md`, with evidence
  from a read-only re-analysis of the stored v0.8 pilot codes
  (`reports/pilots/v08_redesign_evidence.json`: 14 equal eigenvalues
  then a ×117–1,160 cliff; λ₁₃ / λ₁₄ median 1.066; 1 − CKA(Z_pre,
  Z_control) = 0.97). The new pilots (seeds 2201–2210 and 3201–3205)
  wait for the PI's review.

## PI decisions of 19:47 and the pilots they asked for (2026-10-25 / 28)

**Seed namespaces.** Every derived data seed is now
`SeedSequence([code, s, stream, t])` with one code per campaign and
pool (`bouba_sens.seeds`; 7101 / 7102 v0.7 B-1 pilot / confirmatory,
7201 / 7202 B-2, 8101 / 8102 v0.8, 9101 / 9102 v0.9).
`tests/unit/test_seeds.py` enumerates the namespaced seeds the runners
draw (B-1 cells keyed as executed: seed · 10 000 + cell 1–30, per-query
probe at cell + 777, check draws 1–200) and finds none shared by two
pools; integer seeds that are not namespaced (model initialisation and
global RNG s … s + 5, B-1 cell seeds and cell + 777, the K = 8 Gumbel
draw seeds) are disjoint between every pilot and every confirmatory
pool; the only shared ones are the base seeds of the B-1 and B-2
confirmatory campaigns (paired initial networks, confirmed by the PI on 2026-10-28, about 18:05, together with keeping the init and Gumbel seeds un-namespaced but disjoint). Fixed
design constants (held-out split, probe-training allocation,
permutation and simulation generators, init seeds 0–5 of T1 cells) are
shared by every pool on purpose. After the Codex review of 2026-10-28
the B-1 evaluation step also uses the namespace and the cell seeds no
longer depend on the run-wide counter; after the Codex review r7 the `eval --run` probe of the grid is namespaced too (`e0e92db`) and the SNR cell was re-run at that commit (`v07_snr_pilot_ns3_seed4201_*`, superseding ns2). Pilots affected by the earlier
overlap of derived seeds, and what was done:

| Pilot | Overlap with a confirmatory pool (legacy seeds) | Enters a rule? | Action |
|---|---|---|---|
| v0.7 B-1 SNR cell 4201 | yes (batch seeds 4201–4400 vs 4001–4353) | no (a check) | re-run in 7101: same SNR, world class and namespace recorded |
| v0.7 B-2 1101–1105 (rounds 1–3) | no | yes (N, label) | re-run as round 4 in 7201; round 4 replaces round 3 for planning |
| v0.8 2101–2110 | yes (2101–2309 vs 2001–2243) | yes | superseded by the rev. B pilot 2201–2210 in 8101 |
| v0.9 3101–3105 (grid) | yes (3101–3554 vs 3001–3493) | yes | superseded by the rev. B pilot 3201–3205 in 9101 |
| v0.9 manipulation checks | no (anchors and draws disjoint) | yes (gate) | re-run in 9101 with the rev. B pilot |
| seed-0 runs | not a pool seed | no | not re-run |

**v0.7 round 4 (namespace 7201, `680e979`):** ΔMI-hat −0.0058, −0.0431,
+0.0153, −0.0033, −0.0337 bit (HL −0.0139); switch on 1 / 5 (1101);
robust power at N = 150 1.000, size at the bounds 0.023 / 0.003; mean
rule 1.000; N stays 150 (`v07_b2_pilot_r4_derived_n150.json`,
`v07_sign_power_r4_n150.json`). **SNR cell re-run (7101):** floors and
means as before, world class and namespace recorded, `--require-world`
passed.

**v0.8 rev. B (seeds 2201–2210, 8101) — gate failed.** The deterministic
soft fused code is constant to float rounding (total variance ≈
8.6 × 10⁻¹⁵); 0 / 10 complete seeds; no σ̂, no N. The Gumbel code's
variation is most likely sampling noise, but this pilot does not
isolate it: the probe is one contiguous window, single-label on
several seeds (between-class share 0 by construction), and the CKA
between the soft and the Gumbel code (0.02) is computed on matrices
below tolerance. Capture 65 min; the run was
interrupted on 2026-10-25 when the summary stopped on a degenerate
spectrum (log `logs/revb_pilots.log`), fixed in `e7c8a06` and
summarised on 2026-10-28 (`logs/revb_resume.log`).

**v0.9 rev. B (seeds 3201–3205, 9101) — headroom gate failed on the
positive control.** Manipulation check passed (Δ_K 0.121, p 1e-4).
Vision alone decodes the label at held-out balanced accuracy 0.986; the
concatenation of all modalities gains nothing (FG_ctrl mean −0.002); no
N ≤ 40 qualifies (σ_plan 0.177). Details and options:
`docs/osf/redesign-proposals-v08-v09-2026-10-25.md`, section C.

Raw v0.8 rev. B code matrices (4.1 GB) stay on macM1
(`~/hypneum-lab/pilots-pr9/runs/v08_revb_pilot_blocks`); the first v0.8
pilot's matrices (2.0 GB, superseded) and two seed-0 / old SNR grids were
deleted on 2026-10-28 to save disk (their summaries are committed).

## Findings at 11:00 and how they were resolved

The list below is the question list sent to the PI at 11:00; each item
is followed by the decision taken.

1. **v0.7 E-2 primary readout.** The hard readout makes E-2 vacuous
   (finding 1). Options include: keep the Gumbel readout as primary
   (as in ADR-0019), redefine the hard readout (for example the soft
   demodulation probabilities without sampling), or drop E-2. Any
   change is a change to the draft before filing. → *Decided 12:10:
   Gumbel primary; later (13:12) K = 8 draws.*
2. **v0.7 probe label coverage.** A single contiguous 512-frame window
   is single-label on 3 of 5 pilot seeds (finding 2). The probe
   definition, and the reading of the "fewer than k + 1 points per
   class" technical-failure rule (classes present vs all four classes),
   need a decision before filing. → *Decided 12:10: class-balanced
   multi-window probe; the rule text unchanged.*
3. **v0.8 hard probe.** Finding 3 implies a protocol failure by
   construction under the hard probe rule (PI decision 2026-10-23).
   This needs a decision before the driver is written and the pilot
   run. → *Decided 12:10: Gumbel readout; 13:12: K = 8 averaged codes
   (confirmed 13:32).*
4. **Drivers.** The v0.8 driver and the v0.9 arm / grid driver (with
   the train-side fusion gain) must be written, reviewed and merged
   before those pilots can run. → *Decided 12:16: written (commits
   477502b, 8d2966a, 6a49b4a, 267d1a9), reviewed before the pilots.*
5. **v0.7 B-1 pilot cell** for the SNR `metadata.json` check: which
   seed, if any. → *Open. A timing / SNR check was run on seed 0
   (historical, outside every pool), see "v0.7 B-1 timing check".*

## Rev. C pilots (2026-10-28, `76ebbd8`, macM1)

Design and results: `docs/osf/redesign-proposals-v08-v09-revc-2026-10-28.md`,
sections D.4–D.5. In brief:

- **v0.8** (2211–2220): stopped at the learning gate (full-network
  held-out balanced accuracy 0.29–0.33 at every lock).
- **v0.9 CC-film** (3206–3210, 10 fps): the manipulation check passes,
  then the run stopped at the learning gate (onsets 100 and 150 fail).
- **v0.9 Studyforrest** (3206–3210): the manipulation check passes, then
  the run stopped at the learning gate (seed 3207 at chance at every
  onset).

No N was derived. Outputs: `reports/pilots/v08_revc_learning_gate.json`
and `reports/pilots/v09_revc_{ccfilm,studyforrest}_{pilot,n_rule}.json`.
