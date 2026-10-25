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
| bouba_sens commit (code that ran) | `b09070bc6af4a78cd1ea8f57d20918f2a443f84d` (PR branch head) plus the four new scripts in `scripts/pilots/` (this commit) |
| nerve-wml | tag `v1.7.0`, `c36822aac38dc83e2c9dd14f504948d0d17267eb` (the CI pin, `.github/workflows/ci.yml`) |
| Host | macM1: Apple M1 Max, 32 GB, macOS 27.0.1 (26A434), CPU only, `nice -n 10`, one job at a time |
| Python / torch / numpy | 3.14.4 / 2.11.0 / 2.4.4 (`uv run --frozen`, `uv.lock` of the branch; local uv rewrote only the lock-file revision header, not committed) |
| World cache | `StudyforrestRealWorld`, 9,000 frames, label counts [180, 3480, 4280, 1060], 10 label runs (median run 530 frames) |
| Date (UTC) | 2026-10-25 10:53–11:01 |

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
| v0.7 B-2 (§4.3) | 1101–1105 | **run**; primary readout degenerate (see finding 1) | 57 s total; 10–13 s per seed | `reports/pilots/v07_b2_pilot_seeds1101-1105.json`, `…_derived.json`, `v07_b2_probe_diagnostic.jsonl` |
| v0.8 (§7) | 2101–2110 | **blocked**: driver missing (draft filing note, §9) | — | none; structural check on seed 0 only (`v08_hard_rank_diagnostic_seed0.jsonl`) |
| v0.9 manipulation check (§3.3) | 3101–3105 | **run, gate passed** | 35 s | `reports/pilots/v09_manipulation_check_seeds3101-3105.json` |
| v0.9 variance components of D, runtime (§7) | 3101–3105 | **blocked**: driver for arms / grids missing (draft filing note, §9) | — | none |

## v0.7 — B-2 pilot

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
| hard (primary, registered) | 0 | 0 | undefined (degenerate: all values identical) | — | none assignable (see finding 1) |
| Gumbel (E-2g, descriptive only) | 0.01554 | 0.02420 | 1.000 / 1.000 | 4 / 6 | "adequately powered" (descriptive) |

N stays 95 (§7). The Gumbel row is not the registered computation
(§6: "the E-2 power is computed on the hard readout"); it is reported
because the hard row is undefined.

**Runtime (§7).** 57 s for the five seeds on macM1 (seed 1101: 13 s
including start-up and the world check; 1102–1105: 10–11 s each).
At 11 s per seed, the 95 confirmatory B-2 runs take about 18 min on
this host.

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
