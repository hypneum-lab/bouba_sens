# ADR-0019b — B-2 re-run with MINE at dimensionally-justified N ; the "probe batch ≥128" recommendation is replaced by N ≳ f(D)

> **Numbering note (2026-10-25, issue #21).** ADR number 0019 is shared
> with `0019-multi-seed-q3-retract.md` (ADR-0019a). Suffixes follow
> first-commit order; file names are unchanged so existing links keep
> working. Index: `docs/adr/README.md`.

**Status:** Accepted — revises the methodology recommendation from ADR-0017
/ paper §6.3. Supersedes the "probe batch ≥128" heuristic with a
dimensionality-grounded criterion N ≳ f(D).
**Date:** 2026-06-03
**Sprint:** E1 + E2 (EXPERIMENT-BACKLOG.md, from external review REVIEW.md)
**Companion to:** ADR-0017 (Sprint 14 retraction), paper §3.2 + §6.3.

## Context

The external reviewer (REVIEW.md ②) flagged that the published
methodology recommendation **"probe batch ≥128"** is under-derived:
MI sample complexity scales with the **dimensionality** of the probe
representation, not with batch size. "n≥128" is a heuristic, not a
principled bound. Backlog E1+E2 asks to:

1. **(E1.1)** compute the actual dimensionality D of the Me3 probe;
2. **(E2)** fix N from a principled criterion N ≳ f(D);
3. **(E1.3)** re-estimate B-2 with MINE at that N, cross-checked
   against Kraskov k-NN and adaptive binning, with bootstrap CIs.

This ADR also resolves the explicit open question left in paper §6.3:
whether the ~0 B-2 reflects the network (no migration) or a property of
the **1-D mean-pooled probe** that hides high-dimensional structure from
the estimators.

## The probe-shape bug that §6.3 / Sprint 17 missed

`AdaptationLoop.lesion_phase` stores the Me3 probe as
`fused.flatten(1).mean(-1)` (`loop.py:189,246`) — a **1-D scalar per
sample**. The full fused representation is `(B, K, d_hidden) = (B, 7, 128)
= (B, 896)`.

Consequence, confirmed empirically (`reports/e1_b2_dimensional_mine.json`,
and visible already in `reports/sprint17_kraskov_vs_mine.csv`): **MINE on
the 1-D mean-pooled code returns exactly 0.0 on every seed** — not because
B-2 is null, but because a learned scalar critic on a single near-constant
feature cannot beat its Donsker-Varadhan floor. The "Kraskov and MINE agree
B-2 ≈ 0" claim in §6.3 / README F3 is therefore **not** a two-estimator
agreement on the network: one of the two estimators is structurally pinned
to 0 by the probe shape. This is a methodological defect in the comparison,
independent of the B-2 verdict.

## E1.1 — measured dimensionality D of the Me3 probe

Probe captured at full `(N, d_fused=896)` (not mean-pooled), LOCK_AFTER=100
+ HARD gating, N=512, 5 seeds. Two effective-dimension estimators on the
post-lesion probe:

| Estimator | per-seed range | mean |
|-----------|---------------:|-----:|
| Participation ratio (linear, covariance eigenspectrum) | 9.65 – 12.96 | **11.67** |
| Two-NN intrinsic dim (Facco 2017) | 10.95 – 13.61 | **12.38** |

So the **ambient probe dimension is 896**, but the **effective / intrinsic
dimension is D ≈ 12** (both estimators agree). The Me3 representation lives
on a ~12-D manifold inside an 896-D ambient space.

## E2 — the criterion N ≳ f(D)

Replacing "n ≥ 128" with an estimator-aware, dimensionality-grounded rule:

- **Plug-in / binning** (alphabet grows exponentially in dimension):
  N ≫ (#bins)^D. Tractable only on a **low-D summary** of the probe
  (the 1-D mean-pool, or a few leading PCs), never the 896-D ambient.
- **k-NN Kraskov / MINE** (sample complexity scales with **intrinsic**,
  not ambient, dimension): the practical requirement is
  **N ≳ c · k · 2^D_int** for density-style methods and
  **N ≳ batch_size and growing with D_int** for MINE.

> **Citation note (2026-11-05, no change to the criterion).** The
> intrinsic-not-ambient argument comes from Abdelaleem, Martini &
> Nemenman 2025 (arXiv:2506.00330), who establish it for neural
> estimators, with N ≳ K_Z²/I (p. 11). The same paper reports KSG
> failing on 500-dimensional data with latent dimension 10 at both
> N = 256 and N = 2¹⁴ (Fig. 10, p. 31), so it does not support
> extending the rule to k-NN Kraskov; for Kraskov the bound above is
> a working assumption. Two further departures are recorded in the
> paper limitations: Me3 feeds discrete labels to an estimator built
> for continuous variables (ties, Kraskov et al. 2004 p. 6; Ross 2014
> is the continuous–discrete estimator), and the two-NN dimension in
> `scripts/spikes/e1_b2_dimensional_mine.py` is a maximum-likelihood
> ratio on all points at N = 512, without the 10 % cut and block
> analysis of Facco et al. 2017 (p. 3–6).

With measured **D_int ≈ 12**, even the favourable MINE scaling puts the
honest sample requirement **well above** the heuristic 128, and above the
N=480 pooling used in Sprint 17. The N=512 run here is at the **floor** of
defensibility for the 1-D summary and is **under-powered** for the full
896-D / D_int≈12 probe. **The "≥128" recommendation is retired.** The
recommendation that ships is:

> **Set N from the estimator's intrinsic-dimension sample complexity, not a
> fixed batch.** Report the measured effective dimension D alongside every
> MI estimate; for k-NN/MINE on the Me3 probe (D_int ≈ 12) treat any
> N ≲ 10³ as a lower bound that can only resolve "structural sign", not a
> calibrated magnitude.

## E1.3 — B-2 re-estimated, three estimators, bootstrap CIs

`scripts/spikes/e1_b2_dimensional_mine.py`, **gaussian synthetic world**
(the studyforrest real bridge needs `BOUBA_SENS_STUDYFORREST_DATA`, absent
in this environment — see Caveat). N=512, 5 seeds, 1000-resample bootstrap
CIs on the per-seed B-2 = MI_post − MI_pre (bits):

| Estimator (probe shape) | mean B-2 (bits) | bootstrap 95% CI |
|-------------------------|----------------:|------------------|
| Kraskov k-NN (1-D mean-pool) | **−0.031** | [−0.069, +0.005] |
| Adaptive binning (1-D, √N bins) | **−0.016** | [−0.041, +0.008] |
| MINE (1-D mean-pool) | **0.000** | [0.000, 0.000] |
| MINE (full 896-D probe) | **−0.038** | [−0.105, +0.002] |

Per-seed B-2 (bits):

| seed | D_PR | D_2NN | Kraskov | binning | MINE 1-D | MINE full |
|-----:|-----:|------:|--------:|--------:|---------:|----------:|
| 0 | 12.96 | 13.61 | −0.073 | −0.032 | 0.000 | −0.016 |
| 1 | 12.33 | 12.68 | +0.033 | +0.013 | 0.000 | −0.007 |
| 2 | 12.40 | 12.21 |  0.000 | +0.012 | 0.000 | +0.011 |
| 3 | 11.01 | 12.44 | −0.082 | −0.062 | 0.000 | −0.171 |
| 4 |  9.65 | 10.95 | −0.034 | −0.011 | 0.000 | −0.008 |

## E1.3b — independent full-sonde replication (`b2_full_sonde.csv`)

A second runner (`scripts/spikes/e1_b2_full_sonde.py`, gaussian world,
5 seeds, `steps=100`, `n_boot=200`) re-estimates B-2 with **all three
estimators on the full 896-D probe** (the E1.3 table above mixed a 1-D
summary for Kraskov/binning with full-D only for MINE). This isolates
the effect of giving every estimator the full-dimensional sonde:

| seed | N | d | Kraskov | MINE (full) | binning (full) |
|-----:|--:|--:|--------:|------------:|---------------:|
| 0 | 240 | 896 | −0.077 | −0.003 | −10.72 |
| 1 | 240 | 896 |  0.000 | +0.012 | −24.77 |
| 2 | 240 | 896 |  0.000 | −0.008 | −12.06 |
| 3 | 240 | 896 | −0.063 | **+0.120** | −8.36 |
| 4 | 240 | 896 | +0.080 | +0.032 | −3.40 |
| **mean** | | | **−0.012** | **+0.031** | **−11.86** |

Three readings:

1. **MINE is no longer pinned to 0.** On the full probe MINE varies
   per seed (−0.008 … +0.120), directly confirming that the exact
   `0.000` of the 1-D mean-pool was a *probe-shape* artefact, not a
   network property.
2. **No consistent positive B-2.** Only seed 3 (+0.120) crosses the
   +0.10 threshold; the MINE-full mean is +0.031 and Kraskov-full
   brackets zero (mean −0.012). The verdict stays **Null** — a single
   seed above threshold out of five is noise, not migration.
3. **Adaptive binning collapses in 896-D** (−3 … −25 bits), as the
   N ≫ (#bins)^D argument predicts: histogram MI is degenerate at
   ambient dimension and is **unusable on the full probe** — it is
   informative only on a low-D summary (the 1-D path of the E1.3
   table). This is corroborating evidence for the N ≳ f(D) criterion.

`b2_full_sonde.csv` is gaussian-world, same caveat as E1.3.

## Decision

1. **B-2 is null / estimator-limited, NOT positive.** On the gaussian
   world at dimensionally-justified N, every estimator's 95% CI brackets
   zero; no estimator crosses the +0.10 threshold on any seed. The
   `(B, 896)` full probe — the "richer probe" §6.3 deferred to v0.2 — does
   **not** unlock a positive B-2: MINE-full mean is −0.038 (CI
   [−0.105, +0.002]), if anything slightly negative. The §6.3 hypothesis
   "migration happens but Kraskov cannot see it on a 1-D probe" is **not
   supported** here: feeding MINE the full-dimensional probe does not
   surface a hidden positive signal.

2. **The "MINE agrees with Kraskov at ≈0" framing is corrected.** MINE-1-D
   is pinned to 0 by probe shape, so it was never independent evidence.
   The genuine cross-estimator statement is: **Kraskov, binning, and
   MINE-on-the-full-probe all bracket zero** — that is the load-bearing
   agreement.

3. **"Probe batch ≥128" is retired** in favour of N ≳ f(D) with D reported.

4. **Real bridges (aligned 2026-11-06, PI decision; issues #74, #77).**
   B-2 is Null on the CC-film bridge (E1.3d) and on the Gaussian world.
   On the Studyforrest bridge it is **not conclusive**: the contiguous
   probe holds a single label class in 4 of 5 seeds (#77), so MI is
   zero by construction (E1.3c-bis), and with the stratified probe of
   the amendment the learning gate fails on every seed, so B-2 is not
   measurable there (E1.3c-ter). The earlier statement that the null
   does not depend on which half of the modality set carries real
   signal is **no longer supported**: only one real bridge gives an
   informative measurement.

## Pre-registration fidelity

- **No threshold changes** (B-2 still +0.10).
- **No B-2 verdict change**: it was Null (README F3); it stays Null. This
  ADR *strengthens* the null by removing the spurious MINE-1-D "agreement"
  and showing the full-dimensional probe does not rescue a positive B-2.
- The retired item is a **methodology recommendation** ("≥128"), exactly
  the under-derived claim the reviewer flagged — not a registered hypothesis.

## E1.3c — real-bridge promotion (`--world studyforrest`, OpenNeuro ds000113)

The one remaining step flagged in the original caveat below was executed.
The real Studyforrest cache was provisioned from **OpenNeuro `ds000113`**
via the public S3 REST API (`scripts/fetch_studyforrest_sample.py
--source openneuro`, 9000 frames @ 10 Hz). The dead `studyforrest.org`
phase-1 URL (404) was retired in favour of the live S3 bucket.

**Channel provenance (mixed, labelled verbatim in the cache MANIFEST):**

| modality | provenance | real? |
|----------|-----------|:-----:|
| force | phase-1 `ses-forrestgump` physio, 4 columns (trigger, respiratory, cardiac, SpO2) at 100 Hz (Hanke et al. 2014, *Sci. Data* 1:140003, p. 11); extraction assumed 500 Hz and `[trigger, cardiac, respiratory]`, so "cardiac" = respiration, "resp" = cardiac, 5× time compression in the first cache (issue #74, 2026-11-06; corrected cache in E1.3c-bis) | **REAL** (mislabelled in the first cache) |
| gravity | `derivatives/motion/*_bold_moco.txt` rotation regressors | **REAL** |
| tactile | `scenes.csv` scene / DAY-NIGHT / INT-EXT one-hot → 32-D proj | **REAL** |
| labels | `scenes.csv` scene index, quantised to 4 | **REAL** |
| audio | `german_audio_description.csv` narration-timing → 128-D | **PROXY** |
| vision | `scenes.csv` cut structure → (16,16) | **PROXY** |

*Note (2026-11-06, issue #74):* the cache
`data/studyforrest_real/force.pt` used for the E1.3c values below was
built with the earlier extraction code (channels swapped, about 4500 s
of physiology in 9000 frames). The extraction was fixed (PR #75) and
the values were recomputed on 2026-11-06 with the corrected cache; see
E1.3c-bis after the E1.3c verdict.

audio + vision are **annotation-derived proxies**, not the perceptual
stimuli: the Forrest Gump soundtrack/movie are copyrighted and are **not
in OpenNeuro**. So 4/6 channels carry real ds000113 signal; audio/vision
are documented proxies. The world loads as `StudyforrestRealWorld`
(mode `real5`, verified — not the mock stub).

B-2, N=512, 5 seeds, 500-resample bootstrap, effective dim **D ≈ 12.5**
(PR 12.9, two-NN 12.5 — matching gaussian's D ≈ 12):

| Estimator (probe shape) | gaussian mean | **real-bridge mean** | **real 95% CI** |
|-------------------------|--------------:|---------------------:|-----------------|
| Kraskov k-NN (1-D mean-pool) | −0.031 | **−0.00097** | [−0.0029, +0.000] |
| Adaptive binning (1-D, √N bins) | −0.016 | **−0.0024** | [−0.0072, +0.000] |
| MINE (1-D mean-pool) | 0.000 | **−2.0e-7** | [−6.6e-7, +8.3e-8] |
| MINE (full 896-D probe) | −0.038 | **−1.9e-5** | [−5.7e-5, −1.8e-9] |

Per-seed real-bridge B-2 (bits):

| seed | D_PR | D_2NN | Kraskov | binning | MINE 1-D | MINE full |
|-----:|-----:|------:|--------:|--------:|---------:|----------:|
| 0 | 13.70 | 12.03 | +0.0000 | +0.0000 | +0.0e0 | +0.0e0 |
| 1 | 12.81 | 11.61 | −0.0049 | −0.0120 | +0.0e0 | −9.5e-5 |
| 2 | 13.24 | 12.77 | +0.0000 | +0.0000 | +0.0e0 | +0.0e0 |
| 3 | 12.31 | 12.85 | +0.0000 | +0.0000 | −0.0e0 | −0.0e0 |
| 4 | 12.48 | 13.18 | +0.0000 | +0.0000 | −1.0e-6 | −0.0e0 |

**Verdict on the real bridge = identical to gaussian: B-2 Null.** Every
estimator's 95% CI brackets zero; no estimator crosses +0.10 on any seed.
The probe-shape conclusions transfer unchanged (MINE-1-D pinned at the DV
floor; full-probe MINE surfaces no positive signal, mean −1.9e-5). The
real-bridge magnitudes are *smaller* than gaussian's — expected, since the
lesioned `audio` channel is an annotation proxy carrying less
audio-specific information to migrate — but the qualitative null is the
same. Artefact: `reports/e1_b2_dimensional_mine_studyforrest.json`.

### E1.3c-bis — recompute with the corrected phase-1 cache (2026-11-06)

The cache was regenerated on macM1 with the fixed extraction
(`fetch_studyforrest_sample.py --source openneuro --duration-seconds
900 --force`). New `force.pt` SHA-256 `d96ad636…b9e9` (first cache
`890b90ae…d773`); the five other tensors are byte-identical. Welch
spectrum over 900 s at 10 Hz: "cardiac" peaks at 1.37 Hz and
"respiration" at 0.30 Hz (first cache: 1.22 Hz and 3.88 Hz, consistent
with the 5× time compression). Same command as E1.3c (`--world studyforrest --seeds 5
--lock-after 100 --steps-train 200 --steps-lesion 100 --batch-size 512
--bootstrap 500`), on macM1 under the heavy lock.

| Estimator (probe shape) | first cache (#74) | **corrected cache** | **corrected 95% CI** |
|-------------------------|------------------:|--------------------:|----------------------|
| Kraskov k-NN (1-D) | −0.00097 | **−0.00022** | [−0.00066, 0.000] |
| Adaptive binning (1-D) | −0.0024 | **+0.00011** | [0.000, +0.00032] |
| MINE (1-D mean-pool) | −2.0e-7 | **+2.4e-7** | [+1.1e-7, +3.9e-7] |
| MINE (full 896-D probe) | −1.9e-5 | **−1.3e-4** | [−3.8e-4, +2.9e-8] |

D_PR 12.98, D_2NN 12.46 (first cache 12.91 / 12.49). With the first
cache (#74) the values were those of the E1.3c table; recomputed on
2026-11-06 with the corrected cache. A same-day replay on the first
cache reproduces Kraskov and binning exactly and every value of seeds
1–4 (seed 0 differs in D and MINE). No estimator's B-2 exceeds the
preregistered +0.10 threshold on any seed. Paired label-permutation
null (1000 draws, same permutation on the pre- and post-lesion probe,
mean over 5 seeds, `scripts/spikes/e1_b2_paired_null.py`): Kraskov
observed 0.000, null median 0.000, P99 0.0044, p = 0.62; binning
observed −0.0011, null median −0.00003, P99 0.0036, p = 0.79 (first
cache p = 0.75 / 0.70). MINE is not in the null (one fit per draw).

*Reading.* The probe batch is a contiguous slice of 512 frames
(51.2 s); the label is the scene index quantised to 4. In 4 of 5 seeds
it holds a single class (`[0,512,0,0]` or `[0,0,512,0]`), in seed 1
500 and 12; `labels.pt` is identical in both caches. MI with a
constant label is zero for any representation, so B-2 is 0 by
construction on those seeds and the measurement cannot fail. The
Phase-1 loss moves from 1.33–1.40 to 0.87–0.92 (ln 4 = 1.39), but
training batches are drawn the same way and reader accuracy was not
measured, so the learning gate is not established. B-2 on the real
bridge is **not conclusive**, with either cache.

The class of the E1.3c result changes from "Null" to a measurement
that is not informative; the question was left to the PI (issue #74),
who decided on 2026-11-06 to align the Decision (item 4) and to rerun
with a stratified probe (amendment and E1.3c-ter below). Artefacts:
`reports/e1_b2_dimensional_mine_studyforrest_phase1fix.json`,
`…_oldcache_replay.json`, `reports/e1_b2_paired_null_studyforrest_*.json`,
logs and cache manifests in `reports/b2_phase1_recompute_20261106/`.

## Amendment 2026-11-06 — probe sampling

**Status: exploratory.** This is a change of method informed by reading
the results of PR #76 (E1.3c-bis) and issue #77 (single-class probe).
It is not preregistered, and nothing measured under it can carry a
confirmatory claim. The rule below is written and committed before any
run that uses it; the code and the run logs come in later commits.

**Root cause addressed.** The B-2 probe is a contiguous slice of
512 frames (`scripts/spikes/e1_b2_dimensional_mine.py:98`, world
`sample()` at `src/bouba_sens/world/studyforrest.py:146-148`). The
Studyforrest label is the scene index quantised to 4, so a 51.2 s slice
lies inside one long scene class in 4 of 5 seeds
(`reports/b2_phase1_recompute_20261106/null_newcache.log`, lines 2-6).
MI with a constant label is zero for any representation.

**Rule (a priori).**

1. *Probe draw.* N = 512 frames from the whole cache (9000 frames for
   Studyforrest), stratified by label class, without replacement.
   Allocation: equal share ⌊512 / C⌋ for the C classes present in the
   cache (128 for C = 4). A class with fewer frames than its share gives
   all its frames; the shortfall is shared equally among the classes
   that still have frames, repeated until 512 frames are allocated or
   the cache is exhausted; an integer remainder goes to the classes in
   increasing label order. The per-class counts of the cache and of the
   probe are written to the output JSON. Inside each class, frames are
   picked by a uniform permutation from a `torch.Generator` seeded with
   the probe seed `seed - 99_999`, as before; the selected indices are
   sorted. Training and lesion batches are unchanged (contiguous
   slices): only the probe changes.
2. *Paired null.* Same probe (same function, same seed). Label
   permutation inside the probe, 1000 draws, the same permutation on the
   pre- and post-lesion probe, statistic = mean B-2 over seeds,
   one-sided p = (k + 1) / (1000 + 1), null seed 20261106 as before.
3. *Learning gate.* Reader accuracy = accuracy of the trained
   integration head (argmax of its logits on the fused code) on the
   probe, after Phase 1 and before the lesion, taken from the same
   forward pass that gives the pre-lesion probe (no extra RNG draw). A
   seed passes if its reader accuracy is at least the majority-class
   rate of its probe plus a margin of **0.10** (absolute; 0.35 for a
   balanced 4-class probe, about five binomial standard errors above
   0.25 at N = 512). The post-lesion reader accuracy on the clean probe
   is reported, not gated.
4. *Classification.* A seed that fails the gate is "not measurable";
   its B-2 is reported but not used. If fewer than 3 of 5 seeds pass,
   B-2 on that cache is "not measurable". Otherwise the mean B-2 over
   the passing seeds is compared with the unchanged preregistered
   threshold **+0.10 bit**, and the paired null p is given for the same
   seeds. Allowed wording: "exceeds / does not exceed the preregistered
   threshold", "not conclusive", "not measurable". Not used: confirmed,
   positive, refuted.
5. *Reproduction.* `--probe-sampling contiguous` (default) reproduces
   the earlier runs; `--probe-sampling stratified` applies this rule.
   It needs a world that exposes `dataset_labels` and `gather`
   (Studyforrest real bridge); other worlds are out of scope here.

**CC-film and Gaussian worlds (not rerun).** CC-film labels are
per-frame quartiles over 600 frames (`scripts/extract_ccfilm_features.py:157-164`),
so any contiguous 512-frame slice holds 62 to 150 frames of each class;
the Gaussian world draws i.i.d. samples. Section E1.3d used the 2-fps
cache `data/ccfilm_real` (600 frames, this ADR E1.3d; commit `fb7af00`
of 2026-06-04). The 10-fps cache `ccfilm_real_10fps` (5,965 frames) was
built on 2026-10-28 for the v0.9 pilots (commit `33b266d`, branch
`docs/osf-prereg-2026-10`, not an ancestor of this branch), and no
script or `reports/MANIFEST.json` entry on this branch refers to it
(the E1.3d JSON artefact `reports/e1_b2_dimensional_mine_ccfilm.json`
is not tracked, so its run log cannot be checked). The bound above
therefore applies to E1.3d, on the evidence of dates and frame counts.

### E1.3c-ter — stratified probe under the amendment (2026-11-06, exploratory)

Same command as E1.3c-bis plus `--probe-sampling stratified`, on the
corrected cache and on the first cache, macM1, commit `540e483`, heavy
lock (`reports/b2_stratified_probe_20261106/`). Cache class counts are
[180, 3480, 4280, 1060] (9000 frames, both caches); every probe holds
[128, 128, 128, 128].

| Estimator (probe) | contiguous, corrected (#76) | stratified, corrected | stratified, first cache |
|-------------------|----------------------------:|----------------------:|------------------------:|
| Kraskov k-NN (1-D) | −0.00022 | +0.0041 [−0.044, +0.044] | −0.020 [−0.070, +0.033] |
| Adaptive binning (1-D) | +0.00011 | +0.026 [0.000, +0.060] | −0.0021 [−0.013, +0.008] |
| MINE (1-D mean-pool) | +2.4e-7 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| MINE (full 896-D probe) | −1.3e-4 | +0.14 [−0.014, +0.44] | −0.0073 [−0.014, −0.0015] |

Means over 5 seeds, bits, 500-resample bootstrap 95 % CI. The
stratified MINE-full mean on the corrected cache is carried by seed 2
(+0.74; the four other seeds lie between −0.018 and −0.001; D_PR of
seed 2 is 5.1 against 12.3–13.4 for the others).

*Learning gate.* Reader accuracy before the lesion is **0.250 on all
10 seeds** (5 per cache), equal to the majority-class rate 0.25 of a
balanced probe: the trained head gives the same class to every probe
frame. The gate (0.35) fails on every seed. After the lesion the
reader accuracy is 0.25 on 9 seeds and 0.365 on seed 3 of the first
cache. The Phase-1 loss still falls from 1.32–1.40 to 0.86–0.91
(ln 4 = 1.39). Training batches are contiguous slices, so most of
them hold one class; a loss below ln 4 together with a constant
prediction on a balanced probe is consistent with a head that tracks
the class of the current scene rather than reading the class from the
input. This reading is an inference; it was not tested.

*Paired null* (`e1_b2_paired_null.py --probe-sampling stratified`,
1000 draws, all 5 seeds; the null run recaptures the probe and its
observed B-2 differs from the main run because the Gumbel
demodulation and the MINE fits use the global random state). Corrected
cache: Kraskov observed −0.072, null median −0.0012, P99 0.036,
p = 1.0; binning observed −0.089, null P99 0.027, p = 1.0. First
cache: Kraskov observed +0.036, null P99 0.038, p = 0.013; binning
observed +0.038, null P99 0.025, p = 0.001. Per-seed observed B-2
moves between the two runs by up to 0.4 bit (seed 4, corrected cache:
Kraskov +0.022 in the main run, −0.388 in the null run), so the label
permutation null does not cover the run-to-run variation of the
representation. These p values are reported, not interpreted.

*Classification under the amendment.* 0 of 5 seeds pass the learning
gate on either cache: B-2 on the Studyforrest bridge is **not
measurable** with the stratified probe, with either cache. No mean
B-2 exceeds the preregistered +0.10 threshold; one seed (corrected
cache, seed 2, MINE-full +0.74) exceeds it, on a network that does
not pass the gate. With the contiguous probe (E1.3c-bis) the result
stays **not conclusive** (single-class probe, #77). Artefacts:
`reports/e1_b2_dimensional_mine_studyforrest_stratified_{newcache,oldcache}.json`,
`reports/e1_b2_paired_null_studyforrest_stratified_{newcache,oldcache}.json`.

## E1.3d — CC-film real-AV bridge (`--world ccfilm`, Big Buck Bunny CC-BY)

E1.3c left one honest gap: on the Studyforrest bridge, `audio` and `vision`
were **annotation-derived proxies**, not the perceptual stimuli (the Forrest
Gump soundtrack/movie are copyrighted and absent from OpenNeuro). This
section adds a **complementary** bridge that closes exactly that gap, at a
different cost.

The source is **Big Buck Bunny** (`BigBuckBunny_320x180.mp4`, Blender
Foundation 2008, **CC-BY 3.0**), fetched from `download.blender.org`
(`scripts/extract_ccfilm_features.py`, 300 s trimmed = 600 frames @ 2 fps).
Audio and vision are **the perceptual signal itself**:

- `audio` (N,128): 128-D **log-mel spectrogram** of the soundtrack —
  `scipy.signal.stft` (n_fft=1024, hop=256, mono 16 kHz) → an **HTK mel
  filter-bank written from scratch** (`hz_to_mel`/`mel_to_hz` + triangular
  bins), pooled onto the frame grid, per-band standardised. No
  torchaudio/librosa.
- `vision` (N,16,16): decoded movie frames (PIL) → per-frame **16×16
  mean-intensity patch grid**, normalised to [0,1].
- `labels` (N,): **REAL shot-cuts** — per-frame `|frame_t − frame_{t-1}|`
  mean, quantised by quartiles into 4 classes (150/150/150/150 populated).

**Channel provenance (labelled verbatim in the cache MANIFEST):**

| modality | provenance | real? |
|----------|-----------|:-----:|
| audio | 128-D log-mel of the soundtrack (scipy STFT + scratch mel bank) | **REAL** |
| vision | (16,16) mean-intensity grid of decoded frames | **REAL** |
| labels | frame-difference shot-cut energy, quartile-quantised to 4 | **REAL** |
| tactile | RNG(seed=0) stream modulated by audio energy-flux envelope | **SYNTHETIC** |
| gravity | same construction (3-D) | **SYNTHETIC** |
| force | same construction (6-D) | **SYNTHETIC** |

A film has **no proprioceptive channel**, so tactile/gravity/force are
explicitly synthetic (seeded, flux-coupled for temporal coherence, **not
measured signal**). The world loads as `CCFilmWorld` (mode `real_av`,
verified in the run logs — **not** the mock stub).

B-2, N=512, 5 seeds, 500-resample bootstrap, effective dim D_PR ≈ 7.6 /
two-NN ≈ 13.1:

| seed | D_PR | D_2NN | Kraskov | binning | MINE 1-D | MINE full |
|-----:|-----:|------:|--------:|--------:|---------:|----------:|
| 0 | 13.02 | 12.53 | −0.0800 | −0.0691 | +0.0e0 | −1.9e-2 |
| 1 |  2.82 | 14.29 | +0.0000 | +0.0144 | +0.0e0 | −2.4e-2 |
| 2 | 13.44 | 13.20 | −0.0274 | +0.0022 | +0.0e0 | +1.3e-2 |
| 3 |  2.72 | 12.32 | +0.0069 | +0.0420 | +0.0e0 | +1.3e-2 |
| 4 |  6.06 | 13.05 | −0.0579 | +0.0416 | +0.0e0 | +4.2e-3 |

| Estimator | mean | 95% CI |
|-----------|-----:|--------|
| Kraskov (1-D) | −0.0317 | [−0.0633, −0.0020] |
| binning (1-D) | +0.0062 | [−0.0327, +0.0363] |
| MINE (1-D) | 0.0000 | [0.0000, 0.0000] |
| MINE (full N×d) | −0.0027 | [−0.0159, +0.0110] |

**Verdict on the CC-film real-AV bridge = B-2 Null.** No estimator's 95% CI
brackets a positive +0.10 effect; MINE-1-D is pinned at the DV floor
(probe-shape artefact, exactly as on gaussian/Studyforrest), full-probe MINE
surfaces no positive signal (mean −0.003), binning brackets zero. The
**only** CI excluding zero is Kraskov, and it excludes it on the **negative**
side (mean −0.032) — i.e. lesion-adapted MI ≤ pre-lesion MI, the opposite of
migration. So even with **genuinely real audio+vision** driving the
representation, B-2 does not flip: the qualitative null transfers a *third*
time, now under real perceptual stimuli.

**What this bridge buys, and what it costs (it is NOT a Studyforrest
replacement):**

- **Gains** the perceptual core Studyforrest's bridge lacked — `audio` and
  `vision` are *real signal*, not annotation proxies. This is the single
  strongest answer to the E1.3c "audio/vision = proxy" caveat: the B-2 null
  is now observed with the actual stimuli in the loop.
- **Loses** Studyforrest's **biological anchoring** entirely — there is no
  fMRI subject, no physio, no head-motion. The proprioceptive channels
  (tactile/gravity/force) are **synthetic**, the inverse of Studyforrest
  (real proprio, proxy AV).
- It is therefore a **different trade-off bridge**, not a strict upgrade.
  Read the two together: Studyforrest contributes real proprioception +
  biological grounding with proxy AV; CC-film contributes real AV with
  synthetic proprioception. **B-2 is Null across both**, plus gaussian — the
  null is robust to which half of the modality set is real.
  *Note (2026-11-06, #77):* this sentence is superseded by Decision
  item 4: the Studyforrest measurement is not conclusive, so only the
  CC-film bridge and the Gaussian world support the null.

Artefact: `reports/e1_b2_dimensional_mine_ccfilm.json`. Cache + per-channel
SHA256/provenance: `data/ccfilm_real/MANIFEST.md`.

## Caveat — world substitution (RESOLVED for the verdict; proxy limit remains)

*Original caveat (kept for history):* the headline B-2 figures were first
produced on the **gaussian synthetic world** because the
`BOUBA_SENS_STUDYFORREST_DATA` cache was absent. The *methodological*
findings — (a) the probe-shape bug pinning MINE-1-D to 0, (b) D_int ≈ 12 ≪
ambient 896, (c) the N ≳ f(D) criterion, (d) full-D MINE does not rescue
B-2 — are world-agnostic and stand.

*Resolution (E1.3c above):* the cache is now provisioned from OpenNeuro
ds000113 and the script was re-run with `--world studyforrest`. The B-2
**verdict is confirmed Null on the real bridge** — the gaussian-only
caveat is **lifted for the verdict**. The gaussian magnitudes are retained
above as the synthetic reference and are not overwritten.
*Note (2026-11-06, #77):* withdrawn for Studyforrest; see E1.3c-bis,
E1.3c-ter and Decision item 4.

*Remaining qualification:* the real bridge carries **real** signal for
force/gravity/tactile/labels but **annotation-derived proxies** for
audio/vision (the perceptual stimuli are copyrighted and outside
OpenNeuro). The force channel was mislabelled and
time-compressed in the first cache (issue #74); it is corrected in the
cache used for E1.3c-bis. This is a genuine real bridge for the proprioceptive +
scene-structure channels, **not** a full perceptual-stimulus replication.
The B-2 null holds with this limitation stated.

## Artefacts

- `scripts/spikes/e1_b2_dimensional_mine.py` — E1/E2 runner (captures full
  `(N, d_fused)` probe, computes D, runs all four estimators + bootstrap).
- `reports/e1_b2_dimensional_mine.json` — gaussian per-seed output.
- `reports/e1_b2_dimensional_mine_studyforrest.json` — **real-bridge**
  per-seed output (E1.3c, OpenNeuro ds000113).
- `reports/e1_b2_dimensional_mine_ccfilm.json` — **CC-film real-AV bridge**
  per-seed output (E1.3d, Big Buck Bunny CC-BY).
- `scripts/extract_ccfilm_features.py` — builds the `data/ccfilm_real/`
  cache: REAL mel-128 audio + REAL 16×16 vision + REAL shot-cut labels +
  SYNTHETIC proprioception. `data/ccfilm_real/MANIFEST.md` carries SHA256s,
  the CC-BY source URL, and per-channel real-vs-synthetic provenance.
- `src/bouba_sens/world/ccfilm.py` — `CCFilmWorld` (mode `real_av`); wired
  into `cli._build_world` under the name `ccfilm`.
- `scripts/fetch_studyforrest_sample.py` — `--source openneuro` provisions
  the real ds000113 cache from S3; `--source surrogate` keeps the offline
  non-scientific fallback. Cache `MANIFEST.md` carries SHA256s + real S3 URLs
  + per-channel provenance (real vs proxy).

## Cross-references

- REVIEW.md ② / open question 2 — the under-derived "≥128" recommendation.
- ADR-0017 §future-directions — "MINE / InfoNCE replacement … test whether
  ~0 B-2 is the network or the estimator". This ADR closes that with: it is
  the network *and* a probe-shape artefact in the prior MINE comparison.
- paper §3.2 (probe definition) + §6.3 (Kraskov vs MINE) — updated in
  lockstep.
