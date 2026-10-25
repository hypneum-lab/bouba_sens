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
| force | cardiac (pulse-ox) + respiration physio, 500 Hz (`recording-cardresp`) | **REAL** |
| gravity | `derivatives/motion/*_bold_moco.txt` rotation regressors | **REAL** |
| tactile | `scenes.csv` scene / DAY-NIGHT / INT-EXT one-hot → 32-D proj | **REAL** |
| labels | `scenes.csv` scene index, quantised to 4 | **REAL** |
| audio | `german_audio_description.csv` narration-timing → 128-D | **PROXY** |
| vision | `scenes.csv` cut structure → (16,16) | **PROXY** |

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

*Remaining qualification:* the real bridge carries **real** signal for
force/gravity/tactile/labels but **annotation-derived proxies** for
audio/vision (the perceptual stimuli are copyrighted and outside
OpenNeuro). This is a genuine real bridge for the proprioceptive +
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
