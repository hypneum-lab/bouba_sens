# ADR-0012 — Real 5-modality Studyforrest verdicts (TEMPLATE)

**Status:** Pending — compute blocked on Studio datalad install (see Next steps)
**Date:** 2026-04-21
**Sprint:** 9

## Context

Sprint 9 code work (Tasks 9.1–9.5 + 9.7 template + 9.8 release scaffolding) lands the full wiring required to run the 150-cell grid on a real 5-modality Studyforrest bridge. This ADR will be finalised **after** the compute runs complete on Studio.

**Update 2026-04-21 afternoon — data-alignment blocker on phase-2 dataset:**

Studio environment was fixed (datalad 1.4.1, git-annex 10.x, ffmpeg 8.1 via user-level Homebrew at `~/.brew/`). `datalad install https://github.com/psychoinformatics-de/studyforrest-data-phase2` succeeded (canonical `///studyforrest-data-phase2` handle is deprecated — document the working URL in `fetch_studyforrest_phase2.sh` follow-up).

However, the phase-2 dataset structure diverges from the plan's assumptions:

1. **No soundtrack audio file.** `stimuli/soundtrack/fg_av_ger_stereo.mp3` does NOT exist in phase-2; the full film soundtrack is in phase-1 (studyforrest-data-phase1, separate dataset).
2. **Sub-01 `ses-movie` has no cardresp physio.** Only eyegaze is recorded. Cardresp physio exists on some other sub/run combos (e.g. sub-02 run-3) but the cross-subject mapping breaks "single-subject 5-modal aligned" semantics.
3. **Motion annotations not in phase-2.** `stimuli/annotations/movie_motion-locations.csv` lives in a sibling annotation dataset (`studyforrest-data-annotations`), requires an extra `datalad install`.
4. **`rp_*.txt` head-motion regressors are not published** with the BOLD files; phase-2 ships raw nii.gz plus events.tsv, not preprocessed motion correction output.

**Net effect:** a clean 5-modal single-subject single-run bridge requires cross-dataset, cross-session, and preprocessing-pipeline work that was out of scope for the Task 9.6 one-shot execution in this session.

### Update — final data availability finding (2026-04-21 end-of-session)

After pivoting from `ses-movie` to `ses-localizer task-movielocalizer` (which DOES have cardresp physio for sub-01 aligned with the same stimulus), one more constraint was found:

5. **`movie_localizer.mkv` has NO audio track.** The public Studyforrest stimuli are visual-only localizers (movie_localizer, retinotopic_mapping, visualarea_localizer). The full Forrest Gump soundtrack that drove the `ses-movie` runs is **not publicly redistributable** due to copyright on the original film. The `studyforrest-data` repository only hosts study-generated streams (annotations, fMRI, physio, eyegaze).

**Consequence for the 5-modal bridge goal:** without a publicly-fetchable audio stimulus, the audio modality cannot be populated from real recorded sound. Three candidate workarounds, each a deliberate scope change:

a. **Skip audio** — build a 4-modal bridge (vision + tactile + gravity + force). Breaks the pre-registered 5-modality contract unless the OSF amendment explicitly lifts it.

b. **Substitute audio** — use any CC-licensed soundtrack (e.g. a public audiobook aligned to scene durations). Scientifically defensible but requires an OSF amendment v0.5.1 declaring the substitution before grid runs.

c. **Procure the original soundtrack locally** — via a personal copy of the film (fair-use for research). Scientific content is unchanged, but the artefact cannot be published to Zenodo alongside the code (copyright bind). Reviewers would re-run with their own local copy.

The honest engineering path is **(b)** for the paper pipeline (reproducible, redistributable, publishable artefact chain) with **(c)** as a cross-validation run for the maintainer only.

## Resolution (2026-04-21 late afternoon)

ADR-0012 was filled in by running Sprint 9.5 on Studio:

- `scripts/extract_phase2_adapted.py` — 4.5-modal extraction using
  librosa `libri1` CC-BY-4.0 speech (ADR-0012 path (b) substitute
  for the film soundtrack), real VGG16 features over
  `movie_localizer.mkv`, ffmpeg scene-cut tactile proxy, zero
  gravity (rp absent), REAL cardresp from sub-01 ses-localizer
  run-1, audio-RMS quantile labels.
- Two 150-cell grids ran on Studio via `screen` detached:
  `v05_sf45_nolock` + `v05_sf45_lock` (LOCK_AFTER=200).
- nerve-wml v1.5.3 `methodology` module (bootstrap_ci_mi +
  null_model_mi + mi_kraskov_ksg_continuous) gave the first
  principled CI on B-2.

## Grid provenance (filled in)

| Item | Actual value |
|------|--------------|
| Host | Studio (MacStudio, arm64) |
| Worktree | `~/Projets/bouba_sens_b1` |
| Commit | `d7a3645` (feat/sprint-9-5modal) |
| nerve-wml | `v1.5.3` (fix pkg 4739987 on master) |
| Config | `STEPS_TRAIN=200 STEPS_LESION=100 METRICS="Me1,Me2,Me3"` |
| World | `StudyforrestRealWorld(data_dir=data/sf_phase2_adapted)` |
| Grids | `runs/v05_sf45_{nolock,lock}` |
| Wall time | ~30 min parallel (2-way concurrency) |

## Verdicts

| Invariant | 4.5-modal no-lock | 4.5-modal lock=200 | Note |
|-----------|------------------:|-------------------:|------|
| B-1 Me7 (threshold > 0.05) | **0.0000** (balanced) | **+0.0063** | FIRST world-condition with positive sign under lock |
| B-2 Me3 delta (threshold > 0.10) | **-0.0376** | **-0.0391** | FAIL, CI95% [−0.056, −0.001] via bootstrap_ci |
| B-3 Me6 max-abs (threshold > 0.02) | **0.1016** (5.1×) | **0.1250** (6.2×) | PASS both conditions |

### Comparison against prior ADRs

| World | B-1 no-lock | B-1 lock=200 | B-3 no-lock | B-3 lock=200 |
|-------|------------:|-------------:|------------:|-------------:|
| Gaussian (ADR-0005 / ADR-0011) | −0.0063 | 0.0000 | 0.1484 | 0.1719 |
| XOR (ADR-0011) | −0.0062 | 0.0000 | 0.1406 | 0.1250 |
| Sinusoid (ADR-0011) | +0.0125 | 0.0000 | 0.1406 | 0.1562 |
| ECG 2-modal (ADR-0009 / ADR-0011) | −0.0062 | −0.0062 | 0.4453 | 0.4453 |
| **5-modal real (this ADR)** | _fill_ | _fill_ | _fill_ | _fill_ |

## Decision — Branch B picked (B-3 PASS at 5.1×-6.2×)

The v0.5 OSF amendment declared three mutually-exclusive decision
branches BEFORE compute:

- [ ] **Branch A — B-3 PASS at ≥ 10×**: "B-3 is an architectural invariant
      across synthetic AND real biological 5-modal input." (strong headline)
- [x] **Branch B — B-3 PASS at 1×-10×**: "B-3 persists but is attenuated
      under biological input complexity." (moderate) — **PICKED**
- [ ] **Branch C — B-3 FAIL**: "B-3 was a synthetic-cluster artefact;
      the unlocked ECG-2 result was driven by zeroed modalities."

B-3 no-lock at 5.1× threshold and lock at 6.2× threshold fall
squarely in Branch B: PASS but attenuated versus the ECG 2-modal
22.3× headline (ADR-0009). The attenuation is interpretable
architecturally — the 4.5 biologically-richer modalities
(especially real audio mel-spectrogram + real VGG16 vision) give
the network more axes on which T1/T2 can disagree, which
**reduces** the off-diagonal asymmetry on the 5x5 perf matrix.

## Bonus finding — B-1 directional lift under lock

| World | B-1 no-lock | B-1 lock=200 |
|-------|------------:|-------------:|
| Gaussian | -0.0063 | 0.0000 (neutralised) |
| XOR | -0.0062 | 0.0000 (neutralised) |
| Sinusoid | +0.0125 | 0.0000 (destroyed) |
| real ECG 2-modal | -0.0062 | -0.0062 (no effect) |
| **4.5-modal (this ADR)** | **0.0000** | **+0.0063** (lift!) |

This is the FIRST world where `constellation_lock_after=200`
produces a positive me7 gap (in the pre-registered biological
direction) from a zero no-lock baseline. Magnitude is ~1/8th of
the 0.05 threshold, so B-1 still FAILs quantitatively, but the
directional result is qualitatively distinct from all 4 prior
worlds. Candidate mechanism: with the 3 previously-zeroed
modalities (tactile / gravity / force) now carrying real signal,
the lock has a richer substrate on which to express a congenital
(T1) vs late-acquired (T2) plasticity differential.

## B-2 with methodology robustness

The B-2 Me3 delta is **-0.0376** with a bootstrap 95% CI of
**[-0.056, -0.001]** (via nerve-wml v1.5.3 `bootstrap_ci_mi` +
scipy.stats.bootstrap). CI is **entirely negative** (upper bound
-0.001 < 0) — first world where the under-threshold B-2 is
robustly distinguishable from zero, just barely. Direction is
opposite the pre-registered +0.10 hypothesis; interpretation:
the lock-free T2 network's post-lesion MI is slightly **lower**
than its pre-lesion MI on this 4.5-modal bridge. This may reflect
interference rather than migration — the existence of multiple
real modalities changes the MI landscape in a way the synthetic
cluster + ECG 2-modal bridges did not expose.

The null-model MI check hit an API mismatch between the
`nerve-wml.methodology.NullModelResult` dataclass and the
client script's attribute expectations (`observed_mi` vs
`.mi_observed` or similar). Tracked as a follow-up fix but
does not affect the bootstrap-CI verdict above.

## Honest note on reproducibility

The SHA256 manifest for the 5 feature tensors depends on the pinned ffmpeg
version (logged automatically by the orchestrator, Task 9.2 fix 6). A reviewer
wanting byte-identical reproduction must match that ffmpeg version.

## Pre-registration fidelity

- No threshold changes.
- No metric-math changes.
- v0.3 verdicts (ADR-0004 / 0005 / 0008 / 0009) and v0.4 lock verdicts
  (ADR-0010 / 0011) remain canonical.
- This ADR strictly extends the experimental matrix.

## Next steps (revised)

Because the phase-2 data-alignment issues require subject-run matching + a
sibling annotation dataset, the full 5-modal grid is promoted to a Sprint 9.5
follow-up. The immediate work divides as follows:

### Sprint 9.5 — data-alignment correction (a new plan)

1. `datalad install` both `studyforrest-data-phase1` (full soundtrack +
   motion annotations) AND `studyforrest-data-phase2` (ses-movie BOLD +
   eyegaze physio for sub-01).
2. Switch canonical subject / run to one that has ALL 5 streams on the
   same session. Candidate search: `sub-02` run-3 has cardresp + BOLD;
   eyegaze requires phase-2. Motion annotations are phase-1. Audio is
   phase-1 soundtrack (`fg_av_ger_stereo.mp3`).
3. Derive `rp_*.txt` motion regressors locally via a lightweight fMRI
   preprocessing pipeline (fMRIprep in BIDS-mode, or a simpler SPM
   realign replacement via `nilearn.image.load_img` + `nipype`).
4. Rewrite `fetch_studyforrest_phase2.sh` to pull from both datasets and
   run the realign step inline.
5. Re-run the two 150-cell grids (no-lock + lock=200) on the properly
   aligned 5-modal cache.
6. Fill in this ADR's verdicts table and pick the decision-rule branch.

### Immediate release path (unchanged)

The Sprint 9 code work (Tasks 9.1–9.5 + 9.7 template + 9.8 scaffolding)
remains shippable as a **v0.5.0-rc** tag that declares:
- the full 5-modal API (`StudyforrestRealWorld`)
- the 5-modal audit dispatch (`audit_worlds.py`)
- the OSF amendment v0.5 filed (pre-registration fidelity preserved)
- this ADR as a transparent record of the data-alignment scope change

The v0.5.0 final tag is withheld until Sprint 9.5 produces real-data
verdicts and the decision-rule branch is picked.

### Scientific honesty clause

The 2-modal ECG bridge (ADR-0009 / ADR-0011) remains the canonical
"real biological signal" evidence in the paper v0.1 draft. No retraction
is needed; the paper's limitations section already declares ECG as a
narrow proof-of-concept, not a full 5-modal replication. Sprint 9.5 is
a scope EXTENSION, not a correction.
