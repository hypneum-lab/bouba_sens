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

## Expected grid provenance (to be filled in after Task 9.6)

| Item | Expected value | Actual |
|------|----------------|--------|
| Host | Studio | _fill_ |
| Worktree | `~/Projets/bouba_sens_b1` | _fill_ |
| Commit | `<sha of feat/sprint-9-5modal HEAD>` | _fill_ |
| nerve-wml | `v1.4.0` | _fill_ |
| Config | `STEPS_TRAIN=200 STEPS_LESION=100 METRICS="Me1,Me2,Me3"` | _fill_ |
| World | `StudyforrestRealWorld(data_dir=data/studyforrest_5modal_sub01_run1)` | _fill_ |
| Grids | no-lock + lock=200 | _fill_ |
| Wall time | ~40 min parallel | _fill_ |

## Verdicts (template — fill in from Task 9.6 aggregation output)

| Invariant | 5-modal no-lock | 5-modal lock=200 |
|-----------|-----------------|------------------|
| B-1 Me7 (threshold > 0.05) | _fill_ | _fill_ |
| B-2 Me3 delta (threshold > 0.10) | _fill_ | _fill_ |
| B-3 Me6 max-abs (threshold > 0.02) | _fill_ | _fill_ |

### Comparison against prior ADRs

| World | B-1 no-lock | B-1 lock=200 | B-3 no-lock | B-3 lock=200 |
|-------|------------:|-------------:|------------:|-------------:|
| Gaussian (ADR-0005 / ADR-0011) | −0.0063 | 0.0000 | 0.1484 | 0.1719 |
| XOR (ADR-0011) | −0.0062 | 0.0000 | 0.1406 | 0.1250 |
| Sinusoid (ADR-0011) | +0.0125 | 0.0000 | 0.1406 | 0.1562 |
| ECG 2-modal (ADR-0009 / ADR-0011) | −0.0062 | −0.0062 | 0.4453 | 0.4453 |
| **5-modal real (this ADR)** | _fill_ | _fill_ | _fill_ | _fill_ |

## Decision (pick ONE branch from the OSF amendment v0.5)

The v0.5 OSF amendment (filed before grid runs, see
`docs/osf/amendment-v0.5-studyforrest-5modal.md`) declared three
mutually-exclusive decision branches BEFORE compute. Pick exactly one after
filling the verdicts table:

- [ ] **Branch A — B-3 PASS at ≥ 10×**: "B-3 is an architectural invariant
      across synthetic AND real biological 5-modal input." (strong headline)
- [ ] **Branch B — B-3 PASS at 1×-10×**: "B-3 persists but is attenuated
      under biological input complexity." (moderate)
- [ ] **Branch C — B-3 FAIL**: "B-3 was a synthetic-cluster artefact; the
      unlocked ECG-2 result was driven by zeroed modalities." (retraction
      of v0.1 headline)

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
