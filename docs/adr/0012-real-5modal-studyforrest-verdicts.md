# ADR-0012 — Real 5-modality Studyforrest verdicts (TEMPLATE)

**Status:** Pending — compute blocked on Studio datalad install (see Next steps)
**Date:** 2026-04-21
**Sprint:** 9

## Context

Sprint 9 code work (Tasks 9.1–9.5 + 9.7 template + 9.8 release scaffolding) lands the full wiring required to run the 150-cell grid on a real 5-modality Studyforrest bridge. This ADR will be finalised **after** the compute runs complete on Studio.

Blocker recorded 2026-04-21: Studio (MacStudio) has neither Homebrew, datalad, nor git-annex installed. The fetch script (`scripts/fetch_studyforrest_phase2.sh`) hard-requires datalad + git-annex to pull the BIDS subset. Installing these requires sudo + Homebrew setup (~15 min manual step on Studio). The user will run the setup in a follow-up session.

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

## Next steps

1. Install Homebrew + datalad + git-annex on Studio (manual, requires sudo):

   ```bash
   ssh studio
   # one-shot Homebrew install
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   brew install datalad git-annex ffmpeg
   ```

2. Run the fetch + extract pipeline from the Sprint 9 plan Task 9.6.
3. Launch the two 150-cell grids (no-lock + lock=200) per Task 9.6 Step 3.
4. Fill in this ADR's verdicts table and pick the decision-rule branch.
5. Update `docs/paper/paper-v0.1-draft.md` §5.5 with the matrix and the
   picked branch's headline framing.
6. Bump `src/bouba_sens/_version.py` + `pyproject.toml` to `0.5.0`, write
   the `CHANGELOG.md` v0.5.0 entry, tag + push.
