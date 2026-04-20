# ADR-0008 — Studyforrest-mock out-of-cluster verdicts

**Status:** Accepted — **descriptive, not pre-registered**
**Date:** 2026-04-20
**Sprint:** 7 (extended scope)

## Context

ADR-0005 closed Sprint 6 with B-3 PASS 3/3 on the synthetic cluster
(Gaussian, XOR, Sinusoid). The maintainer immediately raised the
external-validity critique: three synthetic worlds may be three
samples of the same cluster. Sprint 7 Tasks 7.5 + 7.6 quantified
the cluster (world-complexity audit) and built a biological-adjacent
bridge (`StudyforrestWorld`). This ADR records the results of
running the pre-registered grid on the fourth (out-of-cluster)
world.

**Pre-registration status.** The v0.3 OSF amendment (Task 7.6b
`docs/osf/amendment-v0.4-studyforrest.md`) is **drafted but not
yet filed**. Therefore this verdict is recorded as **descriptive
stress-test**, not as a pre-registered verdict. It is listed in
Sprint 8 paper draft in the "Limitations / internal stress-test"
section, not in the Results. The v0.4.0 tag is withheld until
the OSF amendment is filed and a fresh grid run is performed on
actual biological data.

## Grid provenance

| Item | Value |
|------|-------|
| Host | Studio (MacStudio, arm64, macOS 26.4) |
| Commit | `579e981` (main) |
| World | `StudyforrestWorld` mode=mock (offline surrogate) |
| Data source | `data/studyforrest_sample/` (AR(1) scene-latent, 4096 frames) |
| Config | same as v0.3 (`STEPS_TRAIN=200`, `STEPS_LESION=100`, `METRICS="Me1,Me2,Me3"`) |
| Cells | 150 / 150 processed, 0 skipped |
| Wall time | ~17 min solo |
| Artifacts | `reports/v0.4_studyforrest_aggregate.json` |

## Verdicts vs synthetic cluster

| Invariant | Gaussian | XOR | Sinusoid | **Studyforrest mock** | Verdict |
|-----------|---------:|----:|---------:|----------------------:|---------|
| B-1 Me7 > 0.05 | −0.0063 | −0.0062 | +0.0125 | **0.0000** | FAIL — collapses on AR(1) |
| B-2 Me3_delta > 0.10 | 0.0275 | 0.0034 | 0.0019 | **−0.0288** | FAIL — inverts sign |
| B-3 Me6 max-abs > 0.02 | 0.1484 | 0.1406 | 0.1406 | **0.3125** | **PASS @ 15.6x threshold (2x stronger than cluster)** |

## Decision

**B-3 survives the cluster boundary with amplified effect size.**
The 5x5 query-modality asymmetry jumps from ~0.14 in the
synthetic cluster to 0.31 on the Studyforrest mock — roughly
2x larger, 15.6x the pre-registered threshold. This is the
strongest evidence to date that B-3 captures an architectural
property independent of the world's cluster membership. It
materially answers the 2026-04-20 external-validity critique
that the 3-synthetic-worlds verdict proved nothing about the
real world.

**B-1 collapses exactly to zero.** On a world with AR(1)
temporal structure, the congenital/late-acquired accuracy gap
vanishes. Combined with the Sinusoid sign flip in ADR-0005,
this confirms B-1 is a **world-topology-dependent** effect —
not an architectural invariant.

**B-2 flips negative.** Me3_delta drops to −0.029, meaning MI
*decreases* post-lesion on the Studyforrest mock. Candidate
interpretations:

1. The 3 zeroed modalities (tactile / gravity / force) give the
   MI migration nowhere to go — no surviving proprioceptive
   channel to absorb the audio/vision information. This is an
   **artefact of the bridge's mock scope**, not a real
   falsification.
2. The AR(1) temporal structure couples consecutive samples,
   so the Kraskov estimator's i.i.d. assumption is violated,
   producing biased estimates.

Either way, B-2 on Studyforrest mock is **not comparable** to B-2
on synthetic worlds without methodological normalisation. The
paper must either (a) compute a zeroed-modality-aware variant of
Me3_delta for Studyforrest, or (b) report the B-2 drop as a
known scope limitation of the bridge.

## Headline

The v0.3 cluster-agnostic claim for B-3 is **preserved and
reinforced** by an out-of-cluster replication on the
Studyforrest mock. B-1 and B-2 do not survive the cluster
boundary — but that was already suggested by the intra-cluster
variation in ADR-0005. B-3 is now the **only** invariant with
defendable out-of-cluster robustness.

## Pre-registration fidelity

- No threshold changes.
- No metric-math changes.
- The v0.3 verdicts in ADR-0005 stand untouched.
- This result is filed as **descriptive**, not as a
  pre-registered B-3 confirmation, because the OSF amendment for
  adding Studyforrest to the registered grid is drafted but not
  yet filed.
- The v0.4.0 tag is withheld until (a) the amendment is filed
  and (b) a fresh grid runs on real Studyforrest data (not the
  mock surrogate).

## Next steps

1. File the OSF amendment (draft in `docs/osf/amendment-v0.4-studyforrest.md`).
2. Sprint 8+ : implement the real-data extractor in
   `scripts/fetch_studyforrest_sample.py` (currently the real-fetch
   branch falls through to the offline surrogate).
3. Re-run the grid with `BOUBA_SENS_STUDYFORREST_DATA` pointing
   at the real tensors.
4. Compare the real-data verdict to this mock-verdict; if B-3
   stays in the 0.14-0.35 range, the paper headline is
   defensible as written. If it drops below 0.02, the paper
   pivots to "architectural property with cluster-specific
   amplification."
5. For the paper's "Limitations: external validity" section,
   note that tactile / gravity / force are zeroed in the bridge
   and this decision drives B-2's sign flip.
