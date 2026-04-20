# OSF pre-registration amendment — bouba_sens v0.4

**Status:** Draft, ready to file
**Parent pre-registration:** `dream-of-kiki` OSF DOI `10.17605/OSF.IO/Q6JYN` (locked 2026-04-19)
**Derived spec:** `bouba_sens` repo, branch `main`, tag `v0.3.0`
**Amendment tag (on file):** `bouba_sens/v0.4-studyforrest-extension`
**Amendment date:** 2026-04-20

> **Filing note.** This file is the machine-readable draft. The
> human-readable registration form pasted into the OSF amendment
> flow is `docs/osf/amendment-v0.4-studyforrest.txt` (plain text
> export, same content).

---

## 1. What is being amended

The v0.3 pre-registered grid runs on three synthetic worlds
(GaussianWorld, XORWorld, SinusoidWorld) with the 5-modality
factorisation + 4-class label defined in spec sections 3.1 and
3.2. ADR-0005 and ADR-0007 (committed to the `hypneum-lab/bouba_sens`
repo) established that these three worlds form a tight cluster in
world-complexity space: they share `intrinsic_dim_pca` (~30 per
perceptive modality), near-zero `mi_pairwise` (~0.02), near-zero
`temporal_autocorr`, and equivalent `support_compactness`.

**Amendment adds a fourth world — StudyforrestWorld — to the
pre-registered grid.** StudyforrestWorld produces 2 real modalities
(audio spectrogram + visual CNN features from the public
Studyforrest dataset, Creative Commons, Hanke et al. 2014 / 2016)
with the other 3 modalities (tactile, gravity, force) zeroed.
A `mock` mode generates AR(1) scene-latent surrogate data with
the same statistical signature when network access is unavailable.

No existing pre-registered threshold (B-1: 0.05, B-2: 0.10, B-3:
0.02 per spec section 1.2) is touched. The 5 seeds x 5 modalities
x 2 timings x 3 SNR = 150-cell grid structure is preserved per
world. Sample sizes and `STEPS_TRAIN`/`STEPS_LESION` defaults
(200 / 100) are unchanged.

## 2. Motivation

ADR-0007 recorded the following out-of-cluster measurements for
the Studyforrest mock surrogate (which uses the same API as the
synthetic worlds):

| Metric | gaussian | xor | sinusoid | studyforrest_mock | Gap |
|--------|---------:|----:|---------:|------------------:|----:|
| `intrinsic_dim_pca` audio | 30 | 30 | 29 | 4 | 7x |
| `mi_pairwise` | 0.015 | 0.023 | 0.018 | 0.392 | 17x |
| `temporal_autocorr` | 0.00 | 0.07 | -0.02 | 0.82 | qualitatively new |

Three of the six audit metrics place Studyforrest clearly outside
the synthetic cluster. Running the pre-registered grid on this
fourth world is the minimum external-validity stress test
warranted by the B-3 PASS 3/3 finding. Not running it would be
opportunistic: it would let the paper claim world-agnosticity
that is only verified within the synthetic cluster.

## 3. Protocol

The amendment is a **pure extension**: the v0.3 verdicts
(Gaussian + XOR + Sinusoid) are retained verbatim as historical
record in ADR-0005 and in `reports/v0.3_*_aggregate.json`. The
v0.4 grid adds:

1. `StudyforrestWorld` pre-fetched sample extraction
   (`scripts/fetch_studyforrest_sample.py`) with a public mirror
   checksum manifest so reproduction is byte-identical.
2. `WORLD=studyforrest` variant of the existing 150-cell grid,
   run on the same Studio hardware under the same CLI.
3. Aggregate JSON written to `reports/v0.4_studyforrest_aggregate.json`
   with the same schema as v0.3, feeding the same
   `scripts/aggregate_grid.py` pipeline.
4. ADR-0008 records the four-world verdict table and the
   architecturally-load-bearing finding: does B-3 survive the
   cluster boundary?

### Decision rule for paper (Sprint 8)

| B-3 status on studyforrest | Paper claim |
|----------------------------|-------------|
| PASS at ≥ 5x threshold | "Out-of-cluster replicated; B-3 is world-agnostic." |
| PASS at 1x-5x threshold | "B-3 survives the cluster boundary with diminished effect size; architectural property holds." |
| FAIL | "B-3 passes on the synthetic cluster only; the claim is synthetic-cluster-agnostic, not world-agnostic. Section 'External validity' becomes load-bearing." |

## 4. What is NOT being amended

- Thresholds 0.05 / 0.10 / 0.02 — frozen per spec 1.2.
- Metric implementations (`me1_accuracy`, `me2_recovery_auc`,
  `me3_delta`, `me6_asymmetry`, `me6_max_abs_off_diag`,
  `me7_congenital_gap`, `me9_bootstrap`) — frozen since Sprint 3.
- The 5-modality factorisation + 4-class label structure.
- The `run_grid.sh` orchestrator semantics — only a WORLD knob.

## 5. Pre-registration fidelity checklist

- [x] No threshold change.
- [x] No metric-math change.
- [x] Existing verdicts retained in an immutable ADR (ADR-0005).
- [x] New verdicts published under a new ADR and a new version tag
      (v0.4.x) so `git blame` trivially attributes each claim.
- [x] Dataset checksum manifest committed before grid runs so the
      data used for the verdict is byte-fixed.
- [x] Mock + real variants produce the same schema; the paper
      reports both so readers can see the stats-surrogate gap.

## 6. Timeline

| Step | Owner | ETA |
|------|-------|-----|
| File amendment on OSF (this doc → OSF web form) | maintainer | 2026-04-21 |
| Extract Studyforrest tensors via `fetch_studyforrest_sample.py` | Studio | 2026-04-21 |
| Run `WORLD=studyforrest` grid on Studio | Studio | 2026-04-21 |
| ADR-0008 + tag `v0.4.0` | maintainer | 2026-04-21 |
| Sprint 8 paper draft | maintainer | 2026-04-22 |

The mock-mode grid may run before the OSF filing since it does
not consume pre-registered data; its verdict is **descriptive
only** and goes into the paper's "Limitations / internal
stress-test" section, not the Results. Real-data runs strictly
post-filing.
