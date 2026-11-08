# bouba_sens

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19701353.svg)](https://doi.org/10.5281/zenodo.19701353)

> A benchmark for cross-modal plasticity in artificial neural systems.
> Hypneum Lab — 2026.

`bouba_sens` studies how a 5-modality agent (audio, vision, tactile, gravity,
force) reorganises itself when one sensory channel is lost or degraded —
inspired by the cross-modal cortical recruitment observed in congenital and
late blindness (Amedi 2007, Merabet 2010, Heimler & Amedi 2020).

## Three testable invariants

- **B-1 — Congenital gap:** lesion pre-training yields better adaptation than
  lesion post-convergence.
- **B-2 — MI migration:** mutual information between surviving-modality
  neuroletters and the target label rises post-lesion.
- **B-3 — Perceptive/proprioceptive asymmetry:** losing vision/audio/tactile
  produces a quantitatively different plastic response than losing
  gravity/force.

## 2026-05-11 update — Q3+ FINAL Retract, TMLR submission BLOCKED

**§5.5 B-1 dose-response signature requires reformulation.** The
N9 Q3+ 10-seed re-run with the correct test (quadratic regression
over LOCK_AFTER) returned **sign p = 0.62**, paired t-test
**p = 0.28** — verdict **FINAL Retract** per the Q3 pre-registered
decision criteria. The earlier Q3 INTERIM verdict had used a
Jonckheere-Terpstra test which the critic v2 review correctly
flagged as the wrong test for a non-monotone (peaked) hypothesis ;
fixing that and adding 5 more seeds collapsed the qualitative
+0.0125 Sprint 10 peak to noise.

**TMLR benchmarks-track submission is BLOCKED.** The §5.5
reformulation (Version C, full retract —
`docs/paper/§5.5-reformulation-FINAL.md`) has landed in the
manuscript; the remaining gates are the critic review and the
N≥30 confirmatory decision. Decision-of-record : ADR-0019.

**N12 targeted subgroup replication (DONE — LOST 0/2).** Pooled-v2
analysis surfaced two candidate subgroups — **tactile-floor** and
**force-plus10** — at sub-threshold but consistent positive shifts.
The pre-registered N12 replication on N=10 new seeds failed at
Bonferroni α=0.025 for both (ADR-0019), which selected the full
retract (Version C).

**Multi-seed-first-class lesson promoted portfolio-wide.** The Q3
collapse from a 5-seed qualitative claim to a 10-seed null is the
canonical evidence for the methodology rule that single-seed (or
under-seeded) claims are fragile. Adopted across `nerve-wml`,
`dream-of-kiki`, and this repo as a hard pre-publication gate.

No new positive finding is announced ; this update is an honest
scope statement on what tonight's verdicts removed and what the
running sweep may add back.

| Axis | Status |
|---|---|
| **B-3** Me6 perceptive/proprioceptive asymmetry | **Final (PI 2026-10-27): on the Gaussian world, with a network that learns the task (straight-through emission), no lesion effect is detected — neither a lesion × query interaction (p = 0.49, tagged preregistration `prereg-b3-lesion-interaction-20261027`) nor a lesion-driven asymmetry; only the query-modality effect, built by the world, remains. The mock world stays inconclusive (network below the learning criterion) and is not retried.** *Earlier, default (non-learning) regime, ADR-0024:* **v2 (ADR-0023, PI 2026-10-26): asymmetry built by the worlds (unequally informative modalities), no lesion effect** — linear probe, positive control met (vision margin 0.485 Gaussian / 0.282 mock); preregistered criterion met (median 0.414 / 0.332, p = 5·10⁻⁵, blocks 1 / 8) but H0 was ill-posed; residual after removing the query-modality effect at the null level (Gaussian p ≈ 0.55). *v1 (historical):* **Gaussian: calibrated, compatible with the paired null** (was *uncalibrated*, see F1 and Erratum; ADR-0022, PI decision 2026-10-25, Gaussian grid: observed median 0.156 vs null mean 0.141 / P99 0.156, p = 0.17; iid reference almost identical). Mean query accuracy 0.254 = 4-class chance → the zero-masking probe is a design defect to fix before any new B-3 reading. Studyforrest mock: 0.305, rejected by per-item / block-8 nulls but compatible under block flips of 16 / 32 (p = 0.06 / 0.31), accuracy 0.240 = chance → not conclusive, block length left to the PI. Real ECG (0.445): not recomputed (no data cache on electron-server), still uncalibrated. The former "PASS on magnitude (5–6× threshold)" was never compared with a null; the "architectural property" reading stays withdrawn (entropy proxy, paper §3.3/§4.4) *Me6 at n ∈ {64, 256, 1024, 4096} on the existing cells (ADR-0026, tag `prereg-me6-large-n-20261031`, #63, 2026-10-31):* learning gate not passed (argmax-bridge accuracy 0.24–0.27), so no row is a B-3 reading; Gaussian rows in excess over the paired null at n ≥ 256 in both arms, mock rows compatible; the frozen 0.02 rule fires under the paired null in 0.748 / 0.981 of draws on the mock at n = 4096 and decides nothing. Under the lab vocabulary (PM §6) the B-3 magnitude is **non concluant** until the pre-registered network-level null (PR #66) is frozen and run. |
| **B-1** Me7 congenital gap (§5.5) | **FINAL Retract** (Q3+ 10-seed, ADR-0019) — reformulation **landed in manuscript** |
| **B-2** Me3 MI migration | **Null** on the Gaussian world and the CC-film bridge (ADR-0019: full-D probe, N≳f(D)); **not conclusive** on the Studyforrest bridge (single-class probe, erratum 12, issue #77) |
| **TMLR submission** | **BLOCKED** — §5.5 retraction landed; pending critic-review gate + N≥30 decision |
| **N12 subgroup replication** | **DONE — LOST 0/2** at Bonferroni α=0.025 (ADR-0019) |

Cumulative on main HEAD `3fbeee7` : **156 commits**, **v0.5.9**, 19
ADRs (0004–0019), 185+ tests, pooled-v2 analysis + N12 pre-reg
locked. Honest scope ; no overselling.

---

## Status

**v0.5.9 — Paper v0.1 draft (TMLR submission blocked; see
2026-05-11 update above).** Sprints 0–17 closed, 28 ADR files
(numbers 0001–0022, seven numbers shared — index in
`docs/adr/README.md`), 372 unit+smoke tests. 9 grids (150 cells each) on the
4.5-modal real Studyforrest Phase-2 bridge + 5 worlds synthetic.
§5.5 B-1 retraction now landed in the manuscript (ADR-0019).

- Paper draft : `docs/paper/paper-v0.1-draft.md`
- Pre-registration : **local, git-dated** — design spec `docs/superpowers/specs/2026-04-20-bouba-sens-design.md` (commit `87f723f`, 2026-04-20) ; amendments v0.4–v0.6 under `docs/osf/` (drafted as OSF amendments, **never filed on OSF**). No bouba_sens hypothesis is registered on OSF : `10.17605/OSF.IO/Q6JYN` covers dream-of-kiki H1–H4 only (issue #20, Erratum below).
- Target venue : TMLR benchmarks track (primary), NeurIPS D&B (fallback)

## Headline findings (v0.5.5)

Thresholds locked in git before the campaign (design spec
2026-04-20, commit `87f723f` ; not registered on OSF) : B-1 ≥ 0.05,
B-2 ≥ 0.10, B-3 ≥ 0.02 ; no threshold changes across all ADRs.

| Invariant | Threshold | Best grid-median | Seed-stability | Verdict |
|-----------|----------:|-----------------:|:---------------|---------|
| **B-3** Me6 perceptive/proprio asymmetry | 0.02 | **0.109–0.156** (5.5–7.8×) | 5/5 pass the fixed rule — as do 100 % of null grids at n = 64 | **Compatible with the paired null** (ADR-0022, Gaussian: p = 0.17; null P99 = 0.156). The fixed rule cannot separate signal from noise at n = 64; mean query accuracy 0.254 = chance → probe defect (zero-masking) to fix first. Magnitude tracks per-modality entropy, not the partition (§4.4) |
| B-1 Me7 congenital gap | 0.05 | **+0.0125** (25 % of thr) | 3+/5 positive | **Qualitative** — only `LOCK=100 + HARD` on the real bridge (§5.5 Amedi dose-response) ; **N8-Q3 5-seed interim verdict `Retract` 2026-05-11, see CHANGELOG — TMLR blocked, Q3+ 10-seed pending** |
| B-2 Me3 MI migration | 0.10 | mean +0.031 ±0.047 bits (full-probe MINE) | 0/5 cross threshold | **Null** on synthetic worlds + the CC-film bridge (ADR-0019: Kraskov + adaptive-binning + **full 896-D MINE** bracket zero); **not conclusive** on the Studyforrest bridge (erratum 12) |

**F1 — B-3 is an uncalibrated magnitude (requalified
2026-10-25).** The Me6 asymmetry magnitude exceeds the 0.02
threshold by 5–6× across every grid, every world, every seed, and
is insensitive to every plasticity control tested (LOCK_AFTER, transducer gating mode, Gumbel tau, codebook
freeze, HARD → GUMBEL phase transition). The earlier reading of
B-3 as an *architectural* perceptive/proprioceptive property is
**withdrawn** (paper §3.3, §4.4–4.5; ADR-0014, ADR-0015): a
null-model partition control puts the pre-registered partition at
the 44–72th percentile of random 3+2 alternatives, and the
magnitude behaves as a per-modality entropy proxy. The B-3
magnitude result is anchored as the empirical foundation of
[`dream-of-kiki/docs/specs/2026-05-20-biophysical-stratification.md`](https://github.com/hypneum-lab/dream-of-kiki/blob/main/docs/specs/2026-05-20-biophysical-stratification.md)
§4 (embodied sensorimotor grounding stratum). The B-1 Retract
(ADR-0019) is honoured — Stratum 4 explicitly anchors on B-3,
never B-1.

**F1 update — paired calibration (ADR-0022, accepted by the PI on
2026-10-25).** On a fresh Gaussian grid with saved per-item outcomes,
the B-3 median (0.156) is not distinguishable from a paired sign-flip
null (mean 0.141, P99 0.156, p = 0.17), and the mean single-modality
query accuracy is 0.254, i.e. 4-class chance: the zero-masking probe
carries no decodable per-modality information, a design defect to fix
before any new B-3 reading. On the Studyforrest mock (0.305, accuracy
0.240) the verdict depends on the block length of the null and is not
conclusive; the real-ECG grid was not recomputed. Any downstream anchor
on B-3 (including the Stratum 4 anchor above) should be read in that
light.

**F1 update 4 (2026-11-08).** Me6 was rerun at n ∈ {64, 256, 1024,
4096} on the existing cells (ADR-0026, tag
`prereg-me6-large-n-20261031`, #63; 600/600 cells, 2026-10-31). The
learning gate is not passed for the argmax-bridge emission (accuracy
0.24–0.27), so no row is a B-3 reading; Gaussian rows are in excess
over the paired null at n ≥ 256 (six of eight rows, five below the
Bonferroni level) and mock rows are compatible. The frozen 0.02 rule
fires under the paired null in 100 % of draws at n ≤ 1024 and in
0.748 (cell) / 0.981 (legacy) on the mock at n = 4096, so it decides
nothing; under the lab vocabulary (PM §6) the B-3 magnitude is **non
concluant** until the pre-registered network-level null (PR #66) is
frozen and run. Details:
`docs/milestones/2026-10-31-me6-large-n-results.md`.

**F1 update 3 — final B-3 wording (PI decision 2026-10-27).**
On the Gaussian world, with a network that learns the task
(straight-through emission, Phase-1 loss 0.81–1.13, Me1 0.467), no
lesion effect is detected: neither a lesion × query interaction
(exact seed-level test, p = 0.49, tagged preregistration
`prereg-b3-lesion-interaction-20261027`) nor a lesion-driven
asymmetry; only the query-modality effect, built by the world,
remains. The mock world stays inconclusive (its pilot network is below
the learning criterion) and is not retried.

**F1 update 2 — probe v2 (ADR-0023, accepted by the PI on
2026-10-26).** The zero-masking probe is replaced by a linear read-out
of each modality's frozen representation (positive control met on both
worlds). Under the preregistered paired block null
(`prereg-b3-probe-v2-20261025`) B-3 v2 rejects symmetry on the Gaussian
and mock worlds (p = 5·10⁻⁵), but the PI reading is: an asymmetry built
into the worlds (unequally informative modalities), no lesion effect.
The preregistered criterion is met but H0 was ill-posed; the residual
after removing the query-modality effect is at the null level. The
networks' code channel is collapsed (one argmax code per modality),
so the probe reads encoder states, not what the nerve transmits.

*Requalification (2026-10-25, PI decision ; ADR-0021 ; Refs #22).*
The 0.02 threshold was never compared with a null. Under an
independent binomial null at the production probe size (n = 64),
the median-of-30 Me6 is 0.13–0.16 (P99 0.15–0.19) : an independent
binomial null invalidates the reading of the bare exceedance of
0.02. The synthetic-world medians (0.141–0.148) lie inside that
range ; mock (0.3047) and ECG (0.4453) exceed it, but the iid
reference is not the null of their contiguous probes. No
compatibility or rejection verdict on B-3 is established before a
paired calibration of the median (same probe for the five queries,
shared seeds and worlds). The values stand ; the interpretation is
withdrawn, and the Stratum 4 anchoring above therefore rests on an
uncalibrated statistic until that calibration is done.

**F2 — B-1 does not survive multi-seed validation (ADR-0019
FINAL Retract).** The single-seed +0.0125 peak (Sprint 10,
ADR-0013) inverts to −0.0250 on the canonical seed; 10-seed
sign-test p=0.62 (chance); N12 pre-registered subgroup
replication LOST 0/2 at Bonferroni α=0.025. B-1 is below the
noise floor. The Sprint 10 single-seed result is a methodological
lesson on multi-seed validation discipline, not a reproducible
finding (see §5.5 retraction and ADR-0019).

**F3 — B-2 is below threshold, Null confirmed/renforced (ADR-0019).**
The earlier "bimodal positive B-2 at tau=0.30" claim (ADR-0016) was
retracted after per-seed re-aggregation (ADR-0017 §14a). The probe-shape
follow-up declared for v0.2 is now **closed (ADR-0019)** :

- *Probe shape, not estimator agreement.* The Me3 sonde was made
  configurable; its default is now the **full representation `(B, 896)`**.
  The old 1-D mean-pool **pinned MINE to exactly 0.000 on every seed** —
  a probe-shape artefact (the Donsker-Varadhan critic floors on a scalar
  probe), *not* an independent two-estimator agreement. `mi_migration`
  now handles the multi-D probe.
- *Sample-size criterion re-derived.* The Me3 probe has ambient
  dimension 896 but **effective dimension D ≈ 12** (participation ratio
  11.7 / two-NN 12.4). The fixed "probe batch ≥128" heuristic is replaced
  by an estimator-aware bound **N ≳ f(D)** (MI sample complexity scales
  with intrinsic, not ambient, dimension).
- *Verdict.* B-2 remains **Null** — **no estimator crosses +0.10**.
  Feeding MINE the full 896-D probe (the v0.2 "richer probe rescues B-2"
  hypothesis) surfaces no positive signal; adaptive binning collapses in
  896-D exactly as the `N ≫ (#bins)^D` argument predicts. The null holds
  on the gaussian reference and on the CC-film bridge; on the
  Studyforrest bridge the measurement is not conclusive (erratum 12,
  issue #77), so robustness to which half of the modalities is real is
  not established.

**F4 — Two real bridges with honestly-labelled channel provenance.**
B-2 Null was re-checked on two real-signal bridges with **opposite**
real/synthetic modality splits (complements, *not* a replacement of the
headline synthetic findings) :

- **Studyforrest (OpenNeuro ds000113).** gravity / tactile / labels =
  **REAL** ds000113 signal ; force = **REAL** physiology but mislabelled
  and time-compressed in the cache used for the reported values
  (erratum 11, issue #74) ; audio / vision = annotation
  **PROXY** (the Forrest Gump soundtrack and film are copyrighted and
  absent from OpenNeuro — labelled as proxies, not "real movie vision").
- **CC-film (Big Buck Bunny, CC-BY 3.0).** audio (log-mel) / vision
  (frames) / labels (shot-cuts) = **REAL** perceptual signal ; tactile /
  gravity / force = **SYNTHETIC** (a film has no proprioception).

B-2 is Null on the CC-film bridge and the three synthetic worlds; on the
Studyforrest bridge it is **not conclusive** (erratum 12). The earlier
claim that the null is robust to which half of the modality set carries
the true signal is no longer supported.
The abstract is aligned to this per-channel provenance (no "real movie
vision" overclaim).

Full trail : `docs/adr/` (index in `docs/adr/README.md` ; ADR
numbers 0005, 0007, 0014, 0015, 0016, 0017 and 0019 are each shared
by two files, disambiguated as a/b). Every verdict, retraction, and
scope change is recorded inline with its grid commit.

## Erratum (phase 0 — truth, 2026-10-25)

Documentary corrections only ; no value, metric or threshold changed.

1. **"Pre-registered on OSF" withdrawn** (`CITATION.cff`, this README,
   `.zenodo.json`, `CLAUDE.md`, `AGENTS.md`, `docs/paper/`). Root
   cause : confusion with the dream-of-kiki registration
   `10.17605/OSF.IO/Q6JYN` (2026-04-19), which covers H1–H4 of that
   project only. B-1/B-2/B-3 are pre-registered **locally** (git-dated
   spec, 2026-04-20) ; the `docs/osf/` amendments were never filed.
   Issue #20.
2. **B-3 requalified "uncalibrated"** (F1 above, paper §4.4,
   `.zenodo.json`, ADR-0008/0009 notes). Root cause : a threshold
   fixed without a calibrated null. Values unchanged. Issue #22,
   ADR-0021.
3. **Paper §4.4 "Key claim, load-bearing" removed** ; `.zenodo.json`
   no longer carries the withdrawn "minimal configuration" claim.
   Root cause : documentation written ahead of the results. Issue #21.
4. **ADR numbering** : seven numbers shared by two files, now
   disambiguated a/b with a complete index ; ADR-0006 and tags
   `v0.4.0` / `v0.5.0` are not on `main` (PI action, see
   `docs/adr/README.md`). Issue #21.
5. **Reference `amedi2007`** : the BibTeX entry carried the metadata
   of Amedi et al. 2003 (`10.1038/nn1072`) ; it now points to Amedi et
   al. 2007, Nat Neurosci 10(6):687–689, `10.1038/nn1912`, which is
   the paper the repository cites throughout (Crossref-verified).
   Issue #25.

### Erratum (2026-10-26) — the default regime does not learn the task

6. **With the default settings the network does not learn the task**
   (ADR-0024; diagnosis PR #42). Phase-1 loss stays at ln 4 = 1.386
   for all 200 steps; full-network accuracy and Me1 are at 4-class
   chance (0.25–0.29) on every grid checked, historical included
   (Sprint 14c: 0.251). Root cause: the argmax emission leaves only a
   label-agnostic 1e-3 bridge gradient (about 1e-6), which Adam
   rescales into drift that collapses the code channel. **All B-1 /
   B-2 / B-3 results above, and the plasticity-mechanism results, were
   measured in this regime and are requalified accordingly**: they are
   not evidence for or against cross-modal plasticity in an agent that
   performs the task. No number is deleted. Issue #22.

### Erratum (2026-11-04) — wrong DOIs in the registered spec

7. The frozen design spec
   `docs/superpowers/specs/2026-04-20-bouba-sens-design.md` (lines
   508-509) gives two wrong DOIs; the spec is left unchanged. Correct
   identifiers (Crossref): Heimler & Amedi 2020, Neurosci. Biobehav.
   Rev. 116, 494-507, doi:10.1016/j.neubiorev.2020.06.034 (not
   10.1016/j.neuroscience.2019.12.048); Röder, Kekunnaya & Guerreiro
   2021, Neurosci. Biobehav. Rev. 120, 86-99,
   doi:10.1016/j.neubiorev.2020.10.030 (not 10.1073/pnas.2004321118).

### Erratum (2026-11-04) — Studyforrest licence and physiology

8. The OSF amendments describe the Studyforrest data inaccurately;
   the amendment files are left unchanged. Amendment v0.4
   (`docs/osf/amendment-v0.4-studyforrest.md`, line 35) says
   "Creative Commons": the data are released under the Open Data
   Commons Public Domain Dedication and License (PDDL), code under
   MIT (Sengupta et al. 2016, Sci. Data 3:160093,
   doi:10.1038/sdata.2016.93, p. 11). The physiological channel used
   for the force modality (amendment v0.5, table line 32, "ECG +
   respiration") is a pleth pulse from pulse oximetry plus a
   respiratory trace, not an ECG (same article, p. 4 and p. 7).
   Participant 2 has no physiological recordings in the localizer
   session (Table 1, p. 7). No value is changed.
9. The 4.5-modal bridge (ADR-0012) took its vision and scene-cut
   channels from `movie_localizer.mkv`, which is not film footage.
   The file (checked on 2026-11-06) lasts 12 s: 300 frames of
   1280×720 at 25 fps, no audio track. It shows a colored polar
   checkerboard redrawn at 5 Hz (60 patterns, colors redrawn, not a
   contrast reversal), filling only the movie-frame area. It is the
   inside-movie half of a localizer: in the scanner, the PsychoPy
   script alternated it every 12 s with `inv_movie_localizer.mkv`,
   4 cycles, 48 volumes, about 96 s (same article, p. 4 and p. 6;
   the dataset task description and the 48 sub-01 physio triggers
   agree). The vision channel (120 frames at 10 Hz) and the 7
   scene-cuts therefore describe this checkerboard; the cuts are
   pattern changes. Amendment v0.5 (table line 29, "film frames") is
   left unchanged; the paper section on the 4.5-modal bridge now
   describes the stimulus. The VGG16 numbers are kept.
10. The force channel of the 4.5-modal bridge does not hold the
   pulse-oximetry and respiration signal. `extract_force`
   (`src/bouba_sens/world/studyforrest_features.py`, lines 162-166)
   reads the first line of the header-less BIDS physio file as a
   header and falls back to column 0 (trigger) as "ECG" and column 1
   (pulse oximetry) as "respiration". In the regenerated `force.pt`
   the "ECG" channel is zero everywhere, "respiration" is the pulse,
   and the respiratory column is never read. The April tensors are
   lost; the code is unchanged since `d7a3645`, so they probably had
   the same defect (not verified). No value is changed. Issue #72.

### Erratum (2026-11-06) — phase-1 physiology layout

11. The force channel of the Studyforrest OpenNeuro bridge (paper
   §(d), ADR-0019) was read with the wrong layout. The phase-1
   `ses-forrestgump` physio files have four columns (trigger,
   respiratory trace, cardiac trace, oxygen saturation) at 100 Hz
   (Hanke et al. 2014, Sci. Data 1:140003, doi:10.1038/sdata.2014.3,
   p. 11); `scripts/fetch_studyforrest_sample.py` assumed three
   columns (trigger, cardiac, respiratory) at 500 Hz. The cache
   `data/studyforrest_real/force.pt` used for the §(d) values was
   built with this earlier code: its "cardiac" dims hold the
   respiration, its "resp" dims hold the cardiac trace, and its 9000
   frames span about 4500 s of physiology instead of 900 s. The script
   now reads the layout from a sidecar when one exists, else from the
   column count. The reported values have not been recomputed. This
   is separate from item 10. Issue #74.

### Erratum (2026-11-06) — degenerate B-2 probe on the Studyforrest bridge

12. "B-2 is Null on both real bridges" (README F3/F4, paper §(d)–(e),
   ADR-0019 E1.3c and Decision) is withdrawn for the Studyforrest
   bridge. Root cause: the B-2 probe is a contiguous slice of 512
   frames (51.2 s), and the label is the scene index quantised to 4,
   which stays constant over long scenes; in 4 of 5 seeds the probe
   holds a single class, so MI with the label is zero by construction
   and the control cannot fail (`reports/b2_phase1_recompute_20261106/`,
   issue #77). A class-stratified probe (ADR-0019 amendment of
   2026-11-06, exploratory) gives 128 frames per class, but the reader
   accuracy of the trained head is 0.25 on all 10 seeds (both caches),
   so the learning gate fails and B-2 is not measurable there
   (ADR-0019 E1.3c-ter). B-2 stays Null on the CC-film bridge (labels
   are per-frame quartiles over 600 frames, so every probe holds 62 to
   150 frames of each class) and on the Gaussian world (i.i.d.). The
   conclusion that the null does not depend on which half of the
   modality set is real is no longer supported. No threshold changed.
   Issues #74, #77.

### Erratum (2026-11-06) — phase-2 physiology header

13. `extract_force` now reads the phase-2 physio columns by name
   (item 10, issue #72). The column names and the sampling rate come
   from the BIDS sidecar `recording-cardresp_physio.json` (500 Hz;
   trigger, cardiac, respiratory); a missing column or sidecar stops
   the extraction with an error. On sub-01 ses-localizer run-1 the new
   force tensor holds the cardiac trace in channel 0 and the
   respiratory trace in channel 3 (checked on 2026-11-06). Every
   4.5-modal value was computed on the April tensors, probably with
   the old header reading (#72; not verified, the code is unchanged
   since `d7a3645`). No value is recomputed: those tensors are lost and
   the public data still give 120 vision frames against 1800 (issue
   #45). No value is changed.
   Evidence only, not used by any reported value: a phase-2
   `force.pt` was regenerated on 2026-11-06 with the corrected reader
   from sub-01 ses-localizer task-movielocalizer run-1 (48001 rows at
   500 Hz, sidecar `recording-cardresp_physio.json`) by calling
   `extract_force(src, 1800)` directly; 961 frames carry signal and
   839 are zero padding. SHA-256 before (regen35,
   `data/sf_phase2_adapted/force.pt`)
   `659cf150a435fcb00f9759d2690a45eac3391734927bf691e1d5d882ab6e56da`,
   after `052ffa1a2411c000094d7d640cd14182a73c9568bde39ed55476c133011d141c`
   (both re-hashed on macM1 on 2026-11-07). That tensor was built with
   the reader of PR #82; the reader of this item gives an identical
   tensor on the same file (`torch.equal`, checked on 2026-11-07).
   The tensor and its MANIFEST are kept on macM1 in
   `~/hypneum-lab/physio-fix-20261106/`, outside regen35, which is
   untouched. The April tensors (#45) remain
   lost and the 4.5-modal values are not recomputed.

## Quickstart

```bash
# nerve-wml is a sibling path dependency (`[tool.uv.sources]`, not on PyPI);
# CI checks it out at tag v1.7.0 next to this clone
git clone --branch v1.7.0 https://github.com/hypneum-lab/nerve-wml.git ../nerve-wml
uv sync --all-extras
uv run bouba-sens version          # bouba_sens 0.5.11
uv run pytest tests/unit tests/smoke   # 372 unit+smoke tests (CI command)
# Single cell (congenital T1, no checkpoint; --out is required)
uv run bouba-sens lesion --world gaussian --seed 0 --modality audio \
    --timing T1 --snr-init 0 --snr-floor -20 --out runs/cell_t1_audio
# Single grid (≈20 min on M3 Ultra, 5-way seed concurrency)
WORLD=studyforrest LOCK_AFTER=100 STEPS_TRAIN=200 STEPS_LESION=100 \
    OUT_ROOT=runs/sprint10_peak METRICS="Me1,Me2,Me3" \
    bash scripts/run_grid.sh
# One-command reproduction of every grid cited in the paper (~8 h)
bash scripts/reproduce_paper_v01.sh
# Verify released artefacts byte-for-byte
uv run python scripts/sha256_manifest.py
diff <(jq -S . reports/MANIFEST.json) <(jq -S . reports/MANIFEST.json.released)
```

## Reproduce in one command

```bash
git clone --branch v1.7.0 https://github.com/hypneum-lab/nerve-wml.git ../nerve-wml  # sibling path dep (not on PyPI; CI pins v1.7.0)
git clone https://github.com/hypneum-lab/bouba_sens.git && cd bouba_sens
uv sync --all-extras && uv run bouba-sens version        # bouba_sens 0.5.11
STEPS_TRAIN=20 STEPS_LESION=20 bash scripts/v01_smoke.sh  # train → lesion (audio, T2) → eval (Me1,Me2,Me7) → HTML; drop the env vars for the 100/100-step default
```

Expected output: `runs/v01_smoke_<ts>/{phase1/model.pt,phase1/checkpoint.pkl,phase2_t2/report.pkl,phase2_t2/eval_report.json,phase2_t2/per_query_me1.json,summary.html}`; last log line `v01 smoke done; open …/summary.html`.
`eval_report.json` has keys `me1, me1_probe, me2, me3_delta, me6_max_abs, me7, me8, me9` (`me1 ≈ 0.4`, `me2 ≈ 0.25–0.27` at 20/20 steps; `me7` is `null` in a single-cell run — it needs a paired T1/T2 cell).
Duration: ≈ 4 min 45 s per run at 20/20 steps on an 8-core Xeon X5680 (Linux, CPU only, 1 core used, max RSS 0.9 GB); the 100/100-step default exceeds 15 min on that CPU (Phase 1 alone ≈ 5 min 40 s), far less on Apple Silicon.
Determinism: bit-exact in the seeded test suite; the CLI training init is not yet seeded, so two runs do not give byte-identical `eval_report.json` (tracked as an issue).
Full reproduction: `bash scripts/reproduce_paper_v01.sh` (every grid cited in paper v0.1, ≈ 8 h on an M3 Ultra, fetches Studyforrest ds000113 first).
Hash manifest: `uv run python scripts/sha256_manifest.py --out /tmp/MANIFEST.json` regenerates the `reports/` manifest for comparison with the committed `reports/MANIFEST.json` (the default `--out` overwrites it).

## Dependencies

- Python 3.14
- PyTorch ≥ 2.5
- `nerve-wml` v1.5.3+ (protocol + methodology: Kraskov, MINE,
  bootstrap CI, null model permutation — Hypneum Lab,
  concept DOI `10.5281/zenodo.19656342`, resolves to the latest release).
  Installed from a sibling clone at `../nerve-wml` (`[tool.uv.sources]`
  in `pyproject.toml`); CI pins tag `v1.7.0`.

## Priority references

1. Amedi et al. 2007 — Shape conveyed by visual-to-auditory sensory substitution activates LOC (Nat Neurosci).
2. Heimler & Amedi 2020 — Are critical periods reversible in the adult brain? Insights on cortical specializations based on sensory deprivation studies (Neurosci. Biobehav. Rev.).
3. Röder, Kekunnaya & Guerreiro 2021 — Neural mechanisms of visual sensitive periods in humans (Neurosci. Biobehav. Rev. 120, 86–99; doi:10.1016/j.neubiorev.2020.10.030).
4. Alper & Averbuch-Elor 2023 — Kiki or Bouba? Sound Symbolism in Vision-and-Language Models (NeurIPS).
5. Girdhar et al. 2023 — ImageBind: One Embedding Space (CVPR).
6. Ma et al. 2022 — Are Multimodal Transformers Robust to Missing Modality? (CVPR).
7. Liang et al. 2021 — MultiBench (NeurIPS D&B).
8. Keller & Mrsic-Flogel 2018 — Predictive Processing: A Canonical Cortical Computation (Neuron).
9. Kleinman, Achille & Soatto 2023 — Critical Learning Periods for Multisensory Integration in Deep Networks (CVPR).

## AI assistance (transparency)

This repository is developed by a one-person lab with AI assistants
used as tools, never as authors:

- **Claude Code (Anthropic)** assists with code, documentation,
  operations and literature search in interactive sessions driven by
  the PI.
- **Codex CLI (OpenAI, `gpt-6.1-sol` / `gpt-6-astra`)** runs
  independent review passes before a release, a pre-registration or a
  submission; the briefs, reports and verdict tables are published next
  to the reviewed material whenever a pass is run.

Every scientific claim, verdict and pre-registration decision is taken
by the PI; results are reproduced by the test suite and CI, and review
reports are evidence, not endorsements. No model is listed as a
co-author of a publication or as a co-author of a commit.

## License

MIT (code) + CC-BY-4.0 (docs). See `LICENSE`.

## Citation

See `CITATION.cff`.
