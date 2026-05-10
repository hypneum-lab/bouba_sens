# Cover Letter — TMLR submission

**Manuscript title.** *bouba_sens: A Pre-Registered Benchmark for
Cross-Modal Plasticity under Critical-Period Constraints*
**Author.** Clément Saillant (Hypneum Lab)
**Submitted.** [date to fill at submission]
**Manuscript file.** `docs/paper/main.tex` (compiled
`docs/paper/build/main.pdf`)
**Repository.** https://github.com/hypneum-lab/bouba_sens
**Pre-registration.** OSF `10.17605/OSF.IO/Q6JYN` (locked
2026-04-19, amendment v0.6 2026-04-23)
**Software DOI.** Zenodo (concept) — to be minted alongside this
submission via `.zenodo.json` already at repo root
**Companion software.** `nerve-wml` v1.5.3
(DOI `10.5281/zenodo.19666405`)

---

## To the Action Editor

We submit *bouba_sens* for consideration as a TMLR Featured
Certification candidate (Reproducibility track), with audience
**ML researchers working on multimodal learning, cross-modal
transfer, embodied AI, sensory substitution, and the
neuroscience–AI interface**.

### Why TMLR

The manuscript fits TMLR's reviewing criteria as follows:

1. **Claims supported by evidence.** Every empirical statement in
   the manuscript is backed by an OSF-pre-registered grid (9
   grids, 150 cells each, 5 seeds × 5 modalities × 2 timings × 3
   SNRs) with thresholds locked **before** the evaluation
   campaign. Eighteen ADRs (`docs/adr/0004..0017`) record every
   verdict, retraction, and scope change inline with the
   commit history. The byte-identity of all reported numbers is
   verifiable via `scripts/sha256_manifest.py` against
   `reports/MANIFEST.json`.
2. **Audience.** The benchmark targets researchers studying
   modality-loss generalisation in deep models — a sub-community
   underserved by current vision/language benchmarks, which
   typically assume modality completeness.
3. **No requirement on perceived significance.** We make a
   **narrow and falsifiable** central claim: a frozen PSK mux
   with hard-binary transducer gating is the **minimal
   architectural configuration** that qualitatively reproduces
   the Amedi 2007 T1/T2 asymmetry on the real biological bridge.
   Every compound tested (soft Gumbel gating, finer τ scans,
   codebook freeze, phase-transition schedule) was null or
   harmful. We report the negative results with equal weight to
   the positive one.

### What is new

- **First pre-registered benchmark** for cross-modal plasticity
  measurable on a **real** Studyforrest Phase-2 4.5-modal
  bridge, not a synthetic stand-in.
- **Three orthogonal invariants** (B-1 T1/T2 gap, B-2
  informational migration, B-3 perceptive/proprioceptive
  asymmetry) decouple effects that prior single-axis benchmarks
  conflate.
- **Critical-period dose-response curve.** §5.5 of the
  manuscript reports the first non-monotone Amedi-style
  dose-response curve obtained on the 4.5-modal real bridge:
  B-1 peaks at `LOCK_AFTER=100` (50 % of `STEPS_TRAIN`) at
  `+0.0125` (0.25 × the OSF-locked threshold of `0.05`).
- **Reproducibility infrastructure.** `bash
  scripts/reproduce_paper_v01.sh` regenerates every figure and
  table from a clean checkout in under [N] minutes on commodity
  hardware; the SHA-256 manifest catches any drift.

### What is **not** claimed

We **do not** claim that bouba_sens is a sufficient cross-modal
plasticity benchmark for all questions in the area. The current
release deliberately scopes to three invariants on five worlds +
one real bridge. §6 enumerates eight specific extensions
(multi-subject Studyforrest, longer training horizons, on-line
continual setting, larger-scale acoustic worlds, …) marked as
explicit out-of-scope items.

We also **do not** claim novelty for the architectural primitives
themselves (PSK mux, hard-binary transducer). Their novelty is in
the **falsified ablation matrix**: the manuscript demonstrates
which configurations produce Amedi-style behaviour and which do
not, on a benchmark whose thresholds were locked before the
runs.

### Reviewer recommendations

We propose the following potential reviewers (any subset is
welcome). Selection criteria : (i) at least one publication in
2023-2025 directly overlapping a manuscript section ; (ii) no
co-authorship with the present author within the past three
years ; (iii) institutional and topical diversity across the
four-person panel.

1. **Cees G. M. Snoek** (University of Amsterdam, VIS Lab) —
   *multimodal learning under missing-modality constraints.*
   Co-author of "Learning Unseen Modality Interaction"
   (NeurIPS 2023) and the AnyTouch / OmniBind line of work on
   modality-imbalanced representation learning. Direct
   relevance to §4 (worlds with degraded or absent modalities)
   and §6 (out-of-scope discussion of sufficient-modality
   assumptions). No prior collaboration.

2. **Tessa M. Dekker** (University College London,
   Institute of Ophthalmology) — *cross-modal cortical
   plasticity, computational fMRI methodology.*
   Corresponding author on *Hierarchical cortical plasticity in
   congenital sight impairment* (bioRxiv 2024,
   doi:10.1101/2024.07.04.602138). Provides an independent
   neuroscience perspective on the B-1 congenital-blindness
   invariant ; methodologically rigorous on individual
   variability in plasticity. No Amedi-lab affiliation.

3. **Jessica Hullman** (Northwestern University) —
   *pre-registration methodology in ML.* Co-author of
   *Pre-registration for Predictive Modeling*
   (arXiv:2311.18807), the canonical recent reference on
   transposing OSF-style pre-registration into the predictive
   modeling setting. Direct relevance to §3 methodology and
   §6.6 reproducibility narrative. Provides editorial-grade
   authority on the pre-reg framing.

4. **Alessandro Achille** (AWS AI / Caltech) — *critical-period
   theory in deep neural networks.* First author on *Critical
   Learning Periods Emerge Even in Deep Linear Networks*
   (ICLR 2024 spotlight) and the foundational 2019 critical-
   period series. Direct relevance to §5.5 (the Amedi-style
   dose-response curve framed as a critical-period signature
   in artificial networks). Bridges biological and ML
   communities cleanly.

We are open to additional or substitute names from the action
editor's network ; the OpenReview submission will mirror this
list with any portal-specific adjustments.

### Conflicts to declare

None known. The single author has no funding from organisations
that would benefit from a particular outcome, and no
co-authorship within the past three years with any of the
proposed reviewers.

### Open Science Statement

- Code: MIT-licensed, public on GitHub at
  `github.com/hypneum-lab/bouba_sens`.
- Data: Studyforrest Phase-2 publicly available under the original
  studyforrest data licence; CC-licensed audio substitute for the
  fifth modality; all synthetic worlds regenerable from
  `scripts/build_worlds.py` + frozen seeds.
- Pre-registration: OSF `10.17605/OSF.IO/Q6JYN`, public,
  timestamped 2026-04-19, amendment v0.6 dated 2026-04-23 covers
  the post-lock scope adjustments.
- Software DOI: Zenodo version-DOI minted at submission via the
  GitHub-Zenodo webhook on the next tag (`.zenodo.json` already
  at repo root with full metadata).
- Companion software DOI: `nerve-wml` v1.5.3 archived at
  `10.5281/zenodo.19666405`; the Paper depends on it for the
  cross-substrate measurement primitive.

### Author contributions

Sole author. Conceptualisation, methodology, software,
benchmark design, OSF pre-registration, all empirical runs,
manuscript drafting, and revision.

### Concurrent work

We note for completeness one concurrent work that touches
adjacent topics:

- **RecursiveMAS** (Yang et al., arXiv:2604.25917, 2026-04-28)
  proposes a latent-space recursive computation across
  heterogeneous LLM agents. Although the engineering target
  differs (multi-agent LLMs, not cross-modal plasticity
  benchmarking), the underlying intuition that *shared latent
  state outperforms token-level exchange* is conceptually
  adjacent to bouba_sens's frozen PSK mux gating result. The
  parent research programme's Paper 2 (dream-of-kiki)
  positioning paragraph and rebuttal pocket address this
  proximity in detail; bouba_sens itself does not depend on
  the comparison.

We are not aware of any other concurrent or prior work
producing a pre-registered Amedi-style cross-modal plasticity
benchmark on a real biological bridge.

---

## Submission checklist (TMLR)

- [ ] Manuscript compiles cleanly with `tmlr.sty` in
      under-review mode (`\usepackage{tmlr}` not
      `\usepackage[accepted]{tmlr}`).
- [ ] Anonymisation: TMLR is single-blind (author name is
      visible in submission); confirm OpenReview profile is
      complete.
- [ ] All figures are vector (PDF) where possible; raster
      figures pinned at 300 DPI.
- [ ] Bibliography compiles (`bibtex main`); no broken
      `\cite{}` references.
- [ ] Repository tag created and pushed; tag name to be cited
      in §1 of the manuscript.
- [ ] Zenodo version-DOI minted via webhook; updated in §1
      `.zenodo.json` reference.
- [ ] OSF pre-registration link is publicly resolvable.
- [ ] `scripts/reproduce_paper_v01.sh` end-to-end run validated
      on a fresh checkout within the past 7 days.
- [ ] SHA-256 manifest matches `reports/MANIFEST.json` for every
      reported number.
- [ ] Cover letter PDF (this file rendered) attached to
      OpenReview submission.

---

*To render this cover letter as PDF for OpenReview attachment:*

```bash
pandoc docs/paper/cover-letter-tmlr.md \
       -o docs/paper/build/cover-letter-tmlr.pdf \
       --pdf-engine=xelatex \
       -V mainfont="Helvetica Neue" \
       -V geometry:margin=1in
```
