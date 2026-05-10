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
welcome):

1. **[Name 1]** — multimodal learning + sensory substitution
   methodology
2. **[Name 2]** — cross-modal cortical plasticity
   (computational neuroscience)
3. **[Name 3]** — pre-registered benchmark methodology in ML
4. **[Name 4]** — critical-period theory / developmental ML

Recommendations are placeholders; final names attached at
submission via the OpenReview portal.

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
