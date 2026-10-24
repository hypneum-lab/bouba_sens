# bouba_sens

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
| **B-3** Me6 perceptive/proprioceptive asymmetry | **PASS on magnitude** (5–6× threshold — unchanged) ; the "architectural property" reading is withdrawn (entropy proxy, paper §3.3/§4.4) |
| **B-1** Me7 congenital gap (§5.5) | **FINAL Retract** (Q3+ 10-seed, ADR-0019) — reformulation **landed in manuscript** |
| **B-2** Me3 MI migration | **Null** (confirmed/renforced — ADR-0019: full-D probe, N≳f(D), two real bridges) |
| **TMLR submission** | **BLOCKED** — §5.5 retraction landed; pending critic-review gate + N≥30 decision |
| **N12 subgroup replication** | **DONE — LOST 0/2** at Bonferroni α=0.025 (ADR-0019) |

Cumulative on main HEAD `3fbeee7` : **156 commits**, **v0.5.9**, 19
ADRs (0004–0019), 185+ tests, pooled-v2 analysis + N12 pre-reg
locked. Honest scope ; no overselling.

---

## Status

**v0.5.9 — Paper v0.1 draft (TMLR submission blocked; see
2026-05-11 update above).** Sprints 0–17 closed, 19 ADRs
(0004–0019), 185+ tests. 9 grids (150 cells each) on the
4.5-modal real Studyforrest Phase-2 bridge + 5 worlds synthetic.
§5.5 B-1 retraction now landed in the manuscript (ADR-0019).

- Paper draft : `docs/paper/paper-v0.1-draft.md`
- OSF registration : `10.17605/OSF.IO/Q6JYN`, amendment v0.6
- Draft OSF amendments v0.7–v0.9 (2026-10-23), **DRAFT — not yet filed
  on OSF (to be filed by the PI)**: TOST equivalence for B-1/B-2
  (`docs/osf/amendment-v0.7-tost-equivalence-2026-10-23.md`),
  subspace shift vs `LOCK_AFTER`
  (`docs/osf/amendment-v0.8-subspace-shift-2026-10-23.md`),
  decorrelation-window lesion
  (`docs/osf/amendment-v0.9-decorrelation-window-2026-10-23.md`).
  None alters Q6JYN or ADR-0019; power analysis in
  `scripts/power_prereg_2026_10.py`.
- Target venue : TMLR benchmarks track (primary), NeurIPS D&B (fallback)

## Headline findings (v0.5.5)

Thresholds locked by OSF before the campaign : B-1 ≥ 0.05,
B-2 ≥ 0.10, B-3 ≥ 0.02 ; no threshold changes across all 14 ADRs.

| Invariant | Threshold | Best grid-median | Seed-stability | Verdict |
|-----------|----------:|-----------------:|:---------------|---------|
| **B-3** Me6 perceptive/proprio asymmetry | 0.02 | **0.109–0.125** (5.5–6.3×) | 5/5 pass | **PASS** on magnitude in every configuration ; magnitude tracks per-modality entropy, not the partition (§4.4) |
| B-1 Me7 congenital gap | 0.05 | **+0.0125** (25 % of thr) | 3+/5 positive | **Qualitative** — only `LOCK=100 + HARD` on the real bridge (§5.5 Amedi dose-response) ; **N8-Q3 5-seed interim verdict `Retract` 2026-05-11, see CHANGELOG — TMLR blocked, Q3+ 10-seed pending** |
| B-2 Me3 MI migration | 0.10 | mean +0.031 ±0.047 bits (full-probe MINE) | 0/5 cross threshold | **Null** — confirmed/renforced by ADR-0019: Kraskov + adaptive-binning + **full 896-D MINE** all bracket zero on synthetic + two real bridges |

**F1 — B-3 passes on magnitude only.** The Me6 asymmetry
magnitude passes at 5–6× threshold across every grid, every
world, every seed, and is insensitive to every plasticity control
tested (LOCK_AFTER, transducer gating mode, Gumbel tau, codebook
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
  on the gaussian reference **and on both real bridges** (Studyforrest +
  CC-film, see F4), regardless of which half of the modalities is real.

**F4 — Two real bridges with honestly-labelled channel provenance.**
B-2 Null was re-checked on two real-signal bridges with **opposite**
real/synthetic modality splits (complements, *not* a replacement of the
headline synthetic findings) :

- **Studyforrest (OpenNeuro ds000113).** force / gravity / tactile /
  labels = **REAL** ds000113 signal ; audio / vision = annotation
  **PROXY** (the Forrest Gump soundtrack and film are copyrighted and
  absent from OpenNeuro — labelled as proxies, not "real movie vision").
- **CC-film (Big Buck Bunny, CC-BY 3.0).** audio (log-mel) / vision
  (frames) / labels (shot-cuts) = **REAL** perceptual signal ; tactile /
  gravity / force = **SYNTHETIC** (a film has no proprioception).

**B-2 Null holds on both bridges plus the three synthetic worlds** — the
null is robust to which half of the modality set carries the true signal.
The abstract is aligned to this per-channel provenance (no "real movie
vision" overclaim).

Full trail : `docs/adr/0004` through `0017`. Every verdict,
retraction, and scope change is recorded inline with its grid
commit.

## Quickstart

```bash
uv sync --all-extras
uv run bouba-sens version          # bouba_sens 0.5.5
uv run pytest                      # 185+ items, all green
# Single cell
uv run bouba-sens lesion --world gaussian --seed 0 --modality audio \
    --timing T1 --snr-init 0 --snr-floor -20
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

## Dependencies

- Python 3.14
- PyTorch ≥ 2.5
- `nerve-wml` v1.5.3+ (protocol + methodology: Kraskov, MINE,
  bootstrap CI, null model permutation — Hypneum Lab,
  DOI `10.5281/zenodo.19666405`)

## Priority references

1. Amedi et al. 2007 — Shape conveyed by visual-to-auditory sensory substitution activates LOC (Nat Neurosci).
2. Heimler & Amedi 2020 — Are critical periods reversible in the adult brain? Insights on cortical specializations based on sensory deprivation studies (Neurosci. Biobehav. Rev.).
3. Röder et al. 2021 — Sensitive periods for functional specialization (PNAS).
4. Alper & Averbuch-Elor 2023 — Kiki or Bouba? Sound Symbolism in Vision-and-Language Models (NeurIPS).
5. Girdhar et al. 2023 — ImageBind: One Embedding Space (CVPR).
6. Ma et al. 2022 — Are Multimodal Transformers Robust to Missing Modality? (CVPR).
7. Liang et al. 2021 — MultiBench (NeurIPS D&B).
8. Keller & Mrsic-Flogel 2018 — Predictive Processing: A Canonical Cortical Computation (Neuron).
9. Kleinman, Achille & Soatto 2023 — Critical Learning Periods for Multisensory Integration in Deep Networks (CVPR).

## License

MIT. See `LICENSE`.

## Citation

See `CITATION.cff`.
