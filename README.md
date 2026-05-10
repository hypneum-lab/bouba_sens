# bouba_sens

> A benchmark for cross-modal plasticity in artificial neural systems.
> Hypneum Lab — 2026.

`bouba_sens` studies how a 5-modality agent (audio, vision, tactile, gravity,
force) reorganises itself when one sensory channel is lost or degraded —
inspired by the cross-modal cortical recruitment observed in congenital and
late blindness (Amedi 2007, Merabet 2010, Heimler 2020).

## Three testable invariants

- **B-1 — Congenital gap:** lesion pre-training yields better adaptation than
  lesion post-convergence.
- **B-2 — MI migration:** mutual information between surviving-modality
  neuroletters and the target label rises post-lesion.
- **B-3 — Perceptive/proprioceptive asymmetry:** losing vision/audio/tactile
  produces a quantitatively different plastic response than losing
  gravity/force.

## Status

**v0.5.5 (2026-04-23) — Paper v0.1 submit-ready.** Sprints 0–17
closed, 14 ADRs (0004–0017), 185+ tests. 9 grids (150 cells each)
on the 4.5-modal real Studyforrest Phase-2 bridge + 5 worlds
synthetic.

- Paper draft : `docs/paper/paper-v0.1-draft.md`
- OSF registration : `10.17605/OSF.IO/Q6JYN`, amendment v0.6
- Target venue : TMLR benchmarks track (primary), NeurIPS D&B (fallback)

## Headline findings (v0.5.5)

Thresholds locked by OSF before the campaign : B-1 ≥ 0.05,
B-2 ≥ 0.10, B-3 ≥ 0.02 ; no threshold changes across all 14 ADRs.

| Invariant | Threshold | Best grid-median | Seed-stability | Verdict |
|-----------|----------:|-----------------:|:---------------|---------|
| **B-3** Me6 perceptive/proprio asymmetry | 0.02 | **0.109–0.125** (5.5–6.3×) | 5/5 pass | **PASS** every configuration, architectural invariant |
| B-1 Me7 congenital gap | 0.05 | **+0.0125** (25 % of thr) | 3+/5 positive | **Qualitative** — only `LOCK=100 + HARD` on the real bridge (§5.5 Amedi dose-response) ; **N8-Q3 5-seed interim verdict `Retract` 2026-05-11, see CHANGELOG — TMLR blocked, Q3+ 10-seed pending** |
| B-2 Me3 MI migration | 0.10 | mean +0.033 ±0.047 bits | 0/5 cross threshold | **Null** — two independent estimators (Kraskov + MINE) agree within 0.1 bit |

**F1 — B-3 is an architectural invariant.** The perceptive /
proprioceptive asymmetry passes at 5–6× threshold across every
grid, every world, every seed, and is insensitive to every
plasticity control tested (LOCK_AFTER, transducer gating mode,
Gumbel tau, codebook freeze, HARD → GUMBEL phase transition).

**F2 — B-1 is qualitatively reproduced in exactly one
configuration.** On the 4.5-modal real biological bridge,
`constellation_lock_after=100` with a hard-binary transducer gate
produces a seed-stable +0.0125 Me7 (Sprint 10 peak, ADR-0013).
Every compound attempted since has preserved, weakened, or
destroyed this peak — *never* amplified it. The paper's central
empirical claim is this narrow : a frozen mux with hard
transducer gating is the *minimal* configuration that
qualitatively reproduces the Amedi T1/T2 asymmetry.

**F3 — B-2 is below threshold, estimator-resolved.** The earlier
"bimodal positive B-2 at tau=0.30" claim (ADR-0016) was retracted
after per-seed re-aggregation (ADR-0017 §14a). Sprint 17 replicates
the §6.3 Kraskov limit with MINE (Donsker-Varadhan) on the Sprint 10
peak grid : both estimators agree within 0.1 bit that B-2 does not
exceed threshold on any seed. The remaining ambiguity concerns
probe shape (1-D mean-pool vs full `(B, d_fused)`), not estimator
choice — declared follow-up for paper v0.2.

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
2. Heimler & Amedi 2020 — Revisiting adaptive and maladaptive effects of crossmodal plasticity (Neuroscience).
3. Röder et al. 2021 — Sensitive periods for functional specialization (PNAS).
4. Alper & Averbuch-Elor 2023 — Kiki or Bouba? Sound symbolism in VLMs (NeurIPS).
5. Girdhar et al. 2023 — ImageBind: One Embedding Space (CVPR).
6. Ma et al. 2022 — Are Multimodal Transformers Robust to Missing Modality? (CVPR).
7. Liang et al. 2022 — MultiBench (NeurIPS D&B).
8. Keller & Mrsic-Flogel 2018 — Predictive Processing: A Canonical Cortical Computation (Neuron).

## License

MIT. See `LICENSE`.

## Citation

See `CITATION.cff`.
