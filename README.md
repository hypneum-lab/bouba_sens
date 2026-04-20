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

**v0.3.0 (2026-04-20).** Feature-complete v0.2 design + Sprint 6 Tasks 6.1–6.2 (cross-world replication) merged. 5 ADRs, 152 tests, 66 commits. Sprints 0 → 5 closed ; Sprint 6 open (Tasks 6.3+ = paper draft).

Design: `docs/superpowers/specs/2026-04-20-bouba-sens-design.md`.

## Headline findings (v0.3.0)

Three 150-cell grids on Studio (M3 Ultra, ~17 min each) across three structurally different synthetic worlds — GaussianWorld (orthogonally-projected latent), XORWorld (Rademacher parity), SinusoidWorld (circular latent). Same protocol, same fixed thresholds from the OSF pre-registration.

| Invariant | Threshold | gaussian | xor | sinusoid | Verdict |
|-----------|----------:|---------:|----:|---------:|---------|
| **B-3** Me6 perceptive/proprio asymmetry | 0.02 | **0.148** | **0.141** | **0.156** | **3/3 PASS at ~7-8× threshold** |
| B-1 Me7 congenital gap | 0.05 | -0.006 | -0.006 | **+0.013** | 3/3 FAIL, sign flips on sinusoid |
| B-2 Me3 MI migration | 0.10 | 0.028 | 0.004 | 0.002 | 3/3 FAIL, decays with world complexity |

**F1 — B-3 is world-agnostic.** The perceptive/proprioceptive asymmetry (audio/vision/tactile vs gravity/force) passes on three structurally divergent synthetic worlds at ~7-8× the pre-registered threshold. First cross-world replicated finding of the Hypneum Lab programme.

**F2 — B-1 directionality is topology-dependent.** On orthogonal-factored worlds (Gaussian, XOR), late-acquired lesions recover at least as well as congenital ones (classic critical-period ordering *reversed*). On circular-latent topology (Sinusoid), the ordering holds. Seeds hypothesis H-B1 for a future OSF amendment.

**F3 — B-2 magnitude decays with world complexity.** MI migration is present but weak (Gaussian 0.028 > XOR 0.004 > Sinusoid 0.002), never reaching threshold. The 0.10 threshold was implicitly calibrated Gaussian-like.

Full ADRs: `docs/adr/0003-v01-empirical-verdicts.md`, `0004-v02-invariant-verdicts.md`, `0005-cross-world-replication.md`.

### Limitations — critical validation pending (Sprint 7)

The three findings above are **preliminary**. None has yet been stress-tested against the obvious reviewer objections :

- **B-3 may be tautological.** The perceptive / proprioceptive grouping (3 + 2) is baked into the protocol. A random-partition control (e.g. {audio, gravity} vs {vision, tactile, force}) must show that arbitrary 3 + 2 partitions do *not* pass the 0.02 threshold. If they do, **B-3 measures size-3-vs-size-2 dynamics, not cognitive asymmetry**. Sprint 7 Task 7.1.
- **B-1 sign flip may be noise.** Effect sizes (| 0.006 – 0.013 |) are 5–10× below the 0.05 threshold. A bootstrap 95 % CI on Me7 median across seeds must separate the three worlds for the "topology-dependent" claim to survive. Sprint 7 Task 7.2.
- **B-2 decay may be estimator-specific.** The Kraskov k-NN MI estimator is noisy at high ambient dimension. The Gaussian > XOR > Sinusoid ordering must hold under at least one alternative estimator (binning or MINE). Sprint 7 Task 7.3.

Until Sprint 7 lands, the paper draft flags these findings as *"preliminary empirical, pending critical controls"*. Plan : `docs/superpowers/plans/2026-04-20-bouba-sens-sprint7.md`.

## Quickstart

```bash
uv sync --all-extras
uv run bouba-sens version          # bouba_sens 0.3.0
uv run pytest                      # 152 items, all green
# Reproduce one cell
uv run bouba-sens lesion --world gaussian --seed 0 --modality audio \
    --timing T1 --snr-init 0 --snr-floor -20
# Full 150-cell grid (≈17 min on M3 Ultra)
WORLD=gaussian STEPS_TRAIN=200 STEPS_LESION=100 \
    OUT_ROOT=runs/v02_grid METRICS="Me1,Me2,Me3" bash scripts/run_grid.sh
uv run python scripts/aggregate_grid.py \
    --root runs/v02_grid --out reports/v0.2_aggregate.json
```

## Dependencies

- Python 3.14
- PyTorch ≥ 2.5
- `nerve-wml` v1.2.3+ (neuroletters, γ/θ multiplexing, 3 substrates — Hypneum Lab)

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
