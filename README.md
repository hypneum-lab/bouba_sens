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

**v0.0.1 — Sprint 0 scaffolding.** No working implementation yet.
Design: `docs/superpowers/specs/2026-04-20-bouba-sens-design.md`.
Plan: `docs/superpowers/plans/2026-04-20-bouba-sens-sprint0.md`.

## Quickstart (once Sprint 1 lands)

```bash
uv sync --all-extras
uv run bouba-sens version
uv run pytest
```

## Dependencies

- Python 3.14
- PyTorch ≥ 2.5
- `nerve-wml >=1.1.4,<1.2` (neuroletters, γ/θ multiplexing — Hypneum Lab)

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
