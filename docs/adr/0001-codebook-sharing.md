# ADR-0001 — Codebook sharing between SensoryWMLs

**Status:** Accepted — empirical evidence collected on Studio 2026-04-20.
**Date:** 2026-04-20
**Date run:** 2026-04-20 on Studio (MacStudio-de-MonsieurB.local, arm64, macOS 26.4).
**Authors:** Clément Saillant
**Related:** Spec `docs/superpowers/specs/2026-04-20-bouba-sens-design.md` OQ1 + §2.2 principle 2.

## Context

`bouba_sens` couples five `SensoryWML` instances through a `CrossModalNerve`
that routes neuroletters (64-code alphabet) between them. The `nerve-wml`
invariant **N-5** says each WML has a local codebook. For cross-modal
compensation to work, codes must have *comparable semantics* across WMLs —
either by sharing the alphabet outright (violation of N-5) or by adding a
learnable `CodebookAligner` module.

OQ1 asks which of these two designs to ship in v0.1. A miniature spike
(`scripts/spikes/oq1_codebook.py`) compares a `SharedCodebookModel` and a
`LocalCodebookModel` on a toy 2-modality classification task, 5 seeds × 2000
steps, and prints final accuracy ± std.

## Decision

**Chosen:** SHARED 64-code alphabet across SensoryWMLs for v0.1.

The two designs are statistically indistinguishable on classification accuracy
(Δ = -0.039 %, well inside the ±2 % Occam band). LocalCodebookModel carries
one extra `Linear(2K, K)` aligner (~8 k params at K = 64) with no measurable
accuracy benefit, so Occam's razor picks the shared design.

`configs/v0.1_intact.yaml` keeps `shared=True` as the committed default.

## Empirical evidence

Spike: `scripts/spikes/oq1_codebook.py --mode both --seeds 5 --steps 2000`,
results in `out/oq1_results.json`. Toy task = binary classification on a
2-modality sample drawn from a shared 8-D latent (see script docstring).

| Mode                | Final accuracy (n=5) | Final loss (n=5)     |
|---------------------|----------------------|----------------------|
| SharedCodebookModel | 98.945 % ± 0.212 %   | 0.0713 ± 0.0102      |
| LocalCodebookModel  | 98.906 % ± 0.319 %   | 0.0222 ± 0.0060      |
| **Δ (local−shared)** | **−0.039 %**        | **−0.0491**          |

**Decision rule applied:** `|acc_shared − acc_local| = 0.039 % ≤ 2 %` → shared
wins by Occam. No re-run needed.

**Note on loss gap.** LocalCodebookModel reaches a lower cross-entropy
(0.022 vs 0.071) without a matching accuracy gain — the aligner produces
sharper logits on already-correct predictions, not fewer errors. This
confirms the aligner's extra capacity is spent on calibration, not on
separating classes, and reinforces the Occam choice.

**Decision rule (original, for reference):** if `|shared − local| ≤ 0.02`
(2 % accuracy), shared wins by Occam; if `local > shared + 0.05`, local
wins; otherwise re-run with 10 seeds or deeper architecture before deciding.

## Consequences

- Explicit, documented violation of nerve-wml invariant N-5, scoped to this
  repository. A note is added to `src/bouba_sens/nerve.py` docstring when
  Sprint 2 implements `CrossModalNerve`.
- If v0.2 empirical results (§4.5, §7.3 R4) show compensation degeneracy, we
  revisit and potentially add a `CodebookAligner` — producing a new ADR that
  supersedes this one.

## nerve-wml API gap observations (from Task 0.7)

- **`GammaThetaMultiplexer`** — not present in `track_p.oscillators` as of
  nerve-wml v0.1.0; only `PhaseOscillator` is exposed. Tracked upstream as
  issue #1 on `hypneum-lab/nerve-wml`. Re-enable the symbol in
  `tests/smoke/test_nerve_wml_api.py::REQUIRED_SYMBOLS` once it lands.
- All four currently-required symbols (`Nerve`, `Neuroletter`, `MlpWML`,
  `Transducer`) are present — contract test passes 4/4 as of 2026-04-20.

## Revisit criteria

Re-open this ADR if any of the following holds:

1. Sprint 2 integration shows > 10 % accuracy degradation vs the spike baseline.
2. An empirical test `test_B2_mi_migration` fails consistently across seeds.
3. A peer review (internal or external) raises the violation of N-5 as a
   correctness concern.
4. A follow-up spike run (with 10 seeds or deeper architecture) flips the
   decision direction recorded above.
