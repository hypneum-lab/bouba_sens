# ADR-0001 — Codebook sharing between SensoryWMLs

**Status:** Proposed — execution pending on Studio (M3 Ultra).
**Date:** 2026-04-20
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

**EXECUTION PENDING** — the spike script has been written but NOT executed.

Per the project's compute-routing directive, the spike must be run on Studio
(M3 Ultra 512GB); GrosMac (M5 dev machine) is reserved for lightweight
orchestration and is forbidden for experimental runs. Once Studio produces
`out/oq1_results.json`, update this ADR with:

**Template — fill in after Studio run:**

> **Chosen:** SHARED 64-code alphabet across SensoryWMLs for v0.1.
>
> **Evidence:**
> - Spike results from `out/oq1_results.json` (5 seeds × 2000 steps × 2 modes).
> - Shared: acc = `<XX.XX>%` ± `<X.XX>%`
> - Local:  acc = `<XX.XX>%` ± `<X.XX>%`
> - Delta: `<sign and magnitude>`.
>
> **Decision rule:** if `|shared - local| <= 0.02` (2 % accuracy), shared wins
> by Occam; if `local > shared + 0.05`, local wins; otherwise re-run with 10
> seeds or deeper architecture before deciding.

Until then, `configs/v0.1_intact.yaml` uses the *conservative default*
(`shared=True`, see `architecture.k_letters` comment) so downstream Sprint 1
work is not blocked.

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
