# Experiment backlog — bouba_sens

From the external review (REVIEW.md). Run in a fresh session with the repo env
(`uv sync`). R1 `run_id`; this revises a published methodology recommendation, so update
the relevant ADR + paper draft in lockstep.

## E1 — B-2 re-run with MINE at dimensionally-justified N  [moderate]
**Why**: B-2 retraction attributes the decay to a Kraskov k-NN artefact at n=16; the
"probe batch ≥128" recommendation is under-derived (MI sample complexity scales with
**dimensionality**, not batch size).
**Protocol**:
1. Compute the **actual probe dimensionality** D of the Me3 representation.
2. Set N from a principled criterion (see E2), not the heuristic 128.
3. Re-estimate B-2 MI with **MINE** (Belghazi 2018) at that N; cross-check vs Kraskov and
   adaptive binning.
**Acceptance**: confirm whether B-2 is genuinely 0 (network) or remains estimator-limited;
report MI with bootstrap CIs.

## E2 — Re-derive the sample-size criterion N ≳ f(dim)  [analysis]
**Why**: replace "n≥128" with a dimensionally-grounded bound. MI/entropy estimation needs
N growing with effective dimensionality (plug-in: N ≫ alphabet size; k-NN/MINE: N scaling
with intrinsic dimension). Document `N ≳ f(D)` with the chosen estimator's known
sample-complexity, and apply it across all probe metrics. Update the methods section + ADR.
