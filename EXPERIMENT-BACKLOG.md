# Experiment backlog — bouba_sens

From the external review (REVIEW.md). Run in a fresh session with the repo env
(`uv sync`). R1 `run_id`; this revises a published methodology recommendation, so update
the relevant ADR + paper draft in lockstep.

## E1 — B-2 re-run with MINE at dimensionally-justified N  [moderate]  — **DONE (ADR-0019)**
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
**Outcome (DONE)**: Me3 probe ambient dim 896, **effective D ≈ 12** (participation ratio
11.7 / two-NN 12.4). The Me3 sonde was made configurable (full `(B, 896)` default);
the old 1-D mean-pool **pinned MINE to exactly 0.000** — a probe-shape artefact, not a
two-estimator agreement. Re-estimated with MINE on the full 896-D probe + Kraskov +
adaptive binning, all bracketing zero with bootstrap CIs; **no estimator crosses +0.10**.
Binning collapses in 896-D (`N ≫ (#bins)^D`). **Verdict B-2 Null confirmed/renforced**,
on gaussian **and on both real bridges** (Studyforrest ds000113 + CC-film). Artefacts:
`reports/e1_b2_dimensional_mine_{studyforrest,ccfilm}.json`,
`scripts/spikes/e1_b2_dimensional_mine.py`, `scripts/spikes/e1_b2_full_sonde.py`.

## E2 — Re-derive the sample-size criterion N ≳ f(dim)  [analysis]  — **DONE (ADR-0019)**
**Why**: replace "n≥128" with a dimensionally-grounded bound. MI/entropy estimation needs
N growing with effective dimensionality (plug-in: N ≫ alphabet size; k-NN/MINE: N scaling
with intrinsic dimension). Document `N ≳ f(D)` with the chosen estimator's known
sample-complexity, and apply it across all probe metrics. Update the methods section + ADR.
**Outcome (DONE)**: the fixed "≥128" heuristic is replaced by the estimator-aware bound
**N ≳ f(D)** keyed on the measured effective dimension D (not ambient dim or batch size).
Documented in ADR-0019 and paper draft §6.3.1(b); D is now reported alongside every MI
estimate, and any N ≲ 10³ for the Me3 probe is treated as resolving *sign*, not *magnitude*.
