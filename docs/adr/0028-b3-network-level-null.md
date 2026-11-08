# ADR-0028 — B-3 network-level null: independent units, cyclic form, unit sign flip

**Status:** Proposed — draft, **not frozen, not signed off**. Nothing in
this ADR is decided; every choice below is a proposal listed again in
§16 of the draft (Decisions left to the PI). No grid, pilot, null,
synthetic check or power simulation has been run for it.
**Date:** 2026-10-31, revised 2026-11-08.
**Decision source:** none yet. Requested by the PI on 2026-10-31 after
the Me6 large-n diagnostic, to be frozen **before** any v0.7
ActionSense grid. Revised after two independent reviews of 2026-10-31
(REVISE / BLOQUANT) and the PI input of 2026-11-07.
**Builds on:** ADR-0021, ADR-0022, ADR-0023, ADR-0024, ADR-0026,
ADR-0027; `docs/prereg/me6-large-n/amendment-2026-10-31-diagnostic.md`
(piste C); `docs/milestones/2026-10-31-me6-large-n-results.md`;
`docs/milestones/2026-10-27-b3-lesion-interaction-results.md`;
`docs/osf/erratum-2026-10-31-v0.7-seeds.md`.
**Pre-registration draft:** `docs/prereg/b3-network-null/prereg-draft.md`
(the full design; this ADR records the decision to be taken).

## Context

Me6 compares P[a, b] (query a, network lesioned on b) with P[b, a]
(query b, network lesioned on a). Each column of the 5 × 5 matrix is a
**separate lesioned network**. The ADR-0022 paired sign flip holds the
five networks fixed and randomises item outcomes: it tests "these two
networks have the same per-item distribution", and has no access to
network-to-network variability.

The Me6 large-n grids (ADR-0026) showed the Gaussian Me6 median in
excess of that null at n ≥ 256 in both world-seed arms. The exploratory
diagnostic of 2026-10-31 pointed to **piste C**: a fixed per-cell
asymmetry (D1 split-half r ≈ 0.84–0.90, φ − 1 growing with n),
reproduced on fresh items (D4a), with no coherent direction across
trios (D3), appearing mainly after the lesion phase; a smaller
query-main-effect component (piste B) exists in the `cell` arm at
phase 1 (D4b-T2).

The frozen fixed rule (median Me6 > 0.02) is an extreme-value
statistic: under the ADR-0026 paired null it fired in every draw at
n ≤ 1024 and, at n = 4096, in 0.748 (`cell`) and 0.981 (`legacy`) of
the draws on the mock world, against 0.0028 / 0.0009 on the Gaussian
(`docs/milestones/2026-10-31-me6-large-n-results.md:75-78`). It decides
nothing about asymmetry.

The v0.7 ActionSense amendment (tag `bouba_sens/v0.7-actionsense-prereg`)
decides B-3 with the fixed rule and the ADR-0022 paired null. No
ActionSense grid has run.

The first draft (R replicates per cell sharing probe items and the T2
checkpoint, raw antisymmetric form, per-replicate flip) was found not
freezable: the raw form keeps the query main effects; components shared
by the replicates are read as signal; one negative control was
tautological (p = 1 by construction); no primary row, family,
predictions or justification of R.

## Proposed decision

1. **Units.** The exchangeable unit is an **independent pipeline run**:
   one grid seed Q_g = 330 000 + 200 · g with its own Phase-1 training,
   world (`cell` arm), T2 checkpoint, 30 lesion cells and probe items,
   run by the existing scripts with unchanged lesion and probe-seed
   rules. Pilot g = 0 … 5, confirmatory g = 20 … 20 + G − 1. The band is
   disjoint from every stream of every earlier run, every unit's
   streams are disjoint from every other unit's, and all seeds stay
   below the 2³² bound of `np.random.seed` (inventory in draft §3.1).
   Seeds 120–121 and 500–504 remain free but are not used: consecutive
   Phase-1 seeds share 199 of 200 batch seeds.
2. **Estimand: cyclic part.** C = A − (φ_a − φ_b), φ_a = (1/5) Σ_b A[a, b],
   which removes query and lesion main effects exactly (it equals the
   antisymmetrised residual of the 2026-10-26 additive fit). Dominance
   (gradient) asymmetries are declared not testable by this design.
3. **Statistic and null.** T = median over the 6 (timing, SNR)
   conditions of max over the 10 pairs of |mean over units of C|. Null:
   one sign per unit, applied to all its conditions and pairs;
   B = 20 000; observed, null median, null P99 and p reported. Exact
   under independence of units and central symmetry of each unit's
   tuple, asymptotically valid under independence and mean zero; both
   checked by the controls.
4. **Calibrated rule instead of the 0.02 rule.** Every decision is a
   same-pipeline p. The 0.02 rule is reported for continuity with its
   firing rate under the network null.
5. **One primary row, family F = {P1}, α = 0.01:** straight-through,
   Gaussian, `cell`, v2 probe, cyclic form. Calibration row K1
   (argmax-bridge, v1 n = 4096) is gate-exempt and descriptive.
   Predictions are written per row (draft §6.2).
6. **Controls that can fail:** synthetic size check on defined
   generators (Gaussian, bounded and sparse, skewed, main effects;
   shared components as sensitivity); NC1 no-lesion networks
   (real data, μ = 0 in mean by construction); NC2 disjoint unit pairs
   (premise check); PC planted pure 3-cycle at δ_pc = 0.02 on H0 data.
   Learning gate first for P1.
7. **G from a pilot** by a fixed power rule (smallest G in
   {20, 30, 40, 50} with power ≥ 0.80 at δ_pc, upper confidence bound of
   the pilot τ), two-stage freeze (Stage A before the pilot, Stage B
   before the confirmatory grid).
8. **Readings** in the lab vocabulary only: « confirmé », « nul TOST »
   (simultaneous TOST, margin 0.02), « non concluant » (with its
   reason), « non testé ». A non-rejection is never read as absence.
9. **Order.** Freeze this study before the v0.7 grids; a separate v0.7
   amendment, naming the frozen clauses it replaces (draft §15), may
   then add the network null.

## Code change it needs (described, not implemented)

- `scripts/run_grid_probe_v2.sh` / `scripts/run_grid.sh`: guard on
  |SEEDS|, refusal to resume a cell whose recorded lesion or probe seed
  differs from the rule, opt-in `NO_LESION=1` mode (stream-matched
  no-lesion jobs, directories `<cell>_nolesion`).
- `bouba-sens lesion --no-lesion` (`lesion_phase(lesion=None)`),
  recorded only when set; default outputs bit-identical.
- `scripts/me6_network_null.py` (new): units, cyclic projection, T,
  null, controls, TOST bounds, descriptive rows.
- `scripts/me6_network_null_synth.py` (new): size check and power.
- Tests: bit-identical default, projection identities, exact
  enumeration on a toy grid, NC2 not identically zero, full seed-stream
  disjointness.

## What this ADR does not change

No frozen metric, threshold, past verdict or preregistered reading of
ADR-0026, of the 2026-10-27 grid or of v0.7; no B-1 / B-2 / B-3 status.
The 0.02 rule stays in the frozen texts as registered; this study does
not use it to decide. The ADR-0022 paired null keeps its meaning
(conditional on the networks) and stays reported.

## Open decisions

Unit design; estimand; primary probe; α; equivalence margin; positive
control; G rule and ceiling; calibration row; world-seed arm; mock
world; learning gate; fixed module seeds of `lesion`; no-lesion
coverage; host; freeze stages; v0.7 follow-up; resampling seeds. The
list, with options and a recommendation, is in
`docs/prereg/b3-network-null/prereg-draft.md` §16.

## References

ADR-0021, ADR-0022, ADR-0023, ADR-0024, ADR-0026, ADR-0027;
`docs/osf/amendment-v0.7-actionsense.md`;
`docs/osf/erratum-2026-10-31-v0.7-seeds.md`;
`scripts/me6_paired_null.py`, `scripts/me6_large_n.py`,
`scripts/me6_large_n_diagnostic.py`, `scripts/b3_lesion_interaction.py`,
`scripts/run_grid.sh`, `scripts/run_grid_probe_v2.sh`.
