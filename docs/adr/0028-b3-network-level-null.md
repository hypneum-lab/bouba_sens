# ADR-0028 — B-3 network-level null: replicate lesioned networks per cell

**Status:** Proposed — draft, **not frozen, not signed off**. Nothing in
this ADR is decided; every choice below is a proposal listed again under
§Open decisions. No grid, pilot or null has been run for it.
**Date:** 2026-10-31
**Decision source:** none yet. Requested by the PI on 2026-10-31 after
the Me6 large-n diagnostic, to be frozen **before** any v0.7
ActionSense grid.
**Builds on:** ADR-0021, ADR-0022, ADR-0023, ADR-0024, ADR-0026,
ADR-0027; `docs/prereg/me6-large-n/amendment-2026-10-31-diagnostic.md`
(piste C); `docs/milestones/2026-10-31-me6-large-n-results.md`;
`docs/milestones/2026-10-27-b3-lesion-interaction-results.md`.
**Pre-registration draft:** `docs/prereg/b3-network-null/prereg-draft.md`
(the full design; this ADR records the decision to be taken).

## Context

Me6 compares P[a, b] (query a, network lesioned on b) with P[b, a]
(query b, network lesioned on a). Each column of the 5 × 5 matrix is a
**separate lesioned network** (its own lesion seed). The ADR-0022 paired
sign flip holds the five networks fixed and randomises item outcomes: it
tests "these two networks have the same per-item distribution", and has
no access to network-to-network variability.

The Me6 large-n grids (ADR-0026) showed the Gaussian Me6 median in
excess of that null at n ≥ 256 in both world-seed arms. The exploratory
diagnostic of 2026-10-31 pointed to **piste C**: the per-entry asymmetry
is a fixed property of each cell (D1 split-half r ≈ 0.84–0.90, φ − 1
growing with n), reproduced on fresh items (D4a), with no coherent
direction across trios (D3), and it appears mainly after the lesion
phase (D4b-T1 small or none); a smaller systematic query-main-effect
component (piste B) exists in the `cell` arm at phase 1 (D4b-T2). The
2026-10-27 lesion × query grid met the same phenomenon in its secondary
item-level check (each lesion cell a separate Phase-2 run).

The v0.7 ActionSense amendment (tag `bouba_sens/v0.7-actionsense-prereg`)
decides B-3 with the same ADR-0022 paired null. No ActionSense grid has
run. If network-to-network variability is not part of the null, a v0.7
B-3 "excess" cannot be read as a cross-modal asymmetry.

## Proposed decision

1. **Replicates.** For each (trio, lesioned modality), run R lesioned
   networks that differ only by lesion seed; T2 replicates of a seed
   share that seed's Phase-1 checkpoint. Replicate 0 is the cell as
   `run_grid.sh` / `run_grid_probe_v2.sh` run it today; replicates
   1 … R − 1 use the new lesion-seed rule
   `seed · 10000 + 2000 + 30 · (r − 1) + j` (j = 1 … 30, the cell's
   index within its seed), R − 1 ≤ 25. The rule is disjoint from every
   existing derived seed (§Seed namespace of the draft).
2. **Null.** Recommended: construction (a) of the draft, a
   **replicate-level transpose flip**: the statistic is Me6 on the
   replicate-mean matrix (max over pairs, median over trios), and the
   null flips the antisymmetric part of each replicate's 5 × 5 matrix as
   a whole (one sign per (trio, replicate), shared by the 10 pairs).
   The exchangeable unit is the replicate network set, which is exactly
   what piste C says the item flip misses. Construction (b)
   (between-replicate difference reference) is reported descriptively.
3. **Preconditions and controls** as in the draft: learning gate, a
   synthetic size check of the null on a grid of the real shape before
   the tag, positive and negative controls that can fail.
4. **Seeds.** Pilot 120–121, confirmatory 500–504 (checked unused in
   the repository on main `27e17ab`).
5. **Order.** Freeze (tag) this study before the v0.7 grids; a separate
   v0.7 amendment may then add the null to the ActionSense grids before
   any ActionSense datum.

## Code change it needs (described, not implemented)

- `scripts/run_grid.sh` and `scripts/run_grid_probe_v2.sh`: an opt-in
  `REPLICATES` (default empty: bit-identical to today). With
  `REPLICATES=R`, replicate 0 keeps the current cell directory and
  lesion seed; replicates r ≥ 1 write to `<cell>_r<r>/` with the rule
  above. Probe seeds stay trio-shared and identical across replicates
  (`seed · 10000 + 5000 + trio`, `+ 6000 + trio`). A guard refuses
  R − 1 > 25 and refuses to resume a replicate cell whose recorded
  lesion seed differs from the rule.
- `bouba-sens lesion`: record `replicate` and `lesion_seed_rule` in
  `metadata.json` / `config.yaml` (no behavioural change).
- `scripts/me6_network_null.py` (new): discover replicate trios, Me6 on
  the replicate-mean matrix, null (a), reference (b), synthetic size
  check, controls.
- Tests: default bit-identical; seed table unique and disjoint from the
  existing ranges for every seed set used; null (a) exact enumeration on
  a toy grid; size check on a synthetic H0 grid.

## What this ADR does not change

No metric, threshold (0.02 stays frozen), past verdict, preregistered
reading of ADR-0026 or of the 2026-10-27 grid, or B-1 / B-2 / B-3
status. It does not amend v0.7. The ADR-0022 paired null keeps its
meaning (conditional on the networks) and stays reported.

## Open decisions

R; null construction; seeds; reading thresholds, α and multiplicity;
worlds, emission and world-seed arms; probe sizes; host; whether a
v0.7 amendment follows. The list, with the options, is in
`docs/prereg/b3-network-null/prereg-draft.md` §Open decisions (PI).

## References

ADR-0021, ADR-0022, ADR-0023, ADR-0024, ADR-0026, ADR-0027;
`docs/osf/amendment-v0.7-actionsense.md`;
`scripts/me6_paired_null.py`, `scripts/me6_large_n.py`,
`scripts/me6_large_n_diagnostic.py`, `scripts/run_grid.sh`,
`scripts/run_grid_probe_v2.sh`.
