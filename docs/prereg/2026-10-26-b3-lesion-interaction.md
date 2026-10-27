# Preregistration — B-3 lesion × query interaction (straight-through emission)

**Status:** Final. Amended 2026-10-27 and frozen by the annotated tag
`prereg-b3-lesion-interaction-20261027` **before any confirmatory run**.
The draft of 2026-10-26 was on hold because, with the default emission,
the network never learns the task (ADR-0024; diagnosis
`docs/milestones/2026-10-26-code-collapse-diagnosis.md`). PI decision
2026-10-26 22:28: implement `--emission straight-through` as an option
(default unchanged), pilot it, and run this test if the pilot learns.
**Amendments of 2026-10-27** (all made before any confirmatory run):
the emission mode (§6), the precondition, now the PI's two-part
criterion fixed before the pilot (§2), the pilot outcome (Gaussian
learns, mock does not), the exact p-value floor (§5), and the code
commit and cost (§6).
**Registration level:** L1 (local, git-dated, not filed on OSF).
**Builds on:** ADR-0022, ADR-0023 (probe v2, H0 of symmetry ill-posed),
`docs/prereg/2026-10-25-b3-probe-v2.md`.

## 1. What has already been seen

Everything in ADR-0022 and ADR-0023 (grids with seeds 0–4, 100–101 and
200–204), including the exploratory decomposition showing flat
lesioned-modality means and a residual at its null on the 200–204 grids
(Gaussian p ≈ 0.55), plus the collapse diagnosis (seeds 0–1). None of
these grids enters this test; any reading of them stays exploratory.

## 2. Precondition: a network that learns

Criterion fixed by the PI and committed before the pilot
(`docs/prereg/2026-10-26-emission-pilot-criterion.md`, `b89b9f4`). It
replaces the three-part draft criterion (seeds 100–101), which was never
applied. Pilot: seeds 110–111, Phase 1, `--emission straight-through`. A
world learns if the mean over seeds of the last-20-step Phase-1 loss is
< 1.30 **and** the mean v2 vision margin is ≥ 0.10.

Outcome (`docs/prereg/2026-10-27-emission-pilot-result.md`, `5d447f7`):

- **Gaussian learns** (loss 0.993, margin 0.439).
- **Mock does not learn** (loss 1.3006, above the 1.30 threshold;
  margin 0.295). The mock world is therefore « non concluant (pas de
  réseau qui apprend) » and is not run.

## 3. Probe and outcome

Probe v2 (ADR-0023) unchanged: 256 held-out items per trio, shared
across the five cells of a trio. `P[q, m]` = held-out accuracy on query
q of the model lesioned on m. Only the 20 **off-diagonal** cells
(q ≠ m) are used: the diagonal mixes the cross-modal question with the
direct degradation of the lesioned input.

## 4. Hypothesis and statistic

- **H0** (per world): no lesion × query interaction. For every trio,
  `E[P[q, m]] = μ + a_q + b_m` on the off-diagonal cells (query and
  lesioned-modality main effects allowed).
- Per trio: least-squares additive fit of the 20 off-diagonal cells,
  then the residual matrix `R` (11 degrees of freedom).
- Per seed: `R̄_s` = mean of the six trio residual matrices of seed s
  (trios of one seed share the Phase-1 checkpoint, so the seed is the
  independent unit).
- **Statistic:** `S = Σ_{q≠m} (mean_s R̄_s[q, m])²`.

## 5. Null and inference

- **Seed-level sign flip.** Under H0 each `R̄_s` has expectation 0 and,
  as a contrast, is taken as symmetric. Each seed's `R̄_s` is flipped
  as a whole. With 10 seeds there are 2¹⁰ = 1 024 sign patterns,
  enumerated exactly; p = #{S_null ≥ S_obs} / 1 024. Because S is a
  square, the identity and the global flip give the same value, so the
  minimum attainable p is 2 / 1 024 ≈ 0.002 (< 0.01). The
  test is not affected by dependence between items or between the trios
  of a seed.
- **Blocks inherited from the B-3 v2 pilot** (block 1 Gaussian, 8 mock)
  are used only in a **secondary** item-level check: within each (trio,
  q), permute the four off-diagonal outcome series across lesioned
  modalities by blocks of b items (10 000 permutations). This tests the
  stronger H0 "no lesion effect on off-diagonal decodability" (main
  effect b_m included). It is reported and does not enter the decision.

## 6. Runs

- Code: the commit tagged `prereg-b3-lesion-interaction-20261027`
  (straight-through emission `4e1dffc`, analysis
  `scripts/b3_lesion_interaction.py` `a03142d`).
- **Gaussian world only.** Seeds **300–309** (10 seeds, 300 cells),
  `EMISSION=straight-through scripts/run_grid_probe_v2.sh` with its
  defaults (200 + 100 steps, batch 16, hard gating, probe v2, one probe
  seed per trio). Host electron-server, nice 19, four cells at a time.
- Cost estimate: 150 Gaussian cells took 54 min in the B-3 v2 grid, so
  about **1.8 h** for 300 cells, within the 6 h ceiling.
- Analysis: `scripts/b3_lesion_interaction.py --world gaussian --block 1
  --pilot-learns yes` → `reports/b3_lesion_interaction_gaussian.json`.
  The mock world is reported as « non concluant » with no grid.

## 7. Decision rule (per world, in this order)

1. Precondition (§2) failed → **non concluant**.
2. Positive control (vision margin ≥ 0.10 on the intact networks of
   seeds 300–309) failed → **non concluant**.
3. Fewer than 60 complete trios → **non concluant**.
4. p < 0.01 → **« interaction lésion × modalité présente »**.
5. p ≥ 0.05 → **« compatible avec l'absence d'interaction »**.
6. 0.01 ≤ p < 0.05 → **non concluant**.

The two worlds are read separately and p values are not adjusted across
worlds. The locked 0.02 threshold of B-3 v1 is not used.

## 8. Exploratory outputs

The secondary block-permutation check, per-cell residuals, the diagonal
cells, code usage, and anything on earlier grids.
