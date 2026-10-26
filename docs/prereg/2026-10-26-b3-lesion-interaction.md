# Preregistration (DRAFT, ON HOLD) — B-3 lesion × query interaction

**Status:** **Draft — on hold, not tagged, not run.** The code-collapse
diagnosis of 2026-10-26 (`docs/milestones/2026-10-26-code-collapse-diagnosis.md`,
branch `diag/code-collapse`) shows that with the default emission the
network never learns the task: Phase-1 loss stays at ln 4 and the
sensory states move only through optimiser drift. A lesion × query
interaction on those states would not test cross-modal reorganisation,
so this test is not run on default-trained grids. Before it can be
tagged, two things are needed: (i) the PI chooses an emission fix and
it passes its own pilot (§2); (ii) this file is amended to name the fix,
the code commit and the final cost estimate. The annotated tag
`prereg-b3-lesion-interaction-<date>` is then created **before** any
run.
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

On a pilot of the chosen emission mode (seeds 100–101, both worlds):

- Phase-1 mean training loss over the last 20 steps < 1.30 (ln 4 =
  1.386);
- full-network accuracy on held-out items above the majority baseline by
  ≥ 0.10;
- positive control of probe v2 (vision margin ≥ 0.10).

If the pilot fails, the test is not run and the outcome is reported as
« non concluant (pas de réseau qui apprend) ».

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
  enumerated exactly; p = #{S_null ≥ S_obs} / 1 024 (minimum 0.001). The
  test is not affected by dependence between items or between the trios
  of a seed.
- **Blocks inherited from the B-3 v2 pilot** (block 1 Gaussian, 8 mock)
  are used only in a **secondary** item-level check: within each (trio,
  q), permute the four off-diagonal outcome series across lesioned
  modalities by blocks of b items (10 000 permutations). This tests the
  stronger H0 "no lesion effect on off-diagonal decodability" (main
  effect b_m included). It is reported and does not enter the decision.

## 6. Runs (when unblocked)

- Seeds **300–309** (10 seeds, 300 cells per world), worlds `gaussian`
  and `studyforrest` (mock), `run_grid_probe_v2.sh` defaults plus the
  chosen emission mode. Host electron-server, nice 19, four cells at a
  time.
- Cost estimate from the B-3 v2 grids (150 cells: 54 min Gaussian, 36 min
  mock): about 1.8 h + 1.2 h = **3 h**, within the 6 h ceiling. Above
  6 h, run the Gaussian world only and report the mock world as not
  done.

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
