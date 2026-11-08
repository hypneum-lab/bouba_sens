# B-3 network-level null — pre-registration draft

**Status:** Proposed — **draft, not frozen, no tag, no PI sign-off.**
Every value below is a proposal; the PI decides the items of §16
(Decisions left to the PI) before any freeze. No grid, pilot, null,
synthetic check or power simulation has been run for this study.
**Date:** 2026-10-31, revised 2026-11-08 (change log at the end).
**ADR:** `docs/adr/0028-b3-network-level-null.md`
**Base:** main `c438286` (merged into this branch on 2026-11-08).
**Builds on:** ADR-0021, ADR-0022, ADR-0023, ADR-0024, ADR-0026,
ADR-0027; `docs/prereg/me6-large-n/amendment-2026-10-31-diagnostic.md`;
`docs/osf/erratum-2026-10-31-v0.7-seeds.md`; the two cross-reviews of
2026-10-31 and the PI input of 2026-11-07 (PR #66 comments).

## 1. Why

P[q, m] is the accuracy on query q of the network lesioned on m. In
Me6 = max over the 10 pairs of |P[a, b] − P[b, a]|, the two entries come
from **two different lesioned networks**. The ADR-0022 paired sign flip
keeps those networks fixed: its H0 is conditional on them. The Me6
large-n diagnostic (D1–D4) found the Gaussian excess at n ≥ 256 to be a
fixed per-cell asymmetry, reproduced on fresh items, with no direction
shared across trios, produced mainly by the lesion phase (piste C), plus
a smaller query-main-effect component in the `cell` arm at phase 1
(piste B, D4b-T2). A test that tells a cross-modal asymmetry from
network-to-network variability needs **several independent networks
per condition** and a null whose exchangeable unit is the network
draw, not the probe item.

### 1.1 Why no fixed threshold decides anything here

The frozen B-3 rule (median over trios of Me6 > 0.02) is a fixed
threshold on a max-then-median statistic, i.e. an extreme-value
statistic whose null law moves with n, with the item dependence of the
world and with the number of pairs. Under the ADR-0026 paired null its
firing rate was 1.0 at every n ≤ 1024 in every row; at n = 4096 it was
0.0028 (`cell`) and 0.0009 (`legacy`) on the Gaussian world but
**0.748 (`cell`) and 0.981 (`legacy`) on the mock world**
(`reports/me6_large_n_cell_mock.json`, `frac_above_threshold` 0.7477;
`reports/me6_large_n_legacy_mock.json`, 0.9814;
`docs/milestones/2026-10-31-me6-large-n-results.md:75-78`). A rule that
fires in three quarters or more of H0 draws at the largest probe size
decides nothing about asymmetry. ActionSense probes are contiguous
windows, like the mock, so the problem would be at least as large
there. **In this study every decision is an empirical p from the
same-pipeline network null of §5** (observed value, null median, null
P99, p = (k + 1) / (B + 1), B = 20 000). The 0.02 rule is computed and
reported for continuity only, together with its firing rate under the
network null (§6.1, row D4). The equivalence margin of §8 uses 0.02 as a
bound on an estimated mean with a confidence interval, which is a
different use from a threshold on an extreme-value statistic.

## 2. Estimand and hypotheses

Notation: unit u = one independent pipeline run (its own Phase-1
seed, world draw in the `cell` arm, Phase-1 checkpoint and probe items;
§3); condition c = (timing, SNR floor), 6 conditions; lesioned modality
m; P_{u,c}[q, m] the accuracy on query q of the network of unit u,
condition c, lesioned on m (20 off-diagonal entries used, the diagonal
is never used); A_{u,c}[a, b] = P_{u,c}[a, b] − P_{u,c}[b, a].

### 2.1 Residual (cycle-space) form — fixed, not optional

For any antisymmetric 5 × 5 matrix A, with modalities indexed 1 … 5:

- potential: φ_a = (1/5) Σ_b A[a, b];
- gradient part: G[a, b] = φ_a − φ_b (4 degrees of freedom);
- **cyclic part: C[a, b] = A[a, b] − (φ_a − φ_b)** (6 degrees of
  freedom).

Properties used below. (i) C is linear in P and odd under
transposition: C(Pᵀ) = −C(P). (ii) If P[q, m] = κ + u_q + v_m (query
and lesion main effects, no interaction), A[a, b] = (u_a − v_a) −
(u_b − v_b) is a pure gradient and C = 0: **query and lesion main
effects are removed exactly**. (iii) C equals the antisymmetrised
residual of the unweighted additive fit on the 20 off-diagonal cells of
the 2026-10-26 prereg (`scripts/b3_lesion_interaction.py:71`,
`interaction_residual`); the 2026-10-31 critic review checked this
numerically (difference 4·10⁻¹⁶); the implementation must test it (§11).

What this form cannot see, stated before the data: (a) any
**dominance** asymmetry (e.g. "lesioning vision costs more than
lesioning audio for every query"), which is a gradient and is removed
with the main effects: in this family of nulls the query main effect
cannot be removed without removing the lesion main effect; (b) the 5
symmetric dimensions of the 11-dimensional off-diagonal interaction.
The design tests **cyclic cross-modal asymmetry** only. A test of the
full interaction is a different study (decision 2).

### 2.2 Hypotheses

- **Estimand:** μ_c[a, b] = E_u(C_{u,c}[a, b]), the population mean of
  the cyclic part over independent units, for each condition c and pair.
- **H0 (primary):** μ_c = 0 for every c (no population cyclic
  asymmetry in this network family).
- **H1-B3net:** μ_c ≠ 0 for at least half of the conditions (the
  statistic is a median over conditions, §5, so an effect confined to
  one or two conditions is poorly detected; declared, not hidden).
- **Population, conditioning.** Units are drawn from the generator
  defined by the code at the tag, the Gaussian world law (a new world
  per unit in the `cell` arm), `STEPS_TRAIN = 200`, `STEPS_LESION = 100`,
  batch 16 and the **fixed module seeds of `bouba-sens lesion`** (mux and
  nerve seed 0, SensoryWML seeds 1–5, `src/bouba_sens/cli.py:693-708`),
  which are shared by every T1 network of every unit. The reading is
  conditional on those fixed parts (fixed on review, §16.2 F12).
- B-3 itself (Me6 > 0.02, frozen texts) is not re-decided here. A
  confirmation of H1-B3net is a new, separately named result.

## 3. Design: independent units

**Unit = one grid seed Q_g run through the existing scripts.** Each
unit has its own Phase-1 training (seed Q_g, world `GaussianWorld(seed
= Q_g)` in the `cell` arm, `cli.py:453-454`), its own T2 checkpoint, 30
lesion cells (5 modalities × 2 timings × 3 SNR floors) with lesion seed
Q_g · 10000 + count (unchanged rule), and its own probe items (v2:
Q_g · 10000 + 5000 + trio; per-size v1: Q_g · 10000 + 6000 + trio). The
five columns of a condition share the unit's probe items, as today.

This replaces the earlier "R replicates per cell sharing the trio's
probe items and the seed's T2 checkpoint". The reviews showed that any
component shared by a subset of the networks entering the mean (probe
items, T2 checkpoint, world) is kept whole in the observed mean but
averaged down in a per-replicate sign-flip null, so the test rejects
whenever that component is of the order of the network SD / √R
(critic simulation: rejection 0.71 at α = 0.01 for a shared component
of 0.01 with network SD 0.03, R = 5, K = 30). With independent units
nothing random is shared between units except the declared generator
(§2.2), so the sign flip of §5 has a justified exchangeable unit.

Within a unit, the 6 conditions share the world, the Phase-1 checkpoint
(T2) and, through the inherited `count` arithmetic, overlapping
training batches. This dependence is **kept**, not removed: the sign of
§5 is one per unit, applied to all 6 conditions and 10 pairs of that
unit.

### 3.1 Seed streams: full inventory and rule

Streams consumed by one job (code at `c438286`):

| job | integers consumed | source |
|---|---|---|
| Phase 1, seed Q | world batches Q … Q + 199; module seeds Q … Q + 5; `_seed_everything(Q)`; intact v2 probe Q · 10000 + 5099, pool + 5100 | `loop.py:189`, `cli.py:453-475`, `run_grid_probe_v2.sh:93` |
| lesion, seed L = Q · 10000 + count | fresh batches L … L + 99; initial replay L − 1 … L − 16; FIFO replay L − 101 … L − 200; Me3 probe L − 99 999; v1 historical probe L + 777; `_seed_everything(L)` | `loop.py:299,310,320,351`, `cli.py:679,738` |
| lesion, probes | v2 test Q · 10000 + 5000 + trio, pool + 1; per-size Q · 10000 + 6000 + trio (inside `fork_rng`) | `run_grid_probe_v2.sh:111-113`, `cli.py:134-160` |
| no-lesion control (§7) | the streams of the floor cell it is matched to | — |

Consequences, which the earlier draft did not inventory:

- Consecutive Phase-1 seeds share 199 of 200 Phase-1 batch seeds. The
  proposed seeds 500–504 also reuse the Phase-1 batch seeds of the seen
  grids 400–404 (seed 400 consumed 400 … 599), and 120 those of the
  pilot 110 (110 … 309). In the Gaussian world equal batch seeds give
  the same latent and labels (`world/gaussian.py:53-54`). Seeds 120–121
  and 500–504 remain **free as run identifiers** (both reviews checked
  this; still true on `c438286`), but they are **not independent
  units**: their Phase-1 batch seeds overlap those of each other and of
  seen grids, so in the Gaussian world their training data, and hence
  their networks, are correlated (`world/gaussian.py:53-54`). They are
  therefore excluded from the pilot and from the confirmatory set of
  the recommended design (they would be used only under option B of
  decision 1).
- `np.random.seed` accepts only integers < 2³² (`cli.py:53`), so
  Q · 10000 + 9999 < 4 294 967 296, i.e. Q ≤ 429 495.
- The Me3 probe L − 99 999 falls in the namespace of Q − 10; the replay
  FIFO goes down to L − 200, i.e. into the top of the namespace of Q − 1.

**Unit seed rule.** Q_g = 330 000 + 200 · g, g = 0 … 194 (Q from
330 000 to 368 800). Pilot units g = 0 … 5; confirmatory units
g = 20 … 20 + G − 1; g = 6 … 19 are never used (no replacement units).
Checks, by arithmetic on the table above:

- Phase-1 ranges [Q, Q + 199] are disjoint between units (spacing 200)
  and lie in 330 000 … 368 999, i.e. in the namespaces of 33–36, which
  no run seed uses; they stay below 369 801, the lowest replay seed of
  run seed 37 (37 · 10000 + 1 − 200).
- Lesion namespaces Q · 10000 − 200 … Q · 10000 + 9999 lie in
  3.3·10⁹ … 3.689·10⁹ < 2³², above every historical stream (run seeds
  ≤ 999, derived seeds < 1.1·10⁷; resampling seeds 25 102 026 and
  31 102 026; D4 5 106 000–5 106 045) and disjoint between units
  (spacing 2·10⁶).
- Me3 probes land in the namespace of seed Q − 10 (e.g. 329 990 for
  g = 0), which is not a unit seed and not a historical run seed (run
  seeds ≤ 999); replay bottoms land in the namespace of Q − 1, likewise
  neither a unit seed nor a run seed.
- Within one `run_grid_probe_v2.sh` call `count` runs from 1 to
  30 · |SEEDS|. The highest per-cell offset is the historical v1 probe
  count + 777 (the fresh batches reach only count + 99), and it must
  stay below the v2 probe offsets 5000: **30 · |SEEDS| + 777 < 5000**,
  i.e. |SEEDS| ≤ 140 per call (guard). Recommended: one
  call per arm with the G unit seeds in ascending order, written in the
  Stage-B manifest, so every lesion seed is fixed in advance.

Searched on `c438286`: every `SEEDS` list in the scripts, seed lists in
the docs, the N12 list (3, 11, 19, 29, 37, 41, 47, 59, 67, 79,
`scripts/analyse_n12_subgroup_verdict.py:20`), and any number in
330 000–369 999: no hit except a hash string in `tests/golden`. The
search is redone on each tag commit, and the run records on the hosts
are checked too (erratum 2026-10-31).

## 4. Statistic

For each unit u and condition c, C_{u,c} = cyclic part (§2.1) of
A_{u,c}. Mean over units: C̄_c = (1/G) Σ_u C_{u,c}.

**T = median over the 6 conditions c of max over the 10 pairs {a, b}
of |C̄_c[a, b]|.**

The mean over units comes **before** the max (the mean of per-unit
maxima answers another question and is flip-invariant). This keeps the
Me6 aggregation (max over pairs, then median) on a new object; it is
not the frozen Me6.

## 5. Null: unit-level sign flip

One sign ε_u ∈ {−1, +1} per unit, uniform and independent, applied to
**all** 6 conditions and 10 pairs of that unit:
C̄*_c = (1/G) Σ_u ε_u C_{u,c}; T* with the same aggregation; B = 20 000
Monte Carlo draws, p = (#{T* ≥ T} + 1) / (B + 1); reported: T, null
median, null P99, p. With G ≥ 20 there are ≥ 2¹⁹ distinct patterns
modulo the global sign, so the p floor is 1/(B + 1) ≈ 5·10⁻⁵.

**Assumption, stated and justified.** Validity needs the joint law of
the unit tuples (C_{u,c})_c to be invariant under the signs actually
used (Codex review, §1.1), not only μ_c = 0. Two conditions:

1. **Independence of units.** Provided by construction (§3: distinct
   Phase-1 seed, world and probe items; disjoint streams, §3.1), except
   the declared generator (§2.2). It is checked on real data by NC2
   (§7), which can fail if units are not independent and identically
   distributed.
2. **Central symmetry** of each unit tuple under H0. Sufficient for
   exactness (finite-sample level control, not a rejection rate exactly
   equal to α) [R1]. It holds exactly if the unit noise is Gaussian
   (any covariance). Without it, for independent units with mean zero,
   the sign-flip distribution of √G · C̄ and the sampling distribution
   of √G · C̄ converge to the same Gaussian law, and T is a continuous
   function of C̄, so the test is asymptotically valid [R8]; at
   finite G this is checked by the size check (§7, generators S1–S2)
   and by NC1 on real networks.

The Codex counter-example (one common random sign for every trio and
replicate) is excluded by condition 1: no random component is common
to two units.

**Reference (b), descriptive only.** Per-unit Me6 (frozen statistic,
one network per column) beside a reference made of differences of
disjoint unit pairs, (A_u − A_u′) / √2. Its √2 scaling matches the
variance, not the tails (Codex: a ±c law gives 0 or ±√2·c), so it never
decides; it is reported as a scale of network-to-network spread. The
random sign of the earlier draft is dropped (Me6 is invariant to it).

## 6. Rows, predictions, gate

### 6.1 Rows

| id | role | emission | world / arm | probe | form | null |
|---|---|---|---|---|---|---|
| **P1** | **primary, single confirmatory row** | straight-through | Gaussian / `cell` | v2, 256 test items (§16.2 F3) | cyclic | unit sign flip (§5) |
| K1 | calibration row, gate-exempt | argmax-bridge | Gaussian / `cell` | v1 per-size, n = 4096 | cyclic | §5 |
| D1 | descriptive | as P1 | as P1 | as P1 | **raw** A | §5 |
| D2 | descriptive | as P1 and K1 | as P1 and K1 | v1 per-size n = 64, 256, 1024, 4096 and v2 | cyclic | §5 |
| D3 | descriptive | as P1 and K1 | as P1 and K1 | as the row | per-unit Me6 | ADR-0022 item flip, per unit |
| D4 | descriptive | as P1 and K1 | as P1 and K1 | as the row | frozen 0.02 rule on per-unit Me6 medians | firing rate under §5 |
| D5 | descriptive | as P1 | as P1 | as P1 | per-unit Me6 | reference (b) |

Rows `legacy`, mock world or other emissions run only if the PI says so
(decisions 9, 10) and are then descriptive. No descriptive row carries a
verdict; their p values are unadjusted and labelled as such.
Descriptive p values are never reported as confirmatory in any
abstract, conclusion or summary; they appear only in the per-row
table, labelled 'descriptive, unadjusted'.

### 6.2 Predictions (written before any datum of this study)

Seen data used to write them: the 2026-10-27 straight-through Gaussian
grid (seeds 300–309, v2) and the D1–D4 outputs (§13).

- **P1.** The 2026-10-27 grid has seed-mean interaction residuals below
  0.01 in every cell (`docs/milestones/2026-10-27-b3-lesion-interaction-results.md:39-42`),
  so |C| ≤ 2 · max|residual| < 0.02 in mean. **Prediction: p ≥ 0.01
  (no rejection), every |C̄_c[a, b]| < 0.02.** The expected reading is
  « non concluant »: « nul TOST » is not expected to be reachable
  (§8), and is not available at all if no equivalence test is kept
  (decision 5). The expected reading is restated in the Stage-B
  manifest once G is fixed.
- **K1.** Piste C (fixed per-network asymmetry with no direction across
  trios; D3 p = 0.28 `cell`) predicts **p ≥ 0.05 under the network
  null** while D3 (item flip, per unit) **rejects in most units at
  n = 4096**. A rejection of K1 would mean that a systematic component
  (piste B) survives the removal of main effects.
- **D1 (raw form).** Query main effects of 0.661 / 0.676 / 0.612 /
  0.281 / 0.333 (2026-10-27 grid) give A[vision, gravity] ≈ 0.39 in
  every unit: **p at the floor (5·10⁻⁵)**. This is why the raw form
  cannot be primary.
- **D4.** The 0.02 rule fires in nearly every unit on P1 data (main
  effects) and with high probability under the network null at
  n ≤ 1024; reported, it decides nothing (§1.1).
- **NC1, NC2:** p ≥ 0.01. **PC:** p < 0.01 if G meets the power rule.
- `legacy`, mock, straight-through `legacy`: **no prediction** (no
  basis), descriptive if run.

### 6.3 Learning gate (P1 only)

Evaluated on the confirmatory units **before** T is computed; all three
must hold:

1. Phase-1 loss, mean of the last 20 steps < 1.30 in ≥ 80 % of the
   units (2026-10-26 criterion, per unit);
2. mean v2 vision margin of the intact Phase-1 networks ≥ 0.10;
3. mean v1 off-diagonal **accuracy** (not a margin) of the lesion cells
   at n = 4096 ≥ 0.30, i.e. 5 percentage points above chance (0.25).

Gate failed → P1 is « non concluant (pas de réseau qui apprend) », no
statistic is read. No unit is excluded for failing (no selection).

**K1 and the gate.** K1 is a **calibration row**: argmax-bridge does
not learn (v1 accuracy 0.238–0.269, ADR-0024), it would fail the gate by
design, and it is exactly where the excess of ADR-0026 was found.
K1 is therefore **exempt from the gate**, carries no B-3 reading and no
verdict class, and is reported as "calibration of the network null on
the ADR-0026 phenomenon", with the regime qualification of ADR-0024.

## 7. Controls (each can fail) and size check

**Size check (synthetic, Stage B, before the confirmatory tag).**
1 000 synthetic grids per generator, B = 2 000 each, G = the registered
value, the exact pipeline of §4–§5 (cyclic projection included).
Generators, parametrised by the pilot (§10), fixed in Stage A:

- **S0** centred Gaussian unit tuples with the pilot covariance (exact
  case; implementation check).
- **S1** bounded accuracies: per unit, per column, per query binomial
  outcomes at n items around pilot means, with per-network column and
  query offsets, heteroscedastic by condition, clipped to [0, 1].
- **S2** S1 plus sparse cells (near-constant class, isolated entries far
  from 1/4, as seen in the diagnostic) and skewed offsets (non-symmetric
  law of mean zero). Their parameters are taken from the pilot (§10):
  the proportion of near-constant cells and the interquartile range of
  each entry across units.
- **S3** S1 plus query and lesion main effects of the 2026-10-27
  magnitudes (must be removed by the cyclic form).
- **S4 (sensitivity, not a pass criterion)** S1 plus a component shared
  by subsets of units (pairs, groups of 5), amplitude 0.5, 1, 2 × the
  pilot item-noise SD: shows the inflation that shared components cause
  and that the design avoids; also the raw form on S3 (expected to
  reject).

Pass: rejection rate at α = 0.01 ≤ 0.016 on **each** of S0–S3
(α + 2 binomial SE, SE 0.00315 at 1 000 grids). A rate < 0.004 is
reported as conservatism and its power cost enters §10. Fail on any of
S0–S3 → the construction is not frozen; no rerun with another
generator seed after a failure. The check is a filter against gross
inflation, not a proof of exactness.

**Negative control NC1 — no-lesion networks (real data, can fail).**
For each confirmatory unit and timing, the five "columns" are networks
trained in Phase 2 **without lesion** (`lesion_phase(lesion=None)`,
`loop.py:250`), each stream-matched to the floor cell of its column
(same lesion seed, same probe seed as the (timing, floor) trio): 10 jobs
per unit. Without a lesion the column label has no effect, so the
columns are exchangeable given the unit and μ = 0 holds **in mean by
construction**; the statistic is not identically zero and its null is
not built from a symmetrised copy of it, so the control fails if the
test is miscalibrated on real networks (skew, sparsity, the projection)
or if units are dependent. Same pipeline with 2 conditions (T1, T2;
median of 2 = mean). Required p ≥ 0.01; p < 0.01 → P1 is « non
concluant (contrôle négatif échoué) ». A valid control fails by chance
with probability 0.01; declared. NC1 has lower power than P1 (2 vs 6
conditions); a failure is informative, a pass is not a proof that the
P1 test is well calibrated at all 6 conditions.

**Negative control NC2 — disjoint unit pairs (real lesion data, premise
check).** Units in registered order are paired (u₂ᵢ, u₂ᵢ₊₁) (G even);
Dᵢ,c[q, m] = P_{u₂ᵢ,c}[q, m] − P_{u₂ᵢ₊₁,c}[q, m] on the 20 off-diagonal
entries; statistic median over c of max over entries of |meanᵢ Dᵢ,c|;
null: one sign per pair (swap of the two units). Pairs are disjoint, so
the mean is not identically zero (the earlier cyclic derangement gave
Σ_r (P_r − P_σ(r)) ≡ 0 and p = 1 by construction; it is removed). Its
H0 holds if units are independent and identically distributed; it fails
on drift with the unit index or run order, a stream collision or a
pipeline bug. Required p ≥ 0.01; p < 0.01 → P1 « non concluant
(contrôle négatif échoué) ».

**Positive control PC — planted cyclic effect (H0 data, can fail).**
On the NC1 grid (real networks, μ = 0 in mean), plant in every unit and
both timings a **pure 3-cycle** audio → vision → tactile → audio of
size δ_pc (C[audio, vision] = C[vision, tactile] = C[tactile, audio] =
δ_pc, zero potential, so the cyclic form keeps all of it), by
deterministic item flips (k items per entry; the choice of items does
not change the accuracies, so one planting is one realisation, not 50;
k = round(δ_pc · n / 2) per entry, the realised δ is reported: at
n = 256 and δ_pc = 0.02, k = 3 and the realised δ is 6/256 = 0.0234; at
n = 4096, k = 41 and the realised δ is 82/4096 = 0.0200).
Required p < 0.01, informative only if NC1 passed (a grid that already
rejects passes trivially). Binding power requirement: the synthetic
power of §10 on S1 at the registered G. PC failed → a non-rejection of
P1 is « non concluant (puissance) »; a rejection is still read.

The earlier "negative control 1" (planted symmetric noise on real grids)
is dropped: on data under H0 it can fail only through a bug, and on
data that already reject it forbids any reading.

## 8. Reading rules (9-class vocabulary only)

Order: completeness → gate → NC1, NC2 → P1 → PC → equivalence.

| condition | reading of P1 |
|---|---|
| grid incomplete after the rerun rule (§12) | « non concluant (grille incomplète) » |
| gate failed | « non concluant (pas de réseau qui apprend) » |
| NC1 or NC2 failed | « non concluant (contrôle négatif échoué) » |
| p < α (α = 0.01, decision 4) | « confirmé » (H1-B3net), with C̄ and its simultaneous CI |
| only if decision 5 keeps a TOST: p ≥ α, PC passed, every simultaneous TOST bound inside ±Δ_eq | « nul TOST » |
| p ≥ α otherwise | « non concluant » (or « non concluant (puissance) » if PC failed) |
| not run | « non testé » / « en attente » |

- **Equivalence (TOST).** For each of the 60 (condition, pair) entries,
  two one-sided tests at 0.05 with Bonferroni over the 60 entries,
  unit-level t with G − 1 degrees of freedom: « nul TOST » only if
  every |C̄_c[a, b]| + t_{1 − 0.05/60, G − 1} · s_c[a, b] / √G < Δ_eq.
  If an equivalence test is kept (decision 5; recommended: none),
  Δ_eq = 0.02 (the B-3 scientific threshold carried over as a margin on
  a mean). At τ_c ≥ 0.03 and G ≤ 50, TOST is not expected to be
  reachable; the design can confirm H1-B3net but cannot establish
  absence at this margin.
- **p ≥ α is never read as absence**, equivalence or "compatible".
  Absence is claimed only through « nul TOST », and not at all if no
  equivalence test is kept (decision 5). « réfuté » is not
  reachable in this design.
- **Test family and multiplicity.** Confirmatory family F = {P1}, one
  test, α = 0.01, no correction needed. The max inside T handles the 10
  pairs, the median the 6 conditions; nothing else (probe sizes, arms,
  forms, rows) is in F. Controls are gates with their own fixed α and
  consequence, not members of F. Any row moved into F by the PI before
  the Stage-B tag makes F a Holm family at FWER 0.01.
- Wording outside the classes: numbers only ("T = …; nul réseau
  médiane …, P99 …, p = …").

## 9. Worlds, arms, probes

- **World:** Gaussian only (the mock failed the learning gate under
  straight-through on 2026-10-27; decision 10).
- **Arm:** `cell` only (ADR-0027 intended design; in `legacy` every
  lesion cell reads world 0 while Phase 1 trains on world Q, the
  historical mismatch; decision 9).
- **Probes:** every lesion cell writes v2 (256 test, 512 pool) and the
  v1 per-size files n ∈ {64, 256, 1024, 4096}. P1 uses v2 (§16.2 F3);
  the historical `per_query_successes.json` (L + 777, a different item
  set per column) is never used.

## 10. Pilot, power and the choice of G

The earlier R = 5 had no justification. It is replaced by G, the number
of independent units (old R × 5 seeds), chosen by a rule fixed in
Stage A.

**Pilot (Stage A, after its tag).** Units g = 0 … 5 (6 units), P1
settings, 30 lesion cells + 10 no-lesion jobs per unit. It yields: the
gate; τ_c = between-unit SD of the cyclic entries per condition (v2 and
each v1 size); the item-noise share (even/odd split of the v1 n = 4096
items within units); sparsity and skew descriptors for S1–S2. **No T,
NC or PC statistic is computed on the pilot.** Pilot units never enter
the confirmatory set.

**Scaling.** Under independence, SE(C̄_c[a, b]) = τ_c / √G. The cyclic
projection keeps 6 of 10 dimensions; for exchangeable entries the
per-entry variance of C is 6/10 of that of A. At fixed G a larger n
reduces only the item part of τ_c, not the network part.

**Optimistic bound (single pre-specified entry, z-test, two-sided
α = 0.01, power 0.80):** δ ≈ (2.576 + 0.842) · τ_c / √G = 3.42 · τ_c / √G.

| τ_c | G = 20 | G = 30 | G = 40 | G = 50 |
|---|---:|---:|---:|---:|
| 0.02 | 0.015 | 0.012 | 0.011 | 0.010 |
| 0.03 | 0.023 | 0.019 | 0.016 | 0.015 |
| 0.05 | 0.038 | 0.031 | 0.027 | 0.024 |

This is arithmetic, not a simulation: the max over pairs and the median
over conditions make the real detectable δ **larger**. No straight-through
τ_c exists yet. The only available scale is the Codex review's
approximation of the raw-A RMS at n = 4096 in argmax-bridge, 0.039
(`legacy`) to 0.052 (`cell`), which mixes item, network and cell
components and is not a between-unit SD; the critic review's
simulations of the earlier per-trio design gave power 0.25 at δ = 0.02
(network SD 0.03, R = 5). With τ_c ≈ 0.03, δ = 0.02 is decidable only
for G ≳ 30 even in the optimistic bound.

**Rule (proposal, decision 7).** By simulation on S1 parametrised with
the **upper 80 % confidence bound** of the pilot τ_c (χ² with 5 df),
planted 3-cycle of size δ_pc in all 6 conditions (sensitivity variant:
in 3 of 6; if the effect is confined to T1 or T2, the median over 6
conditions may miss it; the 3/6 variant detects this pattern), 1 000
grids per G: G = smallest value in {20, 30, 40, 50} with power
≥ 0.80 at α = 0.01 (binomial SE ≤ 0.013 at 0.80). If no value reaches
0.80 within the compute ceiling, the decidable δ at G = 50 is reported
and the PI decides go / no-go before the Stage-B tag. If the pilot
(G = 6) shows strongly skewed unit distributions (|skewness| > 2 on any
entry), a permutation or bootstrap variant of the unit-level test is
substituted and declared before the Stage B tag. The pilot record,
the generator, G, the SEEDS list and the expected readings are committed
in the Stage-B manifest before the first confirmatory job.

## 11. Code change (described; not implemented in this PR)

- **No replicate loop is needed:** units are ordinary grid seeds, run
  by the existing `run_grid_probe_v2.sh` with the lesion and probe-seed
  rules unchanged.
- `run_grid_probe_v2.sh` (and `run_grid.sh` for v0.7): guard
  |SEEDS| ≤ 133; refuse to resume a cell whose recorded lesion seed or
  probe seed differs from the rule (the parallel script skips any
  existing `eval_report.json` without checking, `run_grid_probe_v2.sh:103`);
  opt-in `NO_LESION=1` mode: per (unit, timing, modality) one job with
  the floor cell's lesion seed and the (timing, floor) probe seeds,
  directory `<cell>_nolesion`, which the existing anchored regexes
  ignore (`me6_paired_null.py:81-83`, `aggregate_grid.py:30-33`).
- `bouba-sens lesion --no-lesion` (calls `lesion_phase(lesion=None)`),
  recorded in `metadata.json` only when set, so default outputs stay
  bit-identical.
- `scripts/me6_network_null.py` (new): unit discovery (a unit is
  complete only with 30 + 10 jobs, each with `eval_report.json` and the
  expected seeds), cyclic projection, T, null (§5), NC1, NC2, PC
  planting, TOST bounds, rows D1–D5, JSON output with T, null median,
  P99, p, B, seed.
- `scripts/me6_network_null_synth.py` (new): generators S0–S4, size
  check and power (§7, §10).
- Tests: default bit-identical; C = antisymmetrised
  `interaction_residual`; C(Pᵀ) = −C(P); C(κ + u_q + v_m) = 0; exact
  enumeration vs Monte Carlo on a toy G = 6; NC2 not identically zero
  on random input; seed table: every stream of §3.1 for every unit
  disjoint from every other unit and from the historical ranges, all
  `_seed_everything` arguments < 2³².

## 12. Completeness, reruns, reproducibility

- Verdict only on a complete grid: G units × (1 Phase-1 + 30 lesion +
  10 no-lesion) jobs. A failed job is rerun with the same seed, at most
  twice. No unit or seed is replaced; g = 6 … 19 are never used. Any job
  still missing → « non concluant (grille incomplète) ».
- Non-finite losses are kept as recorded; a v2 fitting failure is a job
  failure.
- Recorded at Stage B: code SHA, nerve-wml version, `uv.lock` hash,
  exact commands, SEEDS order, host. Resampling seeds (fixed, §16.2
  F17): 8112026 for the analysis, 8112027 for the synthetic checks; both
  absent from the repository on `c438286`.
- **Two-stage freeze (fixed, §16.2 F15).** Stage A tag (this
  protocol, pilot plan, S0–S4, power rule) before the pilot; Stage B
  tag (pilot record, generator parameters, G, SEEDS, manifest) before
  the first confirmatory job.

## 13. Forking paths and data already seen (declared)

Seen, usable only to motivate this study, never as its data: the Me6
large-n grids, seeds **400–404** (both arms, Gaussian and mock), their
paired nulls and power files; the mock pilots **100–101** (ADR-0026;
also ADR-0022 / ADR-0023 grids); D1–D4 (`reports/me6_diag_*.json`,
logs), including the reprobe seeds 5 106 000–5 106 045; the 2026-10-27
lesion × query grid, seeds **300–309**, and its pilot 110–111 (its
residuals inform the P1 prediction); the ADR-0021 to ADR-0024 grids
(0–4, 200–204) and the smoke seed 999; the N12 grids (seeds listed in
§3.1); the exploratory v2 decomposition and options reports
(`reports/b3_probe_v2_decomposition_exploratory.json`,
`reports/b3_probe_v2_options_exploratory.json`); the emission diagnostic
and pilots (`reports/diag_code_collapse_*.json`,
`reports/emission_pilot_*.json`); the ActionSense S00 extraction and
pilot used for v0.7; the v0.7 reading note and brainstorm
(`docs/prereg/actionsense-v0.7/`). Whether the v0.7 block-length pilot
(ActionSense, seeds 100–101) has run is **not verifiable from this
clone**; it is checked on the hosts at Stage A.

Revisions that preceded this text: the draft of 2026-10-31, two
independent reviews (REVISE / BLOQUANT), the PI input of 2026-11-07
(§1.1), this revision. Values taken from earlier data: gate thresholds
(2026-10-26 criterion), Δ_eq and δ_pc (B-3 threshold, not data), the
main-effect magnitudes of S3 (2026-10-27 grid), the pilot τ_c for G.
Units g = 0 … 194 have produced no data.

## 14. Compute estimate (not run)

Per unit: 1 Phase-1 + 30 lesion + 10 no-lesion = 41 jobs (31 without
no-lesion control). Per-job slot time from electron-server records:
≈ 85 s idle (ADR-0023), ≈ 112 s under load 10–13 (2026-10-27 grid),
≈ 210 s under heavy load with the per-size probes (ADR-0026 smoke);
Phase-1 jobs counted as one slot each, as in the earlier estimate.

| set | jobs | wall, PAR 4: idle / loaded / heavy |
|---|---:|---|
| pilot, 6 units × 41 | 246 | 1.5 / 1.9 / 3.6 h |
| P1 confirmatory, G = 20 × 41 | 820 | 4.8 / 6.4 / 12.0 h |
| P1 confirmatory, G = 30 × 41 | 1 230 | 7.3 / 9.6 / 17.9 h |
| P1 confirmatory, G = 50 × 41 | 2 050 | 12.1 / 15.9 / 29.9 h |
| K1, G = 30 × 31 (no pilot, no no-lesion control) | 930 | 5.5 / 7.2 / 13.6 h |
| **proposed minimum:** pilot + P1 (G = 30) + K1 | 2 406 | 14.2 / 18.7 / 35.1 h |

The earlier design (R = 5, two emissions) was 2 114 jobs. Analysis on
5 × 5 matrices takes seconds per null; size check and power ≈ 1–2 h of
CPU. **Host:** electron-server at nice 19 (PAR 4, `OMP_NUM_THREADS=2`)
or macM1, one host for all rows, free state checked at Stage B. Nothing
is run by this PR.

## 15. Link to v0.7 ActionSense

A separate v0.7 amendment, merged and tagged before any ActionSense
grid, would be needed. It must name the frozen clauses it replaces in
`docs/osf/amendment-v0.7-actionsense.md`:

- §4.1 (grids and seeds): the network null needs G independent units
  per grid; the frozen seeds 300–304 are consecutive and share 199 of
  200 Phase-1 batch seeds (`loop.py:189`), so they cannot be units; new
  unit seeds are needed (the "never used" claim is already corrected by
  `docs/osf/erratum-2026-10-31-v0.7-seeds.md`);
- §4.2 (exact commands): unit SEEDS, `NO_LESION=1`, the new script;
- §4.3 (B-3 bullet): the ADR-0022 paired null as the reading that
  decides, and the 0.02 rule;
- §5.1 (completeness 150 / 150, 30 trios);
- §5.2 (median rule, "no multiple-comparison correction"; its textual
  "≥" contradicts the strict ">" of §4.3 and §8 item 5, to be declared);
- §5.3 (reading "distinguishable from the paired null" at 0.05; H0 of
  symmetric expected performances, i.e. the raw form);
- §5.5 (per-trio Me6 descriptive reports);
- §7 (deviations policy: the amendment itself is the dated deviation)
  and the listing in ADR-0025;
- §8 item 3 (block-length pilot, seeds 100–101, confirmatory seeds);
- §9 ("no change to the shape of one grid … 150 cells", thresholds).

The frozen v0.7 grids keep their registered analysis (fixed rule and
ADR-0022 paired null on the 300–304 grid), reported as registered and
marked by the amendment as not deciding asymmetry (§1.1). On
ActionSense the world is fixed (real recordings), so H0 is conditional
on it; the v2 items are contiguous windows, which the unit flip does
not need to block but which make each unit's item noise larger than
binomial; per-unit windows are required; class imbalance (majority
0.4306) makes main effects likely, so the cyclic form is required. The
amendment states whether the network null decides (recommended) or is
reported beside, and its cost (G × 30 cells per grid, on two or four
grids).

## 16. Decisions left to the PI

### 16.1 Open (11)

Original numbers are kept; decisions 3, 8, 11, 12, 15 and 17 were fixed
on review (§16.2). Recommended option first. None of the following is
decided.

1. **Unit design.** (A) independent units Q_g = 330 000 + 200 · g, own
   Phase 1, world and probe items (recommended); (B) base seeds 500–504
   with R replicates sharing the T2 checkpoint and world, flip per
   replicate, H0 conditional on the checkpoint, S4 becomes a pass
   criterion and Phase-1 batch overlap with 400–404 is declared;
   (C) as B with the seed as flip unit (16 patterns at 5 seeds: not
   decidable at 0.01; would need ≥ 10 base seeds).
2. **Estimand.** (A) cyclic part (recommended); (B) full off-diagonal
   interaction (11 dimensions, statistic to define); (C) raw
   antisymmetric part, methodological only, never primary.
4. **α of the family F = {P1}.** (A) 0.01 (recommended); (B) 0.05;
   (C) F = {P1, P1-`legacy`}, Holm at FWER 0.01.
5. **Equivalence test.** (C) no equivalence test: TOST is not expected
   to be reachable at τ_c ≥ 0.03 and G ≤ 50 (§8), so no absence claim
   is made and a non-rejection stays « non concluant » (recommended);
   (A) Δ_eq = 0.02 on all 60 entries; (B) Δ_eq = 0.01.
6. **Positive control.** (A) δ_pc = 0.02, power 0.80 (recommended);
   (B) δ_pc = 0.03, power 0.80; (C) δ_pc = 0.02, power 0.90.
7. **G rule and ceiling.** (A) smallest G in {20, 30, 40, 50} with power
   ≥ 0.80 at δ_pc, ceiling 50, otherwise PI go / no-go on the reported
   decidable δ (recommended); (B) fixed G = 30 whatever the pilot;
   (C) ceiling 30.
9. **World-seed arm.** (A) `cell` only (recommended); (B) `cell` +
   `legacy` descriptive.
10. **Mock world.** (A) not in this study (recommended); (B) mock pilot,
    then a mock P1-like row only if its gate passes (needs a block-free
    design check on contiguous windows).
13. **No-lesion control.** (A) 10 stream-matched jobs per unit
    (recommended); (B) 30 per unit, one per lesion cell, which also
    allows a lesion-minus-control contrast; (C) none (no NC1, not
    recommended).
14. **Host and ceiling.** (A) electron-server, nice 19, PAR 4, ceiling
    40 h wall (recommended); (B) macM1.
16. **v0.7.** (A) after the Stage-B results, a v0.7 amendment in which
    the network null decides and the frozen reading is reported as
    registered (recommended); (B) network null reported beside, the
    frozen reading still decides; (C) no v0.7 amendment.

### 16.2 Fixed on review (2026-11-08)

These had an objective answer and are part of the protocol; they are
no longer PI decisions.

- **F3 Primary probe: v2, 256 test items.** It is the probe of the
  2026-10-27 grid, of v0.7 and of the diagnostic; another probe would
  invalidate the predictions of §6.2.
- **F8 Calibration row K1:** runs, gate-exempt, no pilot, no no-lesion
  control, descriptive (§6.3).
- **F11 Learning gate:** the three criteria of §6.3, no exclusion of
  units.
- **F12 Fixed module seeds of `lesion` (T1):** kept; the reading is
  conditional on them, declared in §2.2.
- **F15 Freeze:** two annotated tags, Stage A before the pilot, Stage B
  before the confirmatory grid (§12).
- **F17 Resampling seeds:** 8112026 (analysis), 8112027 (synthetic),
  checked absent from the repository on `c438286`.

## References

ADR-0021, ADR-0022, ADR-0023, ADR-0024, ADR-0026, ADR-0027, ADR-0028;
`docs/prereg/me6-large-n/amendment-2026-10-31-diagnostic.md`;
`docs/milestones/2026-10-31-me6-large-n-results.md`;
`docs/milestones/2026-10-27-b3-lesion-interaction-results.md`;
`docs/prereg/2026-10-26-b3-lesion-interaction.md`;
`docs/osf/amendment-v0.7-actionsense.md`;
`docs/osf/erratum-2026-10-31-v0.7-seeds.md`;
`scripts/me6_paired_null.py`, `scripts/me6_large_n.py`,
`scripts/me6_large_n_diagnostic.py`, `scripts/b3_lesion_interaction.py`,
`scripts/run_grid.sh`, `scripts/run_grid_probe_v2.sh`;
`src/bouba_sens/loop.py`, `src/bouba_sens/cli.py`.

- [R1] Koning, N. W. & Hemerik, J. (2024). More efficient exact group
  invariance testing: using a representative subgroup. *Biometrika*
  111(2), 441–458. doi:10.1093/biomet/asad050.
- [R8] Andreella, A. & Finos, L. (2026). Sign-flipping score tests for
  fixed effects in multivariate clustered data. arXiv:2604.27907v2
  (preprint).

Both were checked by the 2026-10-31 Codex review; neither proves the
validity of this statistic, which rests on §5 and on the checks of §7.

## Change log

- **Revision 2026-11-08** (after the reviews of 2026-10-31 and the PI
  input of 2026-11-07). Cyclic (residual) form fixed, with its formula
  and what it cannot test (B1). Replicates sharing probe items and the
  T2 checkpoint replaced by independent units with their own Phase 1,
  world and items; the sign flip is per unit across all conditions; the
  joint-invariance assumption is stated and justified (B2). Negative
  control 2 (derangement, p = 1 by construction) replaced by disjoint
  unit pairs; negative control 1 replaced by a no-lesion control on real
  networks; size check with defined generators including skew,
  sparsity, main effects and shared components; positive control as a
  pure 3-cycle on H0 data at δ_pc = 0.02 tied to the power rule (B3).
  Single primary row, family F = {P1}, per-row predictions, gate rule
  and gate-exempt calibration row, G rule from a pilot instead of an
  unjustified R = 5 (B4). Frozen 0.02 rule replaced by the calibrated
  network null, with the 0.748 / 0.981 mock rates as the reason. Null
  (b) descriptive only. Full seed-stream inventory; 120–121 and
  500–504 kept free but unused (consecutive Phase-1 overlap); new unit
  band 330 000–368 800 within the 2³² bound. v0.7 sections to amend
  listed. "Compatible" readings removed; equivalence only by TOST.
  Compute estimate updated. Base moved to main `c438286`.
- **Revision 2026-11-08, review 2** (independent Opus review of
  `ebc9ea3`: freezable after PI decisions, the four blocking points
  resolved). Fixed on review 2026-11-08: decisions 3 (v2 probe), 8 (K1
  runs, gate-exempt), 11 (three gate criteria), 12 (condition on the
  fixed T1 module seeds), 15 (two tags), 17 (8112026 / 8112027); 11
  decisions remain open. Decision 5 now recommends no equivalence test.
  Text fixes: |SEEDS| guard from 30 · |SEEDS| + 777 < 5000 (≤ 140);
  why 120–121 and 500–504 are not independent units; Me3 namespace
  wording; descriptive p values never in summaries; gate margin
  wording; S2 parameters from the pilot; NC1 power caveat; PC k and
  realised δ at n = 256 and 4096; TOST not expected reachable; 3/6
  sensitivity variant; skewness clause for the unit-level test.
- **2026-10-31** first draft.
