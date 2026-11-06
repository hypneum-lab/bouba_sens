# B-3 network-level null — pre-registration draft

**Status:** Proposed — **draft, not frozen, no tag, no PI sign-off.**
Every value below is a proposal; the PI decides the items of §Open
decisions before any freeze. No grid, pilot, null or synthetic check has
been run for this study.
**Date:** 2026-10-31
**ADR:** `docs/adr/0028-b3-network-level-null.md`
**Base:** main `27e17ab` (diagnostic amendment merged as `a4e6403`;
`27e17ab` adds the documentation licence only).
**Builds on:** ADR-0021, ADR-0022, ADR-0023, ADR-0024, ADR-0026,
ADR-0027; `docs/prereg/me6-large-n/amendment-2026-10-31-diagnostic.md`.

## 1. Why

P[q, m] is the accuracy on query q of the network lesioned on m. In
Me6 = max over the 10 pairs of |P[a, b] − P[b, a]|, the two entries come
from **two different lesioned networks** (columns b and a, different
lesion seeds). The ADR-0022 paired sign flip keeps those networks fixed:
its H0 is conditional on them. The Me6 large-n diagnostic (D1–D4) found
the Gaussian excess at n ≥ 256 to be a fixed, per-cell asymmetry with no
direction shared across trios, produced mainly by the lesion phase
(piste C), plus a smaller query-main-effect component in the `cell` arm
at phase 1 (piste B). A null that can tell a cross-modal asymmetry from
network-to-network variability needs **several networks per column**.

## 2. Estimand and H0

Notation: trio k (seed, timing, SNR floor), K = 30 trios per grid;
lesioned modality m; replicate r = 0 … R − 1; network N(k, m, r);
P_r[q, m] its accuracy on query q; the antisymmetric matrix
A_r[a, b] = P_r[a, b] − P_r[b, a] (10 free entries per trio).

- **Estimand:** μ_k[a, b] = E_net(A_r[a, b]), the expectation of the
  antisymmetric part **over lesion-seed draws** of the networks, given
  the trio's world, Phase-1 checkpoint (T2) and probe items.
- **H0 (network level):** no cross-modal asymmetry beyond
  network-to-network variability. Operational form used by the
  recommended null: for every trio, the joint law of a replicate's 5 × 5
  matrix is invariant under transposition, so A_r and −A_r have the same
  law and μ_k = 0.
- **Not removed:** unequal query main effects violate this H0 without
  any lesion effect (ADR-0022 limit 4, D4b-T2). A rejection is not read
  as a lesion effect (same caveat as v0.7 §5.3). Removing main effects
  (the additive fit of the 2026-10-27 prereg) is an option, §Open
  decisions.

## 3. Design

- **Replicates.** For every (trio, lesioned modality), R networks that
  differ **only by lesion seed**. Replicate 0 is the cell exactly as
  `run_grid.sh` / `run_grid_probe_v2.sh` produce it today (lesion seed
  `seed · 10000 + count`). Replicates 1 … R − 1 are new cells.
- **Phase 1 shared.** The T2 replicates of a seed restore that seed's
  single Phase-1 checkpoint; T1 replicates have none (their head init
  and Phase-2 run follow the lesion seed). The null therefore covers
  Phase-2 (and T1 initialisation) variability; Phase-1 variability stays
  at seed level and is carried by the median over trios. Replicating
  Phase 1 is an open option (cost × R on phase 1, and T1 / T2 would then
  differ in what is replicated).
- **Probe items fixed per trio.** All replicates of a trio use the
  trio's probe seeds (`seed · 10000 + 5000 + trio` for v2,
  `+ 6000 + trio` for the per-size v1 probes), so P_r[a, b] and
  P_r'[b, a] are scored on the same items. The test is conditional on
  the probe items; generalisation over items comes from the trios.

### 3.1 Lesion-seed rule for replicates

For grid seed s, cell index within its seed
j = 6 · modality index + 3 · timing index + SNR index + 1 ∈ {1 … 30}
(modality order audio, vision, tactile, gravity, force; T1 = 0, T2 = 1;
floor, minus10, plus10 = 0, 1, 2), and replicate r ≥ 1:

  **lesion seed = s · 10000 + 2000 + 30 · (r − 1) + j**, with R − 1 ≤ 25.

Its offsets lie in [2001, 2750], inside the namespace of s. Existing
offsets in that namespace: lesion counts 1–300 and their `+ 777`
(historical v1 probe) up to 1077; v2 probe seeds 5000–5005 and 5099
(+ 1 for the training pool); per-size probe seeds 6000–6005; the D4
reprobe seeds (namespace of 510, not used here). The replicate seeds
`+ 777` lie in [2778, 3527], disjoint from [2001, 2750] for R − 1 ≤ 25.
Offsets are < 10 000, so no namespace spills into s + 1. Unlike
replicate 0, j does not depend on the order of `SEEDS`. The
`seed + 1 … + 5` module seeds overlap consecutive lesion seeds exactly
as the current rule does (inherited, not new).

### 3.2 Code change (described; not implemented in this PR)

- `run_grid.sh`, `run_grid_probe_v2.sh`: opt-in `REPLICATES=R` (empty
  default = bit-identical outputs). Replicate 0 keeps its directory and
  seed; replicate r ≥ 1 writes `<cell>_r<r>/` with the rule of §3.1;
  probe seeds unchanged. Guards: refuse R − 1 > 25; refuse to resume a
  replicate cell whose recorded lesion seed differs from the rule; the
  ADR-0027 world-mode guard covers the new directories.
- `bouba-sens lesion`: record `replicate` and `lesion_seed_rule` in
  `metadata.json` and `config.yaml`.
- `scripts/me6_network_null.py`: discovery of replicate trios
  (complete only if all R × 5 cells exist), statistic, null (a),
  reference (b), synthetic size check, controls.
- Tests: default bit-identical; seed table unique and disjoint from all
  existing ranges for seeds 0–4, 100–101, 110–111, 120–121, 200–204,
  300–309, 400–404, 500–504; exact enumeration of null (a) on a toy
  grid; R = 1 rejected (degenerate, §4).

## 4. Null constructions (candidates)

Both keep the Me6 aggregation — max over the 10 pairs, then median over
trios. Me6 is an extreme-value statistic, so the null goes through the
same pipeline (no normal approximation of a single entry).

**(a) Replicate-level transpose flip — recommended.**
Statistic: T_a = median over k of max over pairs |Ā_k[a, b]|, with
Ā_k = (1 / R) Σ_r A_r (replicate-mean antisymmetric part). Null: one
sign ε_{k,r} per (trio, replicate), shared by the 10 pairs (the
replicate's matrix is replaced by its transpose); Ā*_k =
(1 / R) Σ_r ε_{k,r} A_r; same aggregation; B = 20 000 Monte Carlo
draws, p = (#{null ≥ obs} + 1) / (B + 1).
- *Pros.* Exact under the H0 of §2 for any R ≥ 2, whatever the item
  dependence; the exchangeable unit is the replicate network set, which
  is what piste C says the item flip misses; cross-pair structure kept
  (as D2); one sign covers all items, so **no block rule** is needed on
  the mock or on contiguous windows; mirrors the seed-level flip that
  was primary in the 2026-10-27 prereg.
- *Cons.* Changes the statistic (replicate mean, not one network): it
  cannot be the frozen v0.7 Me6. Degenerate at R = 1 (Me6 is
  flip-invariant, diagnostic code fact 5); coarse per trio at small R
  (2^(R−1) distinct values), smoothed by the median over 30 trios. H0
  includes "no query main effect". Cost × R.

**(b) Between-replicate difference reference.**
Statistic: the current Me6 median on replicate 0 (unchanged, as in
ADR-0026 and v0.7). Reference: for each trio, matrices
(A_r − A_r') / √2 over pairs of replicates r < r' in 1 … R − 1 (replicate
0 excluded, so the reference is independent of the observed value);
each draw takes one pair per trio at random with a random sign, then
the same aggregation; B = 20 000.
- *Pros.* Keeps the frozen statistic; H0 is location-only
  (μ_k = 0); differences have mean zero whatever μ_k, so the reference
  carries network-to-network plus item variance; cross-pair structure
  kept.
- *Cons.* Only approximately calibrated: the √2 scaling matches the
  variance, not the tails, and a max-then-median depends on the tails;
  reference draws per trio are few (C(R − 1, 2), with shared
  replicates) and dependent; R − 1 networks per cell serve only the
  reference. Must pass the synthetic size check of §6 to be usable.
- The literal cross-pairing variant (P[q, m] of replicate r against
  P[m, q] of replicate r') gives the sampling distribution of Me6 over
  network draws. It is centred on the alternative, not on H0, so it is
  a descriptive spread, not a null.

**Recommendation: (a)**, with (b) and the ADR-0022 item flip reported
side by side, descriptive. Reason: it is exact for its H0, its
exchangeable unit is the one piste C identifies, it needs no block
rule, and its only real cost (a new statistic) is unavoidable for any
exact network-level test, since the single-network Me6 is invariant to
every whole-matrix flip.

## 5. Worlds, arms, probes, gate

- **Worlds.** Gaussian (primary). Studyforrest mock with the 65 536-row
  stream of ADR-0026 only if its pilot passes the learning gate (it
  failed under straight-through on 2026-10-27); otherwise « non
  concluant (pas de réseau qui apprend) », not run.
- **Emission.** `straight-through` (the only regime that learns,
  ADR-0024): the B-3-eligible rows. `argmax-bridge` Gaussian as a
  **calibration row** (no B-3 reading, ADR-0026 rule): it is where the
  excess was found, so it is the direct test of piste C.
- **World-seed arms (ADR-0027).** `cell` (intended design) and
  `legacy`, each in its own `OUT_ROOT`, same seeds; whether both run is
  open.
- **Probes.** v1 per-size probes at n ∈ {64, 256, 1024, 4096}
  (`PROBE_SIZES`, as ADR-0026; one training grid serves all n) and the
  v2 probe (256 items, v0.7's deciding probe). Primary rows and probe
  are open.
- **Learning gate (precondition, per world and emission).** Pilot:
  the 2026-10-26 criterion (last-20-step Phase-1 loss < 1.30 and mean
  v2 vision margin ≥ 0.10). Confirmatory: vision margin ≥ 0.10 on the
  intact networks of 500–504, and mean v1 off-diagonal accuracy above
  chance by a margin to fix (proposal ≥ 0.30 at n = 4096, chance 0.25).
  Gate failed → « non concluant », no reading.

## 6. Controls (each can fail) and size check

- **Size check, before the tag** (synthetic, no data): grids of the real
  shape (30 trios × 5 × 5 × R × n) with per-network idiosyncratic query
  offsets of zero mean (piste C) at τ ∈ {0, τ_pilot, 2 τ_pilot}, 1 000
  grids, B = 2 000 each. Pass if the rejection rate is ≤ 0.016 at
  α = 0.01 and ≤ 0.064 at α = 0.05 (α + 2 MC s.e.). A construction that
  fails is not frozen. Reported, descriptive: the item flip on the same
  grids (expected to reject for τ > 0) and grids with a query main
  effect (expected to reject, limit of §2).
- **Positive control (planted, consistent sign):** in every replicate of
  every trio, k = round(δ · n / 2) items planted on audio | vision and
  gravity | force as in ADR-0026, δ ∈ {0.02, 0.05, 0.10}, 50 plantings.
  Required: power ≥ 0.80 at δ_pc (proposal 0.05 at n = 4096). Failed →
  a « compatible » reading is reported as « non concluant (puissance) ».
- **Negative control 1 (planted, random sign per replicate):** same
  planting with an independent sign per (trio, replicate) — pure network
  variability. Rejection rate must stay ≤ α + 2 MC s.e. Failed → no
  reading.
- **Negative control 2 (orientation-matched pseudo-pairs, real data):**
  replace A_r[a, b] by P_r[a, b] − P_σ(r)[a, b] (same query, same
  lesioned modality, another replicate, σ a fixed derangement): zero
  mean by construction. Null (a) must give p ≥ 0.05; p < 0.01 → control
  failed, no reading. Limit: it omits the cross-query part of the
  variance.

## 7. Power plan (pilot on separate seeds)

Pilot seeds **120–121** (12 trios), R_pilot = 5, Gaussian, each emission
and arm to be run, mock pilot only if the mock is kept. The pilot
yields: the learning gate; variance components of A (between-replicate
τ, item noise) per n; for the mock, item ACF (descriptive only, (a)
needs no block). **No test statistic of §4 is computed on the pilot.**
Then, by simulation from those components: power of (a) for K = 30 and
R ∈ {3, 5, 8, 10} against δ ∈ {0.02, 0.05, 0.10}; R = smallest value
with power ≥ 0.80 at the PI's δ, within the compute ceiling. The pilot
record and R are committed before the confirmatory grids.

## 8. Seed namespaces

- **Pilot 120–121, confirmatory 500–504.** Checked on main `27e17ab`
  (`grep -rnw` over the repository, all shell `SEEDS` lists and seed
  lists in docs): no run seed 120, 121 or 500–504, and no derived seed
  in 1 200 000–1 219 999 or 5 000 000–5 049 999. Run seeds in use: 0–5,
  7, 17, 23, 31, 42, 53, 73, 89, 100–101, 110–111, 200–204,
  300–309, 400–404, 999; derived: their `· 10000` namespaces,
  5 106 000–5 106 045 (D4, namespace of 510), resampling seeds
  25102026 and 31102026. Phase-1 module seeds s + 1 … s + 5 (121–126,
  501–509) hit no used run seed.
- **Resampling seed:** a new value, proposed at freeze (open).

## 9. Data already seen (declared)

Seen, usable only to motivate this study, never as its data: the Me6
large-n grids, seeds **400–404** (both arms, Gaussian and mock), their
paired nulls and power files; the mock pilots **100–101** (ADR-0026;
also ADR-0022 / ADR-0023 grids on 100–101); the diagnostic outputs
**D1–D4** (`reports/me6_diag_*.json`, logs), including the reprobe seeds
5 106 000–5 106 045; the 2026-10-27 lesion × query grid, seeds
**300–309**, and its pilot 110–111; the ADR-0021 to ADR-0024 grids
(0–4, 200–204) and the smoke seed 999. Seeds 120–121 and 500–504 have
produced no data.

## 10. Compute estimate (not run)

Per (emission, world, arm): confirmatory 5 Phase-1 + 150 · R lesion
jobs (lesion + eval); pilot 2 + 60 · R_pilot. Per-job slot time from
electron-server records: ≈ 85 s idle (ADR-0023), ≈ 112 s under load
10–13 (2026-10-27 grid, straight-through), ≈ 210 s under heavy load
with the per-size probes (ADR-0026 smoke).

| set (R = 5) | jobs | wall, PAR 4: idle / loaded / heavy |
|---|---:|---|
| pilot, one combination | 302 | 1.8 / 2.3 / 4.4 h |
| confirmatory, one combination | 755 | 4.5 / 5.9 / 11 h |
| proposed minimum: straight-through + argmax-bridge, Gaussian, `cell` | 2 × (302 + 755) | 12.5 / 16.4 / 31 h |

Each extra arm or world adds one row of pilot + confirmatory. Analysis
works on 5 × 5 matrices (seconds per null); synthetic size check and
power ≈ 1 h. **Host:** electron-server at nice 19 (PAR 4,
`OMP_NUM_THREADS=2`) or macM1, per the PI rules, one host for all arms,
its free state checked at freeze. Nothing is run by this PR.

## 11. Link to v0.7 ActionSense

A separate v0.7 amendment (not written here), merged and tagged before
any ActionSense grid, could add this null as follows: set
`REPLICATES=R` on the v0.7 grids (all four, or the two
straight-through ones); replicate 0 is the frozen v0.7 cell (lesion seed
`seed · 10000 + count`), so the frozen fixed-rule verdict and ADR-0022
paired null are computed exactly as registered; replicates 1 … R − 1 of
seeds 300–304 take offsets 2001–2750, disjoint from v0.7's counts
1–150 and probe seeds; ActionSense is a real-data world, so the
world-seed arms do not apply (ADR-0027) and null (a) needs no block
length. The amendment would state whether the network null is an
additional reported reading or the one that decides « distinguishable
from the null », and the cost (× R on 600 cells). Freezing this study
first lets that amendment cite a tested construction.

## 12. Open decisions (PI)

None of the following is decided.

1. **R** (proposal 5, fixed from the pilot power plan; R − 1 ≤ 25).
2. **Null construction:** (a) (recommended) or (b); which are reported
   descriptively.
3. **Seeds:** pilot 120–121 and confirmatory 500–504 (proposed, checked
   free); the resampling seed.
4. **Reading thresholds, α, multiplicity:** proposal p < 0.01 excess,
   p ≥ 0.05 compatible, otherwise non concluant (ADR-0026 scheme);
   Bonferroni over the primary rows or not; which rows are primary.
5. **Worlds, emission, arms:** Gaussian only or plus mock; whether the
   argmax-bridge calibration row runs; `cell` only or both arms.
6. **Probe and sizes:** v1 and/or v2 as primary; primary n.
7. **Learning gate** thresholds (proposal: 2026-10-26 criterion, plus
   v1 off-diagonal accuracy ≥ 0.30 at n = 4096).
8. **Positive control** δ_pc and power target (proposal 0.05, 0.80).
9. **Phase 1** shared (proposed) or replicated.
10. **Main effects:** H0 as in §2, or on interaction residuals with
    query and lesion main effects removed.
11. **Host** and compute ceiling.
12. **v0.7:** whether the v0.7 amendment of §11 follows, on which
    grids, and whether the network null decides or is reported beside.

## References

ADR-0021, ADR-0022, ADR-0023, ADR-0024, ADR-0026, ADR-0027, ADR-0028;
`docs/prereg/me6-large-n/amendment-2026-10-31-diagnostic.md`;
`docs/milestones/2026-10-31-me6-large-n-results.md`;
`docs/milestones/2026-10-27-b3-lesion-interaction-results.md`;
`docs/prereg/2026-10-26-b3-lesion-interaction.md`;
`docs/osf/amendment-v0.7-actionsense.md`;
`scripts/me6_paired_null.py`, `scripts/me6_large_n.py`,
`scripts/me6_large_n_diagnostic.py`, `scripts/run_grid.sh`,
`scripts/run_grid_probe_v2.sh`.
