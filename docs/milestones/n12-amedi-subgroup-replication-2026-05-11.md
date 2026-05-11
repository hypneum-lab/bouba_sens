# N12 — Amedi subgroup replication (tactile-floor + force-plus10)

**Date pre-registered:** 2026-05-11 (BEFORE any N12 sweep)
**Spec source:** `HYPNEUM-PLANS/2026-05-11-niveau12-amedi-subgroup-replication.md`
**Pre-registration mirror:** `HYPNEUM-PLANS/preregistrations/n12_amedi_subgroup_replication.md`

## Cross-reference

N9-Q3+ pooled-v2 analysis (commit pending in this same wave, file
`reports/v0.5_amedi_curve_pooled_v2.json`, script
`scripts/analyse_amedi_pooled_subgroup.py`) found 2 candidate
subgroups where a non-monotone Amedi-style signature emerges naïvely :

- **tactile-floor** : c=-2.107e-5, uncorrected p=0.020, peak@94.6
- **force-plus10**  : c=-1.786e-5, uncorrected p=0.048, peak@106.9

Neither survives Bonferroni correction across 15 modality × SNR
subgroups (α=0.0033). N12 is a **pre-registered targeted
replication** of these 2 subgroups specifically, with N=20 seeds
(vs N=10 in N9-Q3+) to bring effective Bonferroni α to 0.025 (only
2 specific tests pre-registered, not 15 post-hoc).

## Status

Pre-registered, sweep NOT yet executed. Q3+ FINAL Retract verdict
(see `docs/milestones/q3-amedi-seeds-2026-05-10.md`) stands at the
pooled level until N12 lands.

## H0 (to refute)

The Amedi-style non-monotone signature (concave-down quadratic with
peak at LOCK_AFTER ∈ [85, 115]) emerges in the **tactile + floor
SNR** AND/OR **force + plus10 SNR** subgroups when tested with
**N=10 NEW seeds** (the 10 N9-Q3+ seeds informed subgroup selection
and DO NOT enter the N12 verdict — see Risk section, training/test
split). At least 1 of the 2 pre-registered subgroups produces c<0
with Bonferroni-corrected p<0.025, peak estimate within [85, 115],
**AND** family-wise error rate accounting for the 15-subgroup
exploration is conservatively bounded at 15×0.025=0.375 ; therefore
even a "significant" N12 result is **hypothesis-generating, not
confirmatory**. Confirmatory replication would require α=0.0033
(Bonferroni/15) which is underpowered with N=10 — left to future
N≥30 sprint.

## Methodology

- **Subgroups tested (PRE-REGISTERED, not post-hoc)** : tactile-floor,
  force-plus10
- **Seeds** : 20 total = 10 N9-Q3+ existing seeds
  {0, 17, 42, 73, 101, 7, 23, 31, 53, 89} + 10 NEW seeds
  {3, 11, 19, 29, 37, 41, 47, 59, 67, 79} (deliberate primes,
  no overlap)
- **LOCK_AFTER values** : {50, 75, 100, 125, 150} (same as N8 Q3 /
  N9 Q3+)
- **Re-use N9 data** : skip running existing 10 seeds ; only run
  10 NEW seeds
- **For each new seed** : 5 LOCK_AFTER × 30 cells = 150 cells per
  grid → 1500 NEW cells total
- **Compute** : ~2.5-3h on `kx6tm-23` (parallel-4) — pro-rata from
  N9-Q3+ which ran 750 cells in 1h22min ; N12 = 1500 cells ≈
  2h44min just for the sweep, +10min aggregation
- **Statistical test** : per-subgroup quadratic regression (c<0),
  Bonferroni α=0.025 (only 2 PRE-REGISTERED subgroups)
- **Ancillary** : keep the 13 other subgroups as exploratory
  observations, but they do **NOT** enter the verdict

### Power analysis

With N=10 NEW seeds per subgroup (5 LOCK_AFTER values × 1 me7 per
(seed, LOCK) → 50 quadratic-fit data points but only 10 independent
units), the effective degrees of freedom for the per-seed quadratic
regression is bounded by the smaller of (number-of-LOCKs - 3 = 2)
or (number-of-seeds = 10). The sign test on c<0 across 10 seeds
reaches significance (p<0.05) when ≥9/10 seeds show concave-down —
a strong requirement. The observed effects in N9-Q3+ pooled-v2
(c=-2.1e-5 tactile-floor, c=-1.8e-5 force-plus10) had only 5/10
sign-concordant in the N9 base ; if the true effect is at this
magnitude, N=10 NEW seeds has ~30% power to detect at α=0.025.
Hypothesis-generating framing is therefore appropriate.

## Decision criteria (pre-stated)

- **N12-survive (≥1 of 2 subgroups)** : tactile-floor OR force-plus10
  produces Bonferroni-corrected p<0.025 with peak in [85, 115] →
  §5.5 saved as "Amedi-style signature is **modality-specific**
  (tactile-floor and/or force-plus10), not modality-agnostic ;
  pooled signal is washed by other 13 subgroups". Major paper
  revision but TMLR-defensible. Use Version A in
  `docs/paper/§5.5-reformulation-draft.md`.
- **N12-tie (1 of 2 marginally significant 0.025<p<0.05)** : §5.5
  reframed as "preliminary subgroup-specific signature requires
  N≥30 dedicated to each subgroup". Use Version B.
- **N12-loses (neither subgroup survives Bonferroni)** : §5.5
  retract confirmed at subgroup level too. ADR-0019 finalized.
  TMLR submission proceeds without §5.5 headline ; pivot to other
  findings. Use Version C.

## Compute budget

- N12 sweep : 10 NEW seeds × 5 LOCK × 30 cells = 1500 cells,
  ~2h44min on `kx6tm-23` 24-core CPU parallel-4 (pro-rata from
  N9-Q3+ : 750 cells / 1h22min)
- Aggregation + analysis : ~10 min
- Total : **~2.5-3h**, fits a single evening

## Risk factors

- **Subgroup selection bias** : even with pre-registration, the 2
  subgroups were chosen BECAUSE they showed naïve significance in
  N9-Q3+ pooled-v2. This is a known bias (winner's curse). Mitigate
  by **including the 10 NEW seeds only** in the verdict computation
  (the 10 N9 seeds are a "training set" that informed the subgroup
  choice ; the 10 NEW seeds are the "test set"). Document this
  explicitly when reporting.
- **Compute window** : Granite 30B not affected (CPU sweep), kx6tm-23
  has spare capacity post-Q3+.
- **Multi-modal Amedi hypothesis** : if the signature is genuinely
  modality-specific (tactile / force = proprioceptive), this is
  consistent with sensory-substitution literature where Amedi
  worked extensively. A positive result would strengthen the
  biological grounding of the bouba_sens framework.

## Cross-reference

Reproduction artefacts at `runs/v05_dr_n12_seed{S}_lock{LA}/` (NEW
dir prefix to distinguish from N9). Combined N9+N12 analysis script
`scripts/analyse_amedi_n12_combined.py` (TBD when sweep done). Paper
§5.5 reformulation in `docs/paper/§5.5-reformulation-draft.md`.

## Result (executed 2026-05-11)

Sweep ran on root@kx6tm-23 (24-core CPU parallel-4), wallclock **2h37min** (03:04 → 05:41 CEST). 50 grids = 1500 cells. Bouba_sens HEAD at sweep start : `db0fc80`. Aggregator + analyse_n12_subgroup_verdict.py reproducibility artefacts at `reports/v0.5_amedi_n12_verdict.json`.

### Per-subgroup quadratic regression on 10 NEW seeds

| Subgroup | L50 | L75 | L100 | L125 | L150 | c | bootstrap p (one-tailed) | Peak | Verdict |
|----------|-----|-----|------|------|------|---|--------------------------|------|---------|
| tactile-floor | -0.0031 | -0.0313 | -0.0063 | -0.0031 | -0.0312 | +2.14e-6 | 0.595 | no peak (c>0) | n.s. — sign inverted vs N9 |
| force-plus10 | -0.0281 | +0.0094 | -0.0187 | +0.0156 | +0.0125 | -1.11e-5 | 0.129 | @106.1 | n.s. — direction preserved |

Bonferroni α (2 pre-registered subgroups) = 0.025. Neither subgroup reaches this threshold.

### Verdict

**`N12-loses`** per pre-stated decision criteria — 0/2 subgroups replicate at Bonferroni α=0.025.

### Winner's curse manifestation

The tactile+floor signal flipped sign entirely from N9-Q3+ pooled-v2 (c=-2.1e-5, p=0.020) to N12 (c=+2.14e-6, p=0.595). Maximum regression toward the mean. Force+plus10 preserved its direction (c<0) but magnitude shrank from -1.78e-5 (p=0.048) to -1.11e-5 (p=0.129). This is the empirical confirmation of the family-wise error rate concern flagged in the pre-registration (15-subgroup exploration → effective α=0.05/15=0.0033 for true control would have rejected both N9 candidates as well).

### Consequences

- §5.5 final retract confirmed at subgroup level too (no rescue from modality-specific framing)
- §5.5 reformulation Version C selected and inserted into Paper draft : `docs/paper/§5.5-reformulation-FINAL.md`
- ADR-0019 finalized (no upgrade to Reframe)
- TMLR submission proceeds without §5.5 headline ; pivot to other findings (e.g., methodology paper emphasizing multi-seed-first-class discipline, like the v0.5.0 null-results paper structure)
- Methodology lesson : pre-registered subgroup replication after exploratory winner-selection is hypothesis-generating, not confirmatory ; true confirmatory test would require N≥30 per pre-registered subgroup with α=0.05/15 ≈ 0.0033
