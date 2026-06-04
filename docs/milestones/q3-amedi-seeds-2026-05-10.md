# Q3 — bouba_sens dose-response seed robustness

**Date pre-registered:** 2026-05-10 (BEFORE any seed sweep run)
**Spec source:** HYPNEUM-PLANS/specs/2026-05-10-three-innovation-experiments-design.md
**Plan source:** HYPNEUM-PLANS/2026-05-10-niveau8-three-experiments.md (Task 1)
**Status:** Pre-registered, not yet executed.

## H0 (to refute)

The B-1 peak observed at LOCK_AFTER=100 in `reports/v0.5_amedi_curve.png`
(single-seed) is seed-stable: the non-monotone B-1 reproduces with ≥4/5
seeds, peak position drifts ≤25% (peak ∈ [LOCK_AFTER=75, LOCK_AFTER=125]).

## Methodology

- 5 seeds: 0, 17, 42, 73, 101 (deliberate prime/non-prime mix to avoid
  hidden RNG correlations)
- Per-seed: re-run dose-response generator (`scripts/plot_amedi_curve.py`
  adapted), measure peak position + height + full curve
- Aggregate: median peak position across 5 seeds, IQR (25th-75th
  percentile)
- Statistical test: Jonckheere-Terpstra one-tailed for non-monotonicity
  multi-seed
- Multiple-comparisons correction: Bonferroni for 5 seeds (matches OSF
  pre-reg style)

## Decision criteria (pre-stated)

- **Survives:** peak stable @ LOCK_AFTER=100 ± 25% AND Jonckheere p<0.05
  → §5.5 claim survives, add "median over 5 seeds, IQR=[X,Y]" caveat
  to figure caption + prose
- **Reframe:** peak drifts >25% but Jonckheere still p<0.05 →
  reformulate §5.5 from "peak at LOCK_AFTER=100" to "peak in [50%, 75%]
  of STEPS_TRAIN, position seed-dependent" ; figure becomes "median
  curve with seed-IQR shading"
- **Retract:** Jonckheere p≥0.05 → claim Amedi non-monotonicity
  rétractable, ADR-0019 needed, §5.5 reframes to "non-monotone
  signature requires more N to confirm"

## Result (5-seed interim 2026-05-11)

Executed on `root@kx6tm-23` (commit `db0fc80`), 1h22min wallclock.
Full data: `reports/v0.5_amedi_curve_multiseed_5seed_interim.json`.

| Seed | Peak LOCK_AFTER | Height   |
|------|-----------------|----------|
| 0    | 50              | +0.0250  |
| 17   | 100             | +0.0187  |
| 42   | 100             | +0.0562  |
| 73   | 100             | +0.0375  |
| 101  | 150             | +0.0250  |

- 3/5 seeds peak at LOCK_AFTER=100, but seed=0 (original) at 50,
  seed=101 at 150
- Median peak = 100 ; IQR = [100, 100]
- Drift = 100% (range [50, 150]) — exceeds ±25% Reframe bound
- Median peak height = +0.025 (2× original single-seed claim of
  +0.0125)
- Jonckheere-Terpstra one-tailed ascending p=0.17, descending
  p=0.34 → non-monotonicity **not significant**

## Verdict (interim)

**`Retract`** per pre-registered decision criteria (Jonckheere
p≥0.05). The §5.5 non-monotone B-1 peak claim does not robustly
reproduce across 5 seeds. **TMLR submission blocked** until §5.5
reformulation lands and ADR-0019 is drafted.

**Q3+ extension RUNNING** : 10-seed sweep on `root@kx6tm-23`
(ETA 2026-05-11T02:30 CEST). Verdict may upgrade to `Reframe`
if expanded sample tightens Jonckheere below α=0.05. Final
closeout pending Q3+ completion.

## Q3+ closeout — 10-seed final verdict (2026-05-11)

10-seed sweep completed on `root@kx6tm-23` (5 base + 5 plus =
{0,17,42,73,101,7,23,31,53,89} × {50,75,100,125,150}). Per the
N8/N9 critic MAJOR #4 finding, the verdict is computed via
**per-seed quadratic regression** (`y = a + b*x + c*x^2`, test
`c < 0`), not Jonckheere-Terpstra (which tests monotonic trend
and is the wrong statistical instrument for an inverted-U
hypothesis).

Reusable analysis script : `scripts/analyse_amedi_quadratic.py`
(adapted from the inline `/tmp/q3_combined_verdict.py` used on
kx6tm-23). Final verdict JSON :
`reports/v0.5_amedi_curve_multiseed_10seed_final.json`.

| Statistic                                  | Value         |
|--------------------------------------------|---------------|
| Seeds with c<0 (concave-down)              | 5/10          |
| Sign test p (binomial, c<0)                | 0.62          |
| t-test mean(c)<0, one-tailed p             | 0.28          |
| Median peak position (argmax)              | 100           |
| Drift across seeds                         | 100% (50–150) |
| Median peak position (quadratic, c<0 only) | 100           |

Both the sign test and the t-test fail to reject H0 at α=0.05.
The non-monotone (inverted-U) signature **does not robustly
reproduce** across 10 seeds. The original 5-seed `Retract`
verdict is **CONFIRMED FINAL**. Paper §5.5 must be reformulated
or the claim retracted ; ADR-0019 mandatory before TMLR
re-submission.

## Pooled-v2 nuance discovery (2026-05-11)

After the FINAL Retract closeout above, a follow-up **pooled
subgroup-discovery analysis** was run on `root@kx6tm-23` to test
whether averaging me7 within each LOCK_AFTER (instead of per-seed
quadratic) reveals a different signal — and whether modality × SNR
sub-populations exhibit the Amedi-style signature where the pooled
mixture does not.

Artefacts :
- `reports/v0.5_amedi_curve_pooled_v2.json`
- `scripts/analyse_amedi_pooled_subgroup.py`

### Pooled verdict (150 me7 per LOCK_AFTER)

| LOCK_AFTER | median me7 |
|------------|-----------:|
| 50         | -0.0125    |
| 75         | -0.0063    |
| 100        | +0.0000    |
| 125        | -0.0125    |
| 150        | -0.0063    |

Pooled quadratic : c = -2.1e-6, p(c<0) = **0.19**, peak estimate
LOCK_AFTER ≈ 148.9 (NOT 100). Pooled non-monotone signature **NOT
confirmed**. The 5-seed and 10-seed Retract verdict above stands.

### Per-subgroup signal (15 sub-pops modality × SNR)

Two sub-populations exhibit naïve-significant concave-down quadratic
fits with peaks in the original Amedi window :

| Subgroup       | c          | uncorr p | peak  |
|----------------|-----------:|---------:|------:|
| tactile-floor  | -2.107e-5  | 0.020    | 94.6  |
| force-plus10   | -1.786e-5  | 0.048    | 106.9 |

Bonferroni correction across 15 subgroups (α = 0.05/15 = 0.0033) :
**neither survives**. These are candidate signals, not confirmed
findings. The remaining 13 subgroups show no quadratic signal.

### Implications

The FINAL Retract verdict at the **pooled / per-seed** level is
**not contradicted**. The pooled curve has no Amedi-style signature
and the 10-seed sign and t tests both failed to reject H0.

However, the subgroup analysis hints that the signature, if real,
may be **modality-specific** (proprioceptive : tactile-floor and
force-plus10) rather than modality-agnostic. This is consistent
with the Amedi sensory-substitution literature, but the current
evidence is insufficient (winner's curse + post-hoc subgroup
selection + 2/15 pre-Bonferroni hits is exactly what one expects
under H0 for 15 tests).

A pre-registered targeted replication of these 2 subgroups
(N12 sprint, doubled seed count to N=20) is queued in
`docs/milestones/n12-amedi-subgroup-replication-2026-05-11.md`
and `HYPNEUM-PLANS/preregistrations/n12_amedi_subgroup_replication.md`.
N12 outcome will determine the final §5.5 reformulation
(see `docs/paper/§5.5-reformulation-draft.md` for 3 candidate
versions A / B / C).

The TMLR submission remains blocked on §5.5 reformulation +
ADR-0019, as in the FINAL Retract closeout.
