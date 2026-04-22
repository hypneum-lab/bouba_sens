# OSF pre-registration amendment — bouba_sens v0.6 (Sprint 10–17)

**Status:** Draft, ready to file
**Parent pre-registration:** `dream-of-kiki` OSF DOI `10.17605/OSF.IO/Q6JYN`
**Previous amendment:** `amendment-v0.5-studyforrest-5modal.md` (4.5-modal real bridge)
**Amendment tag on file:** `bouba_sens/v0.6-sprint10-17`
**Amendment date:** 2026-04-23
**Corresponding release:** `bouba_sens` tag `v0.5.8`

## What is being amended

The v0.5 amendment added the 4.5-modal real Studyforrest bridge and the
`constellation_lock_after` hyperparameter exposed by nerve-wml v1.5.3.
Sprints 10 through 17 (2026-04-21 to 2026-04-23) explored the
mechanistic follow-up declared in §6.2 of the paper draft — whether
additional plasticity controls (transducer gating mode, codebook
freeze, phase-transition schedule) can amplify the B-1 peak observed
at `LOCK_AFTER=100` on the 4.5-modal real bridge. This amendment
registers the six new hyperparameters exercised and the alternative
mutual-information estimator used for the §6.3 robustness check.

All six additions are **purely additive**: none touches the three
locked thresholds (B-1 ≥ 0.05, B-2 ≥ 0.10, B-3 ≥ 0.02) or the
Me1/Me3/Me6/Me7 metric definitions.

## New hyperparameters (Sprint 10–14)

| Kwarg | Source | Default | Scope | ADR |
|-------|--------|--------:|-------|-----|
| `transducer_gating` | `bouba_sens.nerve.CrossModalNerve` | `"hard"` | Sprint 11 | 0014 |
| `gumbel_tau` | `bouba_sens.nerve.CrossModalNerve` | `1.0` | Sprint 12 | 0015 |
| `codebook_lock_after` | `bouba_sens.nerve.CrossModalNerve` | `None` | Sprint 13 | 0016 |
| `transducer_gating_schedule` | `bouba_sens.nerve.CrossModalNerve` | `None` | Sprint 14 | 0017 |
| `transducer_gating_target` | `bouba_sens.nerve.CrossModalNerve` | `None` | Sprint 14 | 0017 |

**Semantics (all unchanged post-filing):**

- `transducer_gating ∈ {"hard", "gumbel"}`. `"hard"` = pre-registered
  binary rule `active = gate[src] < 0.1 AND gate[dst] > 0.3`.
  `"gumbel"` = continuous sigmoid of the gate margin scaled by
  `gumbel_tau`, matching the nerve-wml#5 proposal.
- `gumbel_tau > 0` floats. Lower values tighten towards the hard
  rule; higher values smooth towards uniform. Ignored when
  `transducer_gating="hard"`.
- `codebook_lock_after: int | None`. When set, the
  `AdaptiveCodebook.codebook` Parameter has `requires_grad_(False)`
  once a nerve-level step counter crosses the threshold. Third
  compound-critical-period component alongside `constellation_lock_after`
  (v0.5) and `transducer_gating` (this amendment).
- `transducer_gating_schedule: int | None` +
  `transducer_gating_target: str | None`. When both are set, the
  gating mode silently switches from `transducer_gating` to
  `transducer_gating_target` once the nerve-level counter crosses
  the schedule. Resolved inside `fuse()`, so a Phase-1 checkpoint
  restore on T2 cells transparently resumes the correct mode.

**Empirical verdicts filed alongside this amendment:**

- ADR-0013 (Sprint 10): Amedi dose-response peak at
  `LOCK_AFTER=100`, median Me7 = +0.0125 on the 4.5-modal real
  bridge. Seed-stability 3+/5 per ADR-0017 §14a re-aggregation.
- ADR-0014 (Sprint 11): compound `LOCK=100 + GUMBEL` falsifies the
  prior hypothesis that soft gating would enhance migration;
  the hard binary rule is *qualitatively irreducible* to any
  Gumbel sigmoid.
- ADR-0015 (Sprint 12): Gumbel tau scan `{0.1, 0.3, 0.5, 2.0}` —
  tau=0.3 initially reported as anomalous B-2 peak, **retracted**
  in ADR-0017 §14a after per-seed analysis showed exact-zero
  Me3_delta at every seed.
- ADR-0016 (Sprint 13): finer tau scan + codebook freeze.
  Codebook freeze destroys the B-1 peak (+0.0125 → 0.0000). The
  bimodal B-2 claim is **retracted** in ADR-0017 §14a.
- ADR-0017 (Sprint 14): HARD → GUMBEL phase-transition schedule
  at step 200 falsifies the joint B-1 + B-2 hypothesis (B-1 drops
  to +0.0063, B-2 goes to -0.022 grid). Destructive interference
  between the two regimes is the best available reading.

## New estimator (Sprint 17)

| Estimator | Source | Scope | ADR |
|-----------|--------|-------|-----|
| MINE (Donsker-Varadhan) | `nerve_wml.methodology.mi_mine` | §6.3 B-2 robustness | 0017 §14 + §6.3 |

MINE is exercised as a side-by-side alternative to the pre-registered
`sklearn.feature_selection.mutual_info_regression` (Kraskov kNN)
estimator on the Sprint 10 peak grid (`runs/v05_dr_lock100`, N=480
pooled per seed). Both estimators return MI in nats; bouba_sens
converts to bits via `/ log(2)` as before. The Kraskov estimator
remains the **primary** Me3_delta estimator: MINE is reported as a
replication bound, not a replacement.

**Verdict (§6.3, ADR-0017):** Both estimators agree within 0.1 bit
that B-2 does not cross the 0.10 threshold on any of 5 seeds.
Kraskov mean ± std = +0.033 ± 0.047 bits; MINE returns 0.000 on
every seed (the DV clipped lower bound, non-informative below
~0.1 bit at N=480 with d=1). The retraction of the ADR-0016 B-2
bimodal claim stands; the remaining ambiguity concerns probe
dimensionality, not estimator choice.

## What is **not** amended

- No change to the three thresholds (0.05 / 0.10 / 0.02).
- No change to Me1 / Me3 / Me6 / Me7 metric definitions.
- No change to the grid shape (5 seeds × 5 modalities × 2 timings
  × 3 SNRs = 150 cells).
- No change to the paired (T1, T2) pairing for B-1.
- No change to the probe capture protocol (§3 of the paper).
- The 4.5-modal real bridge registered in v0.5 is preserved verbatim;
  all Sprint 10–17 grids ran against the same `StudyforrestRealWorld`.

## Filing

- Upload this file as `amendment-v0.6-sprint10-17.md` to the OSF
  project `Q6JYN` under "Files → Amendments".
- Tag it `bouba_sens/v0.6-sprint10-17`.
- Cross-reference the repository release at
  `github.com/hypneum-lab/bouba_sens` tag `v0.5.8` and the Zenodo
  DOI minted on the same tag (pending activation).
