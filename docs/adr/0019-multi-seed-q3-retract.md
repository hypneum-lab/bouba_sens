# ADR-0019 — Multi-seed Q3 Retract of single-seed §5.5 dose-response signature

**Date:** 2026-05-11
**Status:** ACCEPTED, retract in effect
**Supersedes:** §5.5 single-seed claim (paper v0.5.7 sprint 10/11)
**Related ADRs:** ADR-0006 (critical-validation discipline), ADR-0013 (dose-response Amedi curve original)

## Context

Paper §5.5 (v0.5.7) reported a non-monotone B-1 dose-response signature with peak at LOCK_AFTER=100 and height +0.0125 (0.25× threshold 0.05) on 4.5-modal real bridge. The claim was based on a single seed (`seed=0`) and was framed as the "first qualitatively Amedi signature" of the benchmark.

Pre-registered N8-Q3 (2026-05-10, OSF-style) replayed the dose-response curve with 5 seeds {0, 17, 42, 73, 101}. N9-Q3+ (2026-05-11) extended to 5 additional seeds {7, 23, 31, 53, 89}, reaching N=10. Critic v2 audit (2026-05-11) identified that the originally pre-registered Jonckheere-Terpstra test was the wrong instrument for an inverted-U hypothesis ; the analysis was re-run with per-seed quadratic regression (concavity test).

## Findings

- **Seed=0 (the canonical seed) does NOT reproduce the original claim** : current re-run gives median_me7 = -0.0250 at LOCK_AFTER=100 (sign inverted vs original +0.0125). Indicates pipeline or environment drift between 2026-04 and 2026-05.
- **Per-seed quadratic test on 10 seeds** : 5/10 seeds show c<0 (concave-down peak), 5/10 c≥0. Sign test p=0.62 (chance level). Mean(c) one-tailed t-test p=0.28 (not significant).
- **Drift across seeds** : peak position range = [50, 150] = 100% drift. Median peak (argmax) = 100, but only 4/10 seeds have peak at 100.
- **Median peak height** = +0.025 (2× the original claim of +0.0125), but only 1/10 seeds (seed=42) crosses the B-1 threshold of 0.05.
- **Pooled-v2 subgroup analysis** (300 me7 measurements per LOCK across 10 seeds × 15 modality+SNR subgroups) : pooled signal NOT confirmed (quadratic p=0.19). Per-subgroup discovery : tactile+floor (uncorrected p=0.020) + force+plus10 (uncorrected p=0.048) emerge naïvely, NEITHER survives Bonferroni α=0.0033 across 15 subgroups.

## Decision

**Retract** the original §5.5 single-seed non-monotone claim per the N8-Q3 pre-registered decision criteria (Jonckheere p≥0.05, equivalent quadratic concavity p≥0.05).

Per the pre-registration's "Retract" branch :
- §5.5 is reformulated honestly (3 candidate versions in `docs/paper/§5.5-reformulation-draft.md` conditional on N12 outcome).
- TMLR submission is BLOCKED until §5.5 reformulation lands.
- N12 (subgroup replication, sweep RUNNING on kx6tm-23) is hypothesis-generating, not confirmatory.

## Consequences

- **Paper §5.5 must be rewritten** — version A/B/C drafts available pending N12 verdict
- **Multi-seed-first-class methodology** adopted portfolio-wide (memory entry `feedback_multi_seed_first_class.md`)
- **Critical validation guardrails** fired before external review — second confirmed instance after F1/F2/F3 v0.5.0 retract (ADR-0006). The lab's pre-publication discipline works.
- **Reproducibility evidence** :
  - Raw aggregates : `runs/v05_dr_seed{S}_lock{LA}/v0.1_aggregate.json` for 10 seeds × 5 LOCK
  - 10-seed verdict JSON : `reports/v0.5_amedi_curve_multiseed_10seed_final.json`
  - Pooled-v2 + subgroup JSON : `reports/v0.5_amedi_curve_pooled_v2.json`
  - Quadratic analysis script : `scripts/analyse_amedi_quadratic.py`
  - Pooled+subgroup script : `scripts/analyse_amedi_pooled_subgroup.py`

## Cross-reference

- Paper v0.5.7 §5.5 (pre-retract version) : git tag `v0.5.7` for historical lookup
- N8-Q3 pre-registration : `docs/milestones/q3-amedi-seeds-2026-05-10.md`
- N9-Q3+ extension : `docs/milestones/q3plus-10seeds-2026-05-11.md`
- N12 prereg : `docs/milestones/n12-amedi-subgroup-replication-2026-05-11.md`
- Critic v2 finding 1 (Jonckheere wrong test) : ID a0faff17b71d89f56 audit transcript

## Open items

- N12 sweep verdict (RUNNING ETA 2026-05-11T07:30 CEST on root@kx6tm-23) — may upgrade interpretation but does not unwind the Retract.
- Confirmatory replication at N≥30 with α=0.0033 (Bonferroni/15 subgroups) needed if subgroup signal robust enough to warrant.
- §5.5 reformulation final selection (version A/B/C) blocked on N12 verdict.
