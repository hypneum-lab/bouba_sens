# Architecture Decision Records

ADRs capture decisions whose reversal is expensive. They are numbered
sequentially and never deleted — superseded decisions are marked
"Superseded by ADR-NNNN" at the top.

## Numbering (phase 0 "truth", 2026-10-25, issue #21)

Seven numbers — 0005, 0007, 0014, 0015, 0016, 0017, 0019 — were each
assigned to two files during the April–June 2026 sprints (parallel
worktrees, no index). To keep every existing link valid (GitHub
issues, the Hypneum Lab review of 2026-10-24, Outline, this repo),
**file names are unchanged**. Each pair is disambiguated by a suffix in
the ADR title and in this index: `a` = first committed, `b` = second
(first-commit order in `git log --diff-filter=A`). Cite them as
`ADR-0005a` / `ADR-0005b`, etc.; a bare `ADR-0014` in an older document
should be resolved through this table.

**ADR-0006 is not on `main`.** `docs/adr/0006-critical-validation.md`
exists only in commit `87b850b` ("chore(release): v0.4.0 Sprint 7
close"), reachable from the tags `v0.4.0`, `v0.5.0` and
`osf/amendment-v0.5.1` — none of which is an ancestor of `main`. The
other ADRs refer to it as "ADR-0006 (critical-validation discipline,
Sprint 7)". Bringing the file and the tags back onto `main` (cherry-pick,
or a note explaining why they stay off `main`) is a PI decision and is
not done by a pull request; see the PI action list below.

## Index

| # | File | Title | Date | Status |
|---|------|-------|------|--------|
| 0001 | `0001-codebook-sharing.md` | Codebook sharing between SensoryWMLs | 2026-04-20 | Accepted |
| 0002 | `0002-v01-go-no-go.md` | v0.1 go / no-go decision | 2026-04-20 | Accepted |
| 0003 | `0003-v01-empirical-verdicts.md` | v0.1 empirical invariant verdicts (B-1 / B-2 / B-3) | 2026-04-20 | Accepted |
| 0004 | `0004-v02-invariant-verdicts.md` | v0.2 empirical invariant verdicts (B-1 / B-2 / B-3) | 2026-04-20 | Accepted |
| 0005a | `0005-cross-world-replication.md` | Cross-world replication of B-1 / B-2 / B-3 verdicts | 2026-04-20 | Accepted |
| 0005b | `0005-v03-cross-world-verdicts.md` | v0.3 cross-world invariant verdicts | 2026-04-20 | Accepted — B-3 "architectural invariant" reading withdrawn (ADR-0014b/0015b) and requalified uncalibrated (ADR-0021) |
| 0006 | *(not on `main`)* | Critical validation of v0.3 findings (Sprint 7) | 2026-04-21 | Only in `87b850b` / tags `v0.4.0`, `v0.5.0` — PI action |
| 0007a | `0007-biological-bridge-stub.md` | Biological-adjacent bridge stub (StudyforrestWorld) | 2026-04-20 | Accepted |
| 0007b | `0007-cli-paired-runs.md` | CLI-level paired-run wiring (`eval --t1-ckpt/--t2-ckpt`) | 2026-04-20 | Accepted |
| 0008 | `0008-v04-studyforrest-mock-verdicts.md` | Studyforrest-mock out-of-cluster verdicts | 2026-04-20 | Accepted — descriptive; B-3 requalified uncalibrated 2026-10-25, paired null not conclusive (ADR-0022) |
| 0009 | `0009-real-ecg-verdicts.md` | Real-biological-signal verdicts (MIT-BIH ECG) | 2026-04-20 | Accepted — descriptive, partially retracted 2026-04-22; B-3 requalified uncalibrated 2026-10-25, not recomputed in ADR-0022 (no data) |
| 0010 | `0010-b1-recovery-via-plasticity-lock.md` | B-1 recovery via nerve-wml#4 plasticity lock | 2026-04-21 | Accepted — partial, directional |
| 0011 | `0011-cross-world-plasticity-lock-matrix.md` | Cross-world plasticity-lock matrix | 2026-04-21 | Accepted — B-3 "strongest architectural invariant" headline superseded by ADR-0014b/0015b/0021 |
| 0012 | `0012-real-5modal-studyforrest-verdicts.md` | Real 5-modality Studyforrest verdicts | 2026-04-21 | Pending (template) |
| 0013 | `0013-dose-response-amedi-curve.md` | Dose-response LOCK_AFTER scan (Amedi recovery curve) | 2026-04-21 | Accepted — single-seed; retracted by ADR-0019a |
| 0014a | `0014-compound-lock-gumbel-transducer.md` | Compound lock + gumbel transducer (Sprint 11) | 2026-04-21 | Accepted — compound hypothesis falsified |
| 0014b | `0014-b3-partition-control-verdict.md` | B-3 partition-control verdict (Sprint 9 gateway) | 2026-04-24 | Accepted |
| 0015a | `0015-gumbel-tau-scan.md` | Gumbel tau scan (Sprint 12) | 2026-04-21 | Accepted |
| 0015b | `0015-partition-test-unreachability-lemma.md` | Unreachability of the 95% threshold for the partition-control test | 2026-04-24 | Accepted |
| 0016a | `0016-finer-tau-codebook-freeze.md` | Finer tau + codebook freeze (Sprint 13) | 2026-04-22 | Accepted — B-2 peak claim retracted by ADR-0017a |
| 0016b | `0016-next-gen-partition-control-design.md` | Next-generation partition-control test design (3 candidates) | 2026-04-24 | Proposed |
| 0017a | `0017-sprint14-retraction-and-schedule.md` | Sprint 14: per-seed retraction + phase-transition schedule | 2026-04-22 | Accepted |
| 0017b | `0017-sprint10-mean-statistic-verdict.md` | Sprint 10 mean-statistic verdict (negative on real data) | 2026-04-24 | Accepted — negative result |
| 0018 | `0018-partition-test-framework-exhausted.md` | Partition-test framework exhausted on n=5; new directions | 2026-04-24 | Accepted — methodological closure |
| 0019a | `0019-multi-seed-q3-retract.md` | Multi-seed Q3 Retract of single-seed §5.5 dose-response signature | 2026-05-11 | Accepted — retract in effect |
| 0019b | `0019-b2-dimensional-mine-criterion.md` | B-2 re-run with MINE at N ≳ f(D); retire "probe batch ≥128" | 2026-06-04 | Accepted |
| 0020 | `0020-snr-floors-nearly-identical-under-default-settings.md` | SNR floors were nearly identical under default settings | 2026-10-23 | Accepted — factual audit |
| 0021 | `0021-me6-null-calibration.md` | Me6 null calibration — the B-3 statistic under sampling noise alone | 2026-10-25 | Accepted — validated by the PI |
| 0022 | `0022-me6-paired-null.md` | Me6 paired null — the B-3 median against a sign-flip null on the saved outcomes | 2026-10-25 | Accepted — PI decision 2026-10-25; B-3 Gaussian compatible with the paired null, mock not conclusive, ECG not recomputed; limit 4 corrected 2026-10-26 |

| 0023 | `0023-b3-probe-v2.md` | B-3 probe v2 — linear read-out of frozen per-modality representations | 2026-10-25 | Accepted — PI decision 2026-10-26; B-3 v2 asymmetry built by the worlds, no lesion effect (H0 ill-posed) |

| 0024 | `0024-erratum-default-regime-does-not-learn.md` | Erratum — with the default settings the network does not learn the task; B-1/B-2/B-3 requalified | 2026-10-26 | Accepted — PI decision 2026-10-26 |

| 0026 | `0026-me6-large-n-prereg.md` | Me6 at large probe sizes — preregistered paired nulls, n ∈ {64, 256, 1024, 4096} (argmax-bridge, seeds 400–404) | 2026-10-30 | Proposed — awaiting PI sign-off |

Next free number: **0027** (0025 is reserved for the ActionSense verdicts,
`docs/superpowers/plans/2026-10-28-actionsense-world.md`).

## PI action list (not resolvable by pull request)

1. ADR-0006 and tags `v0.4.0` / `v0.5.0` / `osf/amendment-v0.5.1` are
   off `main` (see above): decide between cherry-picking
   `docs/adr/0006-critical-validation.md` onto `main` (new commit, no
   history rewrite, tags untouched) and recording why they stay off.
2. No bouba_sens hypothesis is registered on OSF (issue #20): deposit
   the v0.7 pre-registration on OSF, or secure it via e-Soleau /
   restricted Zenodo, before any publication-grade claim of
   pre-registration.
