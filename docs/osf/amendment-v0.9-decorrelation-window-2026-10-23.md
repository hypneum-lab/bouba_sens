# OSF pre-registration amendment — bouba_sens v0.9 (decorrelation-window lesion)

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI)
**Parent pre-registration:** `dream-of-kiki` OSF DOI `10.17605/OSF.IO/Q6JYN` (unchanged by this draft)
**Previous amendment on file:** `amendment-v0.6-sprint10-17.md`
**Sibling drafts (same date, filed independently):** `amendment-v0.7-tost-equivalence-2026-10-23.md`, `amendment-v0.8-subspace-shift-2026-10-23.md`
**Amendment tag on file:** `bouba_sens/v0.9-decorrelation-window` (to be created at filing)
**Draft date:** 2026-10-23
**Pre-registration id:** to be assigned by the PI (next free N-id in the HYPNEUM-PLANS series)

> **Filing note.** This is the machine-readable draft. Nothing in this
> file has been run. The manipulation does not exist in the code base
> yet; it must be implemented, unit-tested and frozen (commit hash on the
> OSF form) before any confirmatory seed runs. Pilot seeds (§7) may run
> before filing.

---

## 1. What is being amended

The amendment adds **a new lesion type**, a *decorrelation window*,
and one confirmatory hypothesis family about it. All modalities stay
present during the window. Only their instance-level cross-modal
correlation is removed, for a bounded window of Phase-1 training, and
then restored. No existing lesion, threshold, metric or verdict
changes.

This is **not** a rescue of the retracted §5.5 claim. The B-1 lesion
removes or degrades a channel. The manipulation here removes cross-modal
correlation, which the benchmark has never manipulated, and the outcome
is measured on a new quantity, the fusion gain (§3). ADR-0019 stands
whatever the outcome.

## 2. Motivation

Kleinman, Achille & Soatto, "Critical Learning Periods for
Multisensory Integration in Deep Networks", CVPR 2023
(arXiv:2210.04643), report that a network's ability to integrate
diverse sources depends on exposure to properly correlated signals
early in training. Interfering during this initial stage can
permanently impair integration. They report the effect in deep
(including deep linear) networks but not in shallow ones. Their
critical period is defined by the *timing of a correlated-signal
disturbance*, not by removing a modality. bouba_sens has tested only
removal (B-1, T1 vs T2), and B-1 did not survive multi-seed validation.
A decorrelation window tests the precedent's own manipulation. It is
the most direct route to a *positive* pre-registered critical-period
result in this benchmark, and a pre-registered equivalence bound makes
a null informative too.

The operationalisation below (within-class cross-modal shuffling, the
window grid, the fusion-gain outcome) is ours. It is not copied from
Kleinman et al. The PI should check the correspondence before filing
(§10).

## 3. Manipulation and outcome (pre-registered)

### 3.1 Decorrelation window

During Phase-1 steps t ∈ [o, o + w) (onset o, width w):

- For each training sample i and each modality m **independently**, the
  modality-m input is replaced by the modality-m input of a sample j
  drawn uniformly from samples with the **same label** in a pool of 256
  samples. The pool is drawn per step from the same world, with pool
  seed = training seed + 10⁶ + t. The label of i is kept.
- Effect: each modality's marginal distribution and its class-conditional
  distribution are preserved, so per-modality label information is
  intact. Instance-level cross-modal covariance beyond what class
  identity explains is removed.
- Outside the window, training is identical to the control run.

**Manipulation check (pre-registered, gates the experiment).** On 512
clean vs 512 shuffled samples, compute the mean within-class linear CKA
over the 10 modality pairs, with features flattened. The experiment is
informative only if the shuffled CKA is below the clean CKA by at least
50 % of the clean value, averaged over pilot seeds. If the world has
little within-class cross-modal dependence, the manipulation is a
near no-op. The experiment is then reported as *uninformative in this
world* and no verdict is drawn. Synthetic worlds whose modalities are
conditionally independent given the label fail this check by
construction, which is why they are not used.

### 3.2 Outcome: fusion gain and deficit

At the end of Phase 1 (no lesion), on a clean probe of N = 512:

- acc_all: accuracy with all channels present.
- acc_m: `query_accuracy(m)`, the accuracy when only modality m carries
  signal.
- **Fusion gain** FG = acc_all − max_m acc_m.
- **Deficit** D(s, o) = FG_control(s, o) − FG_decor(s, o). The control
  is the same seed trained for the same total number of steps with no
  window.

Positive D means the window permanently reduced integration.

## 4. Conditions

- World: `StudyforrestRealWorld` (4.5-modal real bridge; the real
  modalities carry temporally structured within-class covariance).
  Subject to the manipulation check.
- HARD gating, no constellation lock (LOCK_AFTER unset; a lock would
  confound the window with a freeze), no codebook lock, batch 16.
- Window width **w = 50** steps.
- **Onset grid o ∈ {0, 50, 100, 150, 200}** (5 ordered levels).
- **Recovery R = 200** clean steps after the window. The total Phase-1
  length is o + w + R, from 250 to 450 steps, and each control is
  length-matched.
- Permanence arm: o = 0 with R = 400 (decorrelated + control).
- Per seed: 6 decorrelated + 6 control Phase-1 runs.

## 5. Hypotheses

| Id | Statement | Test |
|----|-----------|------|
| **D-1** (primary) | Early decorrelation impairs integration: mean D(s, 0) > 0 | one-sample t-test, one-sided, α = 0.05 |
| **D-2** (primary) | Critical period: D decreases with onset | Page's L test, one-sided (decreasing), blocks = seeds, 5 ordered onsets, α = 0.05 |
| D-3 (secondary) | Permanence: D(s, 0) at R = 400 > 0 | one-sample t-test, one-sided |
| D-E (equivalence) | No early-window deficit of interest: \|mean D(s, 0)\| < 0.05 | TOST, bounds ±0.05, α = 0.05 |
| D-4 (exploratory) | Effect on later lesion: after a standard T2 lesion phase (100 steps, floor SNR, each modality), the post-lesion Me1 differs between decorrelated and control | descriptive, no decision |
| D-5 (exploratory) | Representation geometry: S between decorrelated and control end-of-Phase-1 codes (v0.8 metric, k = 12) | descriptive, no decision |

D-1 and D-2 together form the confirmatory family. Both must hold for
the main claim (§6). Requiring both (an intersection-union rule) needs
no α correction.

**SESOI = 0.05 accuracy units** for D-E and for the magnitude reading
of D-1. D is an accuracy difference, like Me7, and 0.05 is the
benchmark's registered smallest meaningful accuracy gap (B-1 threshold,
Q6JYN). Reusing it keeps one effect-size scale across the benchmark.

## 6. Decision criteria (pre-stated)

- **Critical period supported:** D-1 p < 0.05 **and** D-2 p < 0.05
  **and** mean D(s, 0) ≥ 0.05.
- **Critical period, sub-threshold:** D-1 and D-2 p < 0.05 but mean
  D(s, 0) < 0.05. Reported as detectable but below the benchmark's
  smallest meaningful effect.
- **Decorrelation harms, no critical period:** D-1 p < 0.05, D-2
  p ≥ 0.05. The deficit is not timing-specific.
- **Informative null:** D-E passes, i.e. no early-window deficit of
  ≥ 0.05 with 95 % confidence. Consistent with the precedent's report
  that shallow networks lack critical periods; the paper states that
  bouba_sens's fusion depth may be insufficient.
- **Inconclusive:** none of the above.
- **Uninformative:** the manipulation check fails (§3.1).

Paper consequence: a supported result is a *new* pre-registered
critical-period finding, with a correlated-signal manipulation, not
channel removal. It does not reinstate B-1 or §5.5. All outcomes are
publishable.

## 7. Seeds, power and stopping rule

- **Pilot seeds:** 3101–3105, used to validate the pipeline, run the
  manipulation check and estimate SD(D). They are excluded from
  confirmatory tests.
- **Confirmatory seeds:** 3001–3020 (N = 20), disjoint from every seed
  used so far and from the v0.7 / v0.8 drafts. **Reserve:** 3021–3024,
  for technical replacement only.
- **Power** (`scripts/power_prereg_2026_10.py`). With no pilot yet, the
  planning SD of D is borrowed from the largest per-LOCK SD of Me7 in
  Q3/Q3+ (0.0468, also an accuracy difference between two training
  regimes of the same seed), and from its UCL80 (0.0605):

  | Planning SD | N | D-1 power (true D = 0.05) | D-E power (true D = 0) |
  |-------------|--:|--------------------------:|-----------------------:|
  | 0.0468 | 10 | 0.93 | 0.86 |
  | 0.0468 | 20 | 1.00 | 1.00 |
  | 0.0605 | 10 | 0.78 | 0.56 |
  | 0.0605 | 20 | 0.97 | 0.94 |

  D-2 (Page, 5 levels): N = 20 gives power 0.94 for a total trend of
  1 within-seed SD and 0.48 for 0.5 SD (N = 30: 0.98 / 0.59).
  N = 20 is the minimum. The PI may raise it after the pilot, never
  lower it, and must fix it on the OSF form.
- **Fixed-N design.** There is no interim look at outcomes and no
  optional stopping. A technical failure is replaced from the reserve in
  order, blind to the outcome. If more than 4 replacements are needed,
  the run halts and a deviation note is filed. Extending N needs a new
  amendment filed before the new data.

## 8. What is NOT being amended

- Thresholds 0.05 / 0.10 / 0.02, the M2 lesion, the T1/T2 arms and
  every existing metric.
- ADR-0019 verdicts and the §5.5 FINAL text.

## 9. Pre-registration fidelity checklist

- [ ] Filed on OSF by the PI before any confirmatory seed (3001–3024) is run.
- [ ] Decorrelation scheduler + unit tests merged, commit hash on the OSF
      form. Required tests: labels unchanged; per-class marginals
      preserved; identity outside the window; deterministic given seed.
- [ ] Manipulation check passed on pilot seeds and recorded on the form.
- [x] No threshold or existing-metric change.
- [x] Seeds disjoint from all previous and sibling-draft seeds.

## 10. Open decisions (PI, before filing)

1. Correspondence with Kleinman et al. (2023): confirm that within-class
   cross-modal shuffling is an acceptable analogue of their
   correlated-signal disturbance, or switch to their exact manipulation
   after reading the paper.
2. Window width w = 50 and recovery R = 200 (drafted) vs a second width.
   Each added width multiplies compute.
3. Manipulation-check threshold (≥ 50 % CKA reduction, drafted).
4. Whether the onset grid should also include LOCK_AFTER regimes (a
   cross with v0.8). Not drafted, to keep the family small.
5. Optional mitigation arm (cross-modal reconstruction objective, which
   Kleinman et al. report reduces critical-period sensitivity). Not
   drafted.
6. Pre-registration id (N-id) and final amendment number.
