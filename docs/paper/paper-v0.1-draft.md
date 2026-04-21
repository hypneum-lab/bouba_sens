# bouba_sens: A Pre-Registered Benchmark for Cross-Modal Plasticity under Critical-Period Constraints

**Authors.** Clément Saillant (Hypneum Lab).
**Version.** Paper v0.1 draft — skeleton
**Date.** 2026-04-21
**Companion artefacts.**
- Repository : `github.com/hypneum-lab/bouba_sens` (tag `v0.4.0`)
- Protocol dep : `github.com/hypneum-lab/nerve-wml` (tag `v1.4.0`, DOI `10.5281/zenodo.19666405`)
- Pre-registration : OSF `10.17605/OSF.IO/Q6JYN` (locked 2026-04-19)
- ADRs : `docs/adr/0004..0010` in the repo record every verdict inline with its grid commit.

---

## Abstract *(to be refined)*

We introduce `bouba_sens`, a pre-registered benchmark measuring three invariants (B-1 congenital-blindness gap, B-2 informational migration, B-3 perceptive/proprioceptive asymmetry) across five worlds (Gaussian, XOR, Sinusoid, Studyforrest mock, real MIT-BIH ECG) and two architectural conditions (no-lock vs `constellation_lock_after=200`).

Across 750 grid cells under the fixed pre-registered thresholds (0.05 / 0.10 / 0.02, no p-hacking vector), we find :

1. **B-3 is a robust architectural invariant**, passing every world at 7×–22× the threshold and *amplifying* as inputs move from synthetic-factorised to biologically-plausible.
2. **B-1 is world-topology-dependent and lock-sensitive.** Unconstrained, T2 dominates T1 in 4/5 worlds (directional falsification of the pre-registered Amedi 2007 hypothesis). Adding a critical-period plasticity lock produces **directional recovery** (median Me7 from −0.0063 to 0.0000 on Gaussian) without magnitude recovery at the 0.05 threshold.
3. **B-2 is estimator-limited.** Positive in 4/5 unlocked worlds but under-threshold; the lock converts it to slightly negative, which is the architecturally-correct behaviour of a critical-period model.

The benchmark's pipeline and mechanism are released alongside the paper as a reusable instrument for stress-testing other cross-modal architectures.

---

## 1. Introduction

### 1.1 Motivation

Two independent lines of evidence suggest cross-modal asymmetry is a structural property of cognitive architectures :

- **Amedi 2007**, congenital blindness : occipital cortex re-tasked to audition with *greater* plasticity than late-acquired blindness.
- **Bouba/kiki effect** (Ramachandran & Hubbard 2001) : 95 %+ cross-cultural agreement on sound-shape correspondence (“bouba” round, “kiki” spiky) — evidence of a hard-wired **asymmetric** modality mapping.

No pre-registered computational benchmark currently tests whether a given architecture reproduces these two properties simultaneously.

### 1.2 Contribution

`bouba_sens` encodes the two intuitions as three pre-registered quantitative invariants :

| Invariant | Measurement | Threshold | Biological analogue |
|-----------|-------------|----------:|---------------------|
| B-1 | `me7 = me1(T1) - me1(T2)` | > 0.05 | Amedi congenital advantage |
| B-2 | `me3_delta = MI_post - MI_pre` | > 0.10 | Informational migration |
| B-3 | `max off-diag` of 5×5 perf-matrix asymmetry | > 0.02 | Bouba/kiki structural asymmetry |

All three thresholds are frozen by the OSF pre-registration and are not revisited in this paper.

### 1.3 Structure

Section 2 defines the five `WorldSimulator` implementations and the architecture that consumes them. Section 3 details the three invariants. Section 4 reports the main grid verdicts. Section 5 records the `constellation_lock_after` mechanism and its empirical recovery pattern. Section 6 discusses limitations. Section 7 positions the work vs prior literature.

---

## 2. Architecture and datasets

### 2.1 The Nerve protocol (nerve-wml)

*Dependency stack :*
- `Nerve`, `Neuroletter`, `MlpWML`, `Transducer` primitives.
- `GammaThetaMultiplexer` : γ/θ phase-amplitude-coupled carrier with a `[64, 2]` PSK constellation trained end-to-end (Lisman & Idiart 1995 ; Tort et al. 2010).
- As of v1.4.0 : optional `plasticity_schedule : Callable[[int], float]` and `constellation_lock_after : int | None` gate the constellation gradient over training. Section 5 uses both.

### 2.2 bouba_sens architecture

5 × `SensoryWML` (one per modality) share the same mux via an `object.__setattr__` bypass that prevents `nn.Module` double-registration while preserving gradient flow. Lesions are applied through `CrossModalNerve.on_lesion` ; Phase 2 training runs with a θ-replay FIFO buffer (see `src/bouba_sens/loop.py`).

### 2.3 The five worlds

| World | Latent structure | Temporal? | Source |
|-------|------------------|-----------|--------|
| Gaussian | Orthogonal PSK projection of 32-dim N(0, I) | No | Synthetic |
| XOR | Non-linearly-separable 4-class | No | Synthetic |
| Sinusoid | Circular manifold | No | Synthetic |
| Studyforrest mock | AR(1) scene-latent surrogate | **Yes** | Synthetic with bio-plausible stats |
| MIT-BIH ECG | 1-sec-window spectrograms of real ECG | **Yes** | `scipy/dataset-ecg`, CC0, 5 min at 360 Hz |

### 2.4 The six-metric world-complexity audit

Before running the benchmark we quantify how far apart the worlds actually are using `src/bouba_sens/audit/world_complexity.py`. **Key finding (ADR-0007 / Sprint 7 Task 7.5) :** the three synthetic worlds cluster on modality geometry (intrinsic PCA dim ~30, relative gap < 3 %) but diverge on task difficulty (`linear_separability` 0.50 XOR vs 0.89 Gauss / Sin). The real ECG signal sits structurally outside the cluster (intrinsic PCA dim 6 / 28, temporal autocorr 0.97 vs ~0 synthetic). This forms the *honest external-validity caveat* of our evidence.

---

## 3. Invariants

### 3.1 B-1 (Me7) — congenital gap

Per-cell Me1 post-lesion accuracy, paired by `(seed, modality, SNR)` across T1 (congenital : no Phase 1) and T2 (late-acquired : restore Phase 1 checkpoint). The median across 75 pairs per grid is the invariant statistic.

### 3.2 B-2 (Me3 delta) — MI migration

Probe batch captured before `on_lesion` and after Phase 2 training. Codes are the mean-pooled fused representation ; `me3_delta = MI(codes_post; labels) - MI(codes_pre; labels)` via sklearn Kraskov kNN.

### 3.3 B-3 (Me6) — structural asymmetry

`AdaptationLoop.query_accuracy(modality)` zero-masks all but one modality and reports Me1. Stacking the 5 per-query vectors across one `(seed, timing, SNR)` trio yields a 5×5 matrix ; `me6_max_abs_off_diag(perf - perf.T)` is the invariant statistic.

---

## 4. Main results : unlocked grids

*(Inserts from ADR-0004 / ADR-0005 / ADR-0008 / ADR-0009.)*

### 4.1 Within-synthetic-cluster (ADR-0005)

| Invariant | Gaussian | XOR | Sinusoid |
|-----------|---------:|----:|---------:|
| B-1 Me7 | −0.0063 | −0.0062 | +0.0125 |
| B-2 Me3 delta | +0.0275 | +0.0034 | +0.0019 |
| B-3 Me6 | 0.1484 | 0.1406 | 0.1406 |

**Verdict** : B-3 PASS 3/3 (~7× threshold) ; B-1 directionally falsified (3/3 FAIL, 1 world has the correct sign) ; B-2 under-threshold 3/3.

### 4.2 Out-of-cluster mock (ADR-0008)

Studyforrest AR(1) mock : B-3 = **0.3125** (15.6×, +2× amplification), B-1 = 0.0000, B-2 = −0.0288 (artefact of zeroed tactile / gravity / force).

### 4.3 Real biological signal (ADR-0009)

MIT-BIH ECG : B-3 = **0.4453** (22.3×, strongest of any world), B-1 = −0.0062, B-2 = +0.0111.

### 4.4 The B-3 monotone-growth property

> **Key claim, load-bearing.** B-3 grows monotonically with input complexity : 7× synthetic cluster → 15.6× AR(1) mock → 22.3× real ECG. The perceptive/proprioceptive asymmetry is *not* a synthetic-cluster accident ; it intensifies when the inputs carry richer structure.

---

## 5. Mechanism : plasticity lock recovers B-1 directionally

*(Insert from ADR-0010 and — post-grid — the cross-world lock matrix from the current B+C Studio run.)*

### 5.1 Design

`constellation_lock_after=200` with `STEPS_TRAIN=200` gives T2 a locked mux at Phase 2 entry (Phase 1 crossed the threshold, checkpoint saved `requires_grad=False`, `load_state_dict` re-applies the lock). T1 starts fresh, never reaches the 200-step mark during its 100-step Phase 2 → stays plastic. This is the first architectural T1/T2 asymmetry in the benchmark.

### 5.2 Cross-world lock matrix (ADR-0011)

| World | B-1 no-lock | B-1 lock=200 | Delta |
|-------|------------:|-------------:|------:|
| Gaussian | −0.0063 (inverted) | **0.0000** | +0.0063 |
| XOR | −0.0062 (inverted) | **0.0000** | +0.0062 |
| Sinusoid | **+0.0125** (correct sign) | **0.0000** | **−0.0125** |
| real ECG | −0.0062 (inverted) | −0.0062 | 0.0000 |

| World | B-3 no-lock | B-3 lock=200 |
|-------|------------:|-------------:|
| Gaussian | 0.1484 (7.4×) | 0.1719 (8.6×) |
| XOR | 0.1406 (7.0×) | 0.1250 (6.2×) |
| Sinusoid | 0.1406 (7.0×) | 0.1562 (7.8×) |
| real ECG | 0.4453 (22.3×) | 0.4453 (22.3×) |

### 5.3 Interpretation — the lock is homogenising, not recovering

Three distinct regimes emerge from the 4 × 2 matrix :

1. **Synthetic inverted worlds (Gaussian, XOR)** : lock flips
   `me7 = −0.006` to exactly `0.000`. The inversion is gone ;
   no positive gap appears.
2. **Synthetic correct-sign world (Sinusoid)** : lock flips
   `me7 = +0.0125` to exactly `0.000`. **The positive gap that
   pre-existed is destroyed.** The lock does not selectively
   favour T1.
3. **Real ECG** : lock has no measurable effect. The 3 zeroed
   modalities dominate the architecture's response space ; the
   lock cannot induce a T1/T2 differential.

Zero of four worlds shows the pre-registered pattern
(`me7 > 0.05`). The lock acts like a low-pass filter on the
T1/T2 difference — it pushes every world toward `me7 = 0`
regardless of initial sign.

The B-2 side-effect on Gaussian (flip to −0.0092) is the
architecturally-correct behaviour of a critical-period model —
a frozen constellation cannot re-route information — but does
not help B-1 meet its threshold.

**Scientific upshot.** The simple single-parameter lock falsifies
the naïve "lock = Amedi advantage" story. The `plasticity_step`
mechanism is necessary to express a T1/T2 asymmetry but not
sufficient to reproduce the pre-registered congenital advantage
on any of the four synthetic worlds at this parameterisation.

### 5.4 Extension to 4.5-modal real biological bridge (ADR-0012)

OSF amendment v0.5 extends the pre-registered grid with a fifth
world derived from Studyforrest phase-2 (sub-01 ses-localizer
task-movielocalizer run-1): real VGG16 features over
movie_localizer.mkv, ffmpeg scene-cut tactile proxy, real
cardiac+respiration physio, zero gravity (rp regressors not
published), CC-BY-4.0 LibriSpeech audio substitution (path (b)
of ADR-0012 — the Forrest Gump soundtrack is not redistributable).

| World | B-1 no-lock | B-1 lock=200 | B-3 no-lock | B-3 lock=200 |
|-------|------------:|-------------:|------------:|-------------:|
| Gaussian | -0.0063 | 0.0000 | 0.1484 | 0.1719 |
| XOR | -0.0062 | 0.0000 | 0.1406 | 0.1250 |
| Sinusoid | +0.0125 | 0.0000 | 0.1406 | 0.1562 |
| ECG 2-modal | -0.0062 | -0.0062 | 0.4453 | 0.4453 |
| **4.5-modal real** | **0.0000** | **+0.0063** | **0.1016** | **0.1250** |

The 4.5-modal is the first world-condition pair where the lock
produces a positive B-1 (in the pre-registered Amedi direction),
although still below threshold. B-3 attenuates from the ECG
22.3× down to 5.1×-6.2× — consistent with Branch B of the OSF
amendment v0.5 decision rule ("PASS but attenuated under
biological input complexity").

B-2 is measured with a bootstrap 95% CI via nerve-wml v1.5.3
methodology module: median -0.0376, CI [-0.056, -0.001].
First world-condition where B-2 is robustly distinguishable
from zero. Sign is opposite the pre-registered +0.10 direction —
interpretation: multi-modal real inputs change the MI landscape
such that lesion-phase training reduces rather than grows the
probe-code-vs-label MI, consistent with interference rather than
migration.

B-3, however, survives the lock in 5/5 worlds — further
confirming its architectural-invariant status.

### 5.4 Open follow-up (declared)

Dose-response scan `LOCK_AFTER ∈ {100, 200, 400, 800, none}` on a single seed and compound with `nerve-wml#5` (transducer Gumbel-softmax) are declared here as *experiments-in-progress* ; their results will appear in paper v0.2 (Sprint 9+).

---

## 6. Limitations

### 6.1 External validity

Three synthetic worlds (Gaussian, XOR, Sinusoid) sit in the same tight cluster of the world-complexity audit (relative gap < 10 % on most metrics). The Studyforrest mock and MIT-BIH ECG bridges extend the support of our evidence outside that cluster, but neither is a full biological dataset. Full Studyforrest (Forrest Gump audio description + motion annotations) requires `datalad` + git-annex infrastructure that is out of scope for v0.1 and is declared explicitly in ADR-0007.

### 6.2 Scope of the lock mechanism

`constellation_lock_after` freezes the `[64, 2]` PSK constellation but leaves `CrossModalTransducer` gates and `AdaptiveCodebook` entries plastic. A fuller critical-period model would compound all three ; this is deferred to paper v0.2.

### 6.3 Me3 Kraskov-estimator limit

B-2 sub-threshold may be a measurement artefact (noisy kNN on 1-D mean-pooled probes) rather than a falsification of the migration hypothesis. nerve-wml#7 (`MlpWML.from_spectrogram`) and MINE/InfoNCE estimators are a natural follow-up.

---

## 7. Related work

*(short stubs, one paragraph each ; to be fleshed out)*

- **Cross-modal plasticity.** Amedi et al. 2007 ; Bavelier & Neville 2002 ; Merabet & Pascual-Leone 2010.
- **Bouba/kiki & ideasthesia.** Köhler 1947 ; Ramachandran & Hubbard 2001 ; Sidhu & Pexman 2018.
- **γ/θ phase-amplitude coupling.** Lisman & Idiart 1995 ; Tort et al. 2010 ; Colgin 2016.
- **Pre-registered benchmarks.** OpenReview benchmarks track (NeurIPS D&B) ; Gebru et al. 2018 (*Datasheets for Datasets*).

---

## 8. Artefacts, reproducibility, venue

- **Benchmark pipeline** : `run_grid.sh` + `aggregate_grid.py` + 5-command typer CLI. Single `uv sync --all-extras` gives a complete environment.
- **Deterministic grids** : every cell seeded by `(seed_base × 10 000 + cell_count)`. Byte-identical aggregate JSONs on a fresh clone.
- **Pre-registration fidelity** : no threshold change across all seven ADRs (0004 → 0010).
- **Target venue** : TMLR (first attempt, benchmarks track) ; NeurIPS D&B 2026 (fallback).

---

## Appendices

- **Appendix A** : world-complexity audit full table (6 metrics × 5 worlds × 5 seeds).
- **Appendix B** : cross-world lock matrix (current Studio run, to be inserted).
- **Appendix C** : SHA256 manifest for all `reports/*.json` artefacts referenced in the paper.

---

## TODO before paper v0.1 submission

- [ ] Wait for current B-1 cross-world lock grids (xor / sinusoid / ecg) to finish ; populate §5.2 table.
- [ ] Insert world-complexity audit numbers in Appendix A.
- [ ] Tighten Abstract (cut by 50 %).
- [ ] Bibliography : BibTeX entries for all cited works.
- [ ] Convert to LaTeX with the TMLR template.
- [ ] Final full-suite verdict check.

## TODO that can be deferred to paper v0.2

- [ ] Dose-response scan on `LOCK_AFTER`.
- [ ] Full datalad Studyforrest replication.
- [ ] MINE / InfoNCE Me3 estimator alternatives.
- [ ] Compound lock with `CrossModalTransducer` (nerve-wml#5).
