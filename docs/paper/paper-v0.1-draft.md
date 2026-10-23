# bouba_sens: A Pre-Registered Benchmark for Cross-Modal Plasticity under Critical-Period Constraints

**Authors.** Clément Saillant (Hypneum Lab).
**Version.** Paper v0.1 — submit-ready draft
**Date.** 2026-04-22
**Companion artefacts.**
- Repository : `github.com/hypneum-lab/bouba_sens` (tag `v0.5.4`)
- Protocol dep : `github.com/hypneum-lab/nerve-wml` (tag `v1.5.3`, DOI `10.5281/zenodo.19666405`)
- Pre-registration : OSF `10.17605/OSF.IO/Q6JYN` (locked 2026-04-19, amendment v0.6 2026-04-22)
- ADRs : `docs/adr/0004..0017` record every verdict, retraction, and scope change inline with its grid commit.

---

## Abstract

We introduce `bouba_sens`, a pre-registered benchmark measuring
three invariants of cross-modal plasticity — B-1 (congenital-
blindness T1/T2 gap, Amedi 2007), B-2 (informational migration
under lesion, Me3 delta), and B-3 (perceptive/proprioceptive
structural asymmetry, bouba/kiki) — across five synthetic/surrogate
worlds (Gaussian, XOR, Sinusoid, Studyforrest mock, real ECG) and
**two real bridges with opposite real/synthetic modality splits**.
The **Studyforrest** bridge (OpenNeuro ds000113) carries real
cardiac-respiratory force, real head-motion gravity regressors, and
real scene-annotation tactile/labels, while audio and vision are
annotation-derived **proxies** (the copyrighted soundtrack and movie
frames are absent from OpenNeuro — not "real movie vision"). The
**CC-film** bridge (Big Buck Bunny, CC-BY 3.0) inverts this: audio
(log-mel), vision (frames), and labels (shot-cuts) are real perceptual
signal, while tactile/gravity/force are **synthetic** (a film has no
proprioception). The B-2 null below holds on both bridges regardless of
which half of the modality set is real. Thresholds
(0.05 / 0.10 / 0.02) are frozen by OSF registration
`10.17605/OSF.IO/Q6JYN` before the evaluation campaign.

Across 9 grids (150 cells each, 5 seeds × 5 modalities × 2
timings × 3 SNRs) and 19 ADRs, we report:

1. **B-3 raw magnitude tracks per-modality entropy, not
   partition structure** (revised 2026-04-24 per ADR-0014 + 0015).
   Median max off-diag = 0.109–0.125 on the synthetic cluster
   (5.5×–6.3× threshold), grows to 0.31 on AR(1) mock and 0.45
   on real ECG. The growth is real and reproducible, but a null-
   model partition control (9 unique 3+2 alternatives over the 5 modalities, mid-
   rank pctl) puts pre-reg perceptive/proprioceptive at 44–72 %
   percentile across 4 grids — never near the 95 % threshold.
   ADR-0015 proves the threshold is structurally unreachable for
   any modality count under the max-statistic design ; the
   relevant comparison is to the per-(n,k,m) ceiling (≈72 %
   for n=5), against which pre-reg sits in the middle of the
   random-equivalent distribution. The original "B-3 architectural
   property" reading is **withdrawn** ; the magnitude growth is
   a per-modality entropy proxy (controlled numerical sweep
   confirms B-3 ∝ noise scale at fixed k=5 ; B-3 is approximately
   constant when entropy is fixed and only k varies).
2. **B-1 does not survive multi-seed validation (ADR-0019 FINAL
   Retract).** The single-seed +0.0125 peak (Sprint 10) inverts to
   −0.0250 on the canonical seed; 10-seed sign-test p=0.62 (chance);
   pre-registered subgroup replication on N=10 new seeds fails at
   Bonferroni α=0.025. B-1 is below the noise floor of this benchmark
   (see §5.5 for the full retraction).
3. **B-2 does not exceed threshold under any configuration —
   Null confirmed/renforced (ADR-0019).** The earlier "bimodal
   positive B-2" claim (ADR-0016) is **retracted** after per-seed
   re-aggregation (ADR-0017). The Me3 sonde is now configurable with
   the **full `(B, 896)` representation as default**: the old 1-D
   mean-pool **pinned MINE to exactly 0.000 on every seed**, a
   probe-shape artefact (not an independent two-estimator agreement).
   The under-derived "probe batch ≥128" heuristic is replaced by the
   estimator-aware bound **N ≳ f(D)** keyed on the measured effective
   dimension **D ≈ 12** (participation ratio 11.7 / two-NN 12.4).
   Re-estimating with Kraskov, adaptive binning, and **MINE on the full
   896-D probe** — all bracket zero; **no estimator crosses +0.10** on
   any seed, on the gaussian reference and on **both real bridges**
   (Studyforrest + CC-film). Binning collapses in 896-D as
   `N ≫ (#bins)^D` predicts.

An earlier version of this paper claimed that a frozen mux with
hard transducer gating is the minimal configuration that
qualitatively reproduces Amedi T1/T2 asymmetry. That claim rested
on the single-seed B-1 pattern and is **withdrawn together with
it** (§5.5, ADR-0019). The surviving claims are narrow: B-1 is
retracted (below the noise floor of this benchmark), B-2 is null
on every estimator and bridge tested, and B-3 passes on magnitude
only, with its structural (partition) reading withdrawn.
bouba_sens is released at
`github.com/hypneum-lab/bouba_sens` (tag `v0.5.4`, PyPI
`bouba-sens==0.5.4`) as a reusable instrument for stress-
testing other cross-modal architectures under OSF-compliant
pre-registration ; the B-1 retraction (ADR-0019) constitutes a
portable multi-seed validation protocol for future benchmarks.

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

### 2.3 The worlds

Five synthetic/surrogate worlds plus two real bridges with **opposite
real/synthetic modality splits**. Channel provenance is labelled
REAL / PROXY / SYNTHETIC per modality — never blurred.

| World | Latent structure | Temporal? | Source / per-channel provenance |
|-------|------------------|-----------|--------|
| Gaussian | Orthogonal PSK projection of 32-dim N(0, I) | No | Synthetic |
| XOR | Non-linearly-separable 4-class | No | Synthetic |
| Sinusoid | Circular manifold | No | Synthetic |
| Studyforrest mock | AR(1) scene-latent surrogate | **Yes** | Synthetic with bio-plausible stats |
| MIT-BIH ECG | 1-sec-window spectrograms of real ECG | **Yes** | `scipy/dataset-ecg`, CC0, 5 min at 360 Hz |
| **Studyforrest real bridge** | OpenNeuro ds000113 | **Yes** | force/gravity/tactile/labels = **REAL** ds000113 signal; audio/vision = annotation **PROXY** (copyrighted soundtrack/film absent from OpenNeuro). See §5.4 / §6.3.1(d). |
| **CC-film bridge** | Big Buck Bunny (CC-BY 3.0) | **Yes** | audio (log-mel) / vision (frames) / labels (shot-cuts) = **REAL** perceptual signal; tactile/gravity/force = **SYNTHETIC** (a film has no proprioception). See §6.3.1(e). |

The two real bridges carry the true signal on **opposite halves** of the
modality set, so any verdict that holds on both is robust to which
modalities are genuine — this is exploited for the B-2 null in §6.3.1.

### 2.4 The six-metric world-complexity audit

Before running the benchmark we quantify how far apart the worlds actually are using `src/bouba_sens/audit/world_complexity.py`. **Key finding (ADR-0007 / Sprint 7 Task 7.5) :** the three synthetic worlds cluster on modality geometry (intrinsic PCA dim ~30, relative gap < 3 %) but diverge on task difficulty (`linear_separability` 0.50 XOR vs 0.89 Gauss / Sin). The real ECG signal sits structurally outside the cluster (intrinsic PCA dim 6 / 28, temporal autocorr 0.97 vs ~0 synthetic). This forms the *honest external-validity caveat* of our evidence.

---

## 3. Invariants

### 3.1 B-1 (Me7) — congenital gap

Per-cell Me1 post-lesion accuracy, paired by `(seed, modality, SNR)` across T1 (congenital : no Phase 1) and T2 (late-acquired : restore Phase 1 checkpoint). The median across 75 pairs per grid is the invariant statistic.

### 3.2 B-2 (Me3 delta) — MI migration

Probe batch captured before `on_lesion` and after Phase 2 training. Codes are the mean-pooled fused representation ; `me3_delta = MI(codes_post; labels) - MI(codes_pre; labels)` via sklearn Kraskov kNN. The full fused representation is `(B, K, d_hidden) = (B, 896)` with **effective dimension D ≈ 12** (§6.3.1) ; the default mean-pool to `(B,)` is a 1-D summary, and sample-size adequacy for the MI estimators is governed by N ≳ f(D), not a fixed batch (ADR-0019).

### 3.3 B-3 (Me6) — cross-modality asymmetry magnitude (revised reading)

`AdaptationLoop.query_accuracy(modality)` zero-masks all but one modality and reports Me1. Stacking the 5 per-query vectors across one `(seed, timing, SNR)` trio yields a 5×5 performance matrix P ; `me6_max_abs_off_diag(P - P.T)` is the raw scalar statistic.

**On the partition reading (revised 2026-04-24).** The original B-3 narrative read this magnitude as evidence for the perceptive/proprioceptive *partition* — a claim that requires the partitioned variant `me6_max_abs_off_diag_partitioned(P, partition=PERCEPTIVE_PROPRIOCEPTIVE)` to dominate the same statistic under random alternative partitions. The null-model verdict (ADR-0014) and the structural-ceiling lemma (ADR-0015) jointly establish that this reading is unsupported : the partition test cannot return a positive verdict on n=5 modalities with the max statistic, and pre-reg empirically performs no better than a random alternative across 4 grids. **B-3 magnitude is a per-modality entropy proxy** ; the partition reading is reserved for a re-designed test (planned Sprint 10).

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

### 4.4 The B-3 monotone-growth — entropy proxy, not architectural property (revised 2026-04-24)

> **Reframed claim** (replaces the original "load-bearing"
> architectural reading, retracted via ADR-0014 + ADR-0015).
> B-3 raw magnitude grows monotonically across data sources :
> 7× cluster → 15.6× mock → 22.3× ECG. The growth is **real and
> reproducible**, but our null-model partition control shows it
> is **driven by per-modality signal entropy, not by the
> perceptive/proprioceptive partition structure** the B-3
> statistic was originally meant to measure.

**Evidence** :

1. *Partition control (ADR-0014 v2)* — exhaustive enumeration
   of the 9 unique 3+2 partitions of 5 modalities on each grid
   gives pre-reg mid-rank percentiles of 61.1 % (ECG), 72.2 %
   (Mock), 55.6 % (XOR), 44.4 % (Sinu). Mean 58.3 %, well below
   the per-(n=5) structural ceiling of 72.2 % derived in
   ADR-0015. Pre-reg is statistically indistinguishable from a
   random partition of the same modalities.

2. *Structural ceiling (ADR-0015 Lemma)* — the binary
   `passes_95pct` test is unreachable for any matrix design under
   the max-statistic ; the upper bound on mid-rank percentile is
   (1 + α)/2 where α counts the alternatives that miss the hot
   pair. For n=5: ceiling = 72.2 %. For n=10: ceiling = 72.3 %.
   Threshold-based reading must be replaced with ceiling-relative
   reading.

3. *Entropy-proxy verification (ADR-0014 Axe 3)* — controlled
   numerical sweep at fixed k=5 modalities, varying noise scale
   σ ∈ {0.01, 0.05, 0.1, 0.3, 1.0} produces B-3 raw values
   {0.027, 0.138, 0.266, 0.814, 2.549} — the empirical
   0.117–0.445 range across our 4 grids fits naturally as a
   per-modality entropy variation. Modality-count sweep
   (k ∈ {3, 5, 7, 9, 11, 15} at fixed σ=0.1) gives a milder
   logarithmic dependence ; since k=5 is constant across all our
   grids, the empirical growth must come from entropy, not k.

**Implication for the broader benchmark.** B-3 is recovered as a
*per-modality signal-richness diagnostic*, not as a measure of a
cognitive partition the architecture would respect. This does
not invalidate the bouba/kiki benchmark direction — it constrains
the claims one can make from this specific statistic. A
re-designed partition test (ADR-0014 §10) was scoped for
Sprint 10 ; the methodological investigation that followed is
summarised in §4.5.

### 4.5 Partition-test framework boundary (Sprint 10–11 closure)

After ADR-0014 retracted the architectural reading of B-3 on
empirical grounds (4/4 grids fail, mid-rank percentile 22–72 %),
the Sprint 10–11 cycle searched for a replacement test that
could either confirm or definitively refute a partition signal
in the data. The conclusion is **negative and structural** :
on n=5 modalities, no permutation-symmetric statistic on a
fixed asymmetry matrix can resolve the question.

**Three findings together close the partition-test framework on
this benchmark** :

1. **ADR-0015 ceiling lemma.** For any matrix A and any n ≥ 4
   under the max-statistic partition test, the mid-rank
   percentile of pre-reg is upper-bounded by (1 + α(n, k))/2,
   where α counts random alternatives missing the optimum hot
   pair. For (n=5, k=3) the ceiling is **72.2 %** ; the
   `passes_95pct` flag is theoretically unreachable.

2. **ADR-0017 mean-statistic verdict.** Replacing max with mean
   over cross pairs gives a strictly higher ceiling under
   uniform-cross positive control (100 %), but on the 4 v0.X
   grids real data underperforms max-stat by Δ −29.2 % on
   average (mean=29.2, max=58.3). Real bouba_sens grids are in
   the sparse-signal regime where max picks up the dominant
   pair and mean dilutes.

3. **ADR-0018 equivalence theorem.** The natural escalation
   "permutation test on modality labels at fixed partition" is
   *mathematically equivalent* to the partition test at fixed
   labels — both produce the same multiset of values from the
   C(n,k) = 10 distinct partitions, by the orbit-stabilizer
   theorem (120 permutations × 10 partitions × 12 stabilizer
   elements). On n=5, the partition framework is
   information-theoretically capped at log₂(10) ≈ 3.3 bits
   regardless of how the test is instrumented.

**What this paper claims** : the perceptive/proprioceptive
partition reading of B-3 is **not statistically resolvable** on
this benchmark with this modality set. The B-3 magnitude
captured by Me6 is a per-modality entropy proxy (verified by
controlled numerical sweep, ADR-0014 Axe 3) and is reported as
such — useful as a dataset-richness diagnostic, not as a
partition-validating invariant.

**What follow-up work would be needed** to revisit the
partition reading : either a benchmark with substantially more
modalities (n ≥ 12, ADR-0016 §A) or a fundamentally different
statistical framework (e.g., Bayesian model comparison with
explicit priors over partition structures, ADR-0018 Path F).
Both are out of scope for this paper.

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
lock-consistent behaviour of a critical-period model —
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

B-3 magnitude, however, survives the lock in 5/5 worlds —
further confirming its **manipulation-invariance** (per §3.3,
the magnitude itself is reproducible and stable across
plasticity controls ; the partition reading of B-3 is
separately retracted in §4.4 + §4.5).

### 5.5 Dose-response LOCK_AFTER scan — retraction (ADR-0013, ADR-0019)

The B-1 dose-response curve was tested across 20 seeds × 5
`LOCK_AFTER` × 5 modalities × 3 SNR conditions (1500 cells × 5
LOCK = 7500 cells total, see Appendix C). The non-monotone
signature originally reported in a single-seed run (v0.5.0 §5.5)
**does not robustly reproduce**: pooled across 150 measurements per
`LOCK_AFTER`, the quadratic concavity test fails to reject the null
(p=0.19).

Pre-registered subgroup replication of 2 candidate modalities
(tactile+floor SNR and force+plus10 SNR) on N=10 new seeds also
fails to reach Bonferroni-corrected significance (α=0.025):
tactile+floor produces c=+2.14e-6 (sign inverted, p=0.595),
force+plus10 produces c=-1.11e-5 (direction preserved but p=0.129).
We retract the original single-seed claim and report this as a
methodological lesson on multi-seed validation discipline (see
ADR-0019). The dose-response signal, if present, is below the
noise floor of this benchmark and would require substantial
methodological refinement (different metric, finer `LOCK_AFTER`
resolution, or N≥30 per subgroup with α=0.0033 to control for the
15-subgroup family-wise error rate) to characterise.

### 5.6 Compound critical-period (Sprint 11, ADR-0014)

A compound experiment combining the Sprint 10 peak lock
(`LOCK_AFTER=100`) with sigmoid-soft transducer gating
(`transducer_gating="gumbel"`, `gumbel_tau=1.0`) reduces the
B-1 peak from +0.0125 to +0.0062 on the 4.5-modal real bridge —
**a 50 % attenuation, not the expected amplification**.

| Config | B-1 Me7 | B-2 Me3_delta | B-3 Me6 |
|--------|--------:|--------------:|--------:|
| LOCK=100 + HARD gate (Sprint 10) | +0.0125 | -0.0190 | 0.1094 |
| LOCK=100 + GUMBEL gate (Sprint 11) | +0.0062 | -0.0177 | 0.1172 |

Contrary to the naïve hypothesis that soft differentiability
improves gradient flow through the critical-period signal, the
hard binary `CrossModalTransducer` gate appears to act as a
**noise-filter**: by rejecting sub-threshold gate margins, it
preserves the discrete T1/T2 asymmetry. Soft gating dilutes the
signal across all 20 cross-modal pairs proportionally, reducing
the effective signal-to-noise on the Me7 axis.

This is an **architectural constraint** on the space of
mechanisms that could reproduce Amedi 2007 in our setup: at
least one component of the plasticity router needs a hard phase-
transition, not a continuous gate. B-3 remains PASS in both
configurations (0.109 vs 0.117), confirming once more its
lock-and-gate invariant status.

### 5.7 Gumbel tau scan (Sprint 12, ADR-0015)

A 5-point scan of `gumbel_tau ∈ {0.1, 0.3, 0.5, 1.0, 2.0}` at
the peak `LOCK_AFTER=100` tests whether tighter sigmoid recovers
the hard-gate behaviour.

| gumbel_tau | B-1 Me7 | B-2 Me3_delta | B-3 Me6 |
|-----------:|--------:|--------------:|--------:|
| 0.1 | 0.0000 | -0.0268 | 0.109 |
| **0.3** | +0.0063 | **+0.0180** | 0.125 |
| 0.5 | 0.0000 | -0.0404 | 0.125 |
| 1.0 | +0.0062 | -0.0177 | 0.117 |
| 2.0 | +0.0063 | -0.0052 | 0.117 |
| **hard (S10)** | **+0.0125** | -0.0190 | 0.109 |

**No Gumbel tau recovers the hard peak.** B-1 plateaus at ~+0.006
across 1.5 decades and collapses to 0 at the extremes. The hard
gate is **not a limit** of the Gumbel family — it is a
qualitatively distinct routing regime. The discrete
`gate[src] < 0.1 AND gate[dst] > 0.3` threshold produces a
phase-transition in information routing that no continuous
sigmoid approximates.

**Anomalous finding — tau=0.3 is the only positive B-2 of the
4.5-modal chain** (+0.0180, still below 0.10 threshold). Candidate
mechanism: a "Goldilocks zone" where the gate is selective enough
to preserve T1/T2 history yet smooth enough to let MI migrate.
Worth a finer scan in paper v0.3.

B-3 remains lock-gate-tau invariant (0.109-0.125, 5.4×-6.3×
threshold), confirming once more its architectural status.

### 5.8 Finer tau scan + codebook freeze (Sprint 13, ADR-0016)

**Two orthogonal follow-ups** to the tau=0.3 anomaly (§5.7) and
the deferred third plasticity component (§6.2 in earlier drafts).

**13a — bimodal positive B-2.** A finer tau grid
`{0.2, 0.25, 0.35, 0.4}` around the Sprint 12 anomaly reveals
that the positive B-2 region is **not** a plateau:

| tau | B-1 Me7 | B-2 Me3_delta | B-3 Me6 |
|----:|--------:|--------------:|--------:|
| 0.20 | +0.0063 | **+0.0116** | 0.125 |
| 0.25 | +0.0062 | -0.0109 | 0.125 |
| 0.30 | +0.0063 | **+0.0180** | 0.125 |
| 0.35 | 0.0000 | -0.0183 | 0.109 |
| 0.40 | 0.0000 | -0.0036 | 0.125 |

Two positive B-2 peaks at tau=0.20 and tau=0.30 **bracket** a
negative value at tau=0.25 — a non-monotone phase structure,
not a Goldilocks zone. B-1 plateaus at ~+0.006 for tau ≤ 0.30
then collapses to exactly 0.0000 for tau ≥ 0.35: a **critical
transition** where sigmoid sharpness rigidifies just enough to
erase T1/T2 history.

Mechanistic reading: the MI-migration channel requires the
gate to cross the `dst > 0.3` threshold *fast enough* to
preserve Phase-1 selectivity, yet *slowly enough* to let
distribution mass leak across related modality pairs during
Phase 2. The two positive B-2 peaks are distinct
"trigonometric beats" between these two constraints, not a
joint monotone improvement direction.

**13b — codebook freeze destroys the B-1 peak.** Adding
`codebook_lock_after=100` on top of the Sprint 10 hard-gate
lock configuration wipes B-1 from +0.0125 to **exactly 0.0000**:

| Config | B-1 Me7 | B-2 Me3_delta | B-3 Me6 |
|--------|--------:|--------------:|--------:|
| LOCK=100, HARD (Sprint 10 peak) | **+0.0125** | -0.0190 | 0.109 |
| + CODEBOOK_LOCK=100 | **0.0000** | -0.0117 | 0.109 |

The codebook freeze does **not** transfer the noise-filter story
from the transducer gate. Instead, the three plasticity
components carry distinct load-bearing roles:

| Component | Freezing effect | Empirical signature |
|-----------|-----------------|---------------------|
| mux constellation | preserves T1/T2 asymmetry | +0.0125 B-1 peak (Sprint 10) |
| transducer gate | filters gate noise | hard > soft (Sprints 11-12) |
| codebook | degrades T1/T2 asymmetry | destroys peak (Sprint 13) |

The shared 64-entry PSK alphabet must stay **plastic** during
Phase 2 for the Amedi signal to survive — the opposite of the
nerve-wml#5 prior hypothesis.

B-3 remains invariant across all 5 new grids (0.109-0.125),
confirming once more its architectural status.

### 5.9 Per-seed retraction + phase-transition null (Sprint 14, ADR-0017)

Sprint 14 adds per-seed stability tests to every claim the
paper has made and tries one last compound — a phase-transition
schedule that flips the transducer gating from HARD during Phase 1
to GUMBEL tau=0.30 during Phase 2. The results shrink the paper's
load-bearing story and retract one claim outright.

**14a — B-2 tau=0.30 "migration peak" retracted.** Re-
aggregating `runs/v05_s12_tau0_3` per seed shows B-2 is
**exactly 0.000 at every one of 5 seeds**. The grid-median
+0.0180 reported in §5.8 arose from `me9_bootstrap` resampling
a near-zero delta distribution (kNN-Kraskov quantisation);
the headline was bootstrap smoothing, not a migration signal.
B-1 at tau=0.30 is only weakly robust (3/5 seeds at +0.0125,
2/5 at -0.0063, mean +0.005, none passing the 0.05 threshold).

**14b — codebook entropy is the wrong observable.** Median-
over-cells codebook entropy is 4.1545 nats for both
LOCK=100 (peak) and LOCK=100+CBFREEZE (no peak) grids — the
two are indistinguishable at this resolution. The Sprint 13b
outcome stands, but its mechanistic story (§5.8) is softened:
the codebook must stay plastic, yet its role is not captured
by the entropy scalar. A finer observable (per-entry drift,
pair-wise L2) is required.

**14c — phase-transition schedule falsified.**
LOCK=100 + HARD→GUMBEL-tau-0.30 at step 200 gives grid-level
B-1=+0.0063 (2× worse than pure HARD Sprint 10 peak) and
B-2=-0.0220 (worse than any pure-mode run). One seed collapses
to B-1=-0.0375 — below anything observed in prior single-mode
runs, suggesting the mode switch at the P1/P2 boundary
introduces destructive interference between the regimes.

**Sprint 10-14 synthesis.** The only configuration that
produces a positive B-1 peak of practical significance on the
4.5-modal real bridge remains **LOCK=100 + HARD transducer**
(+0.0125 grid, 3+/5 seeds positive), from Sprint 10. Every
compound attempted since has preserved, weakened, or destroyed
it. B-2 is never robustly positive; B-3 magnitude is always
positive and manipulation-invariant across P1/P2 controls (per
§3.3, this magnitude reads as a per-modality entropy proxy
rather than as validation of the perceptive/proprioceptive
partition — see §4.5 for the framework boundary).
The paper's central empirical claim narrows to: **a frozen mux
with hard transducer gating is the minimal configuration that
qualitatively reproduces Amedi-style T1/T2 asymmetry; all
additional plasticity controls tested are null or harmful**.

---

## 6. Limitations

### 6.1 External validity

Three synthetic worlds (Gaussian, XOR, Sinusoid) sit in the same tight cluster of the world-complexity audit (relative gap < 10 % on most metrics). The Studyforrest mock and MIT-BIH ECG bridges extend the support of our evidence outside that cluster, but neither is a full biological dataset. Full Studyforrest (Forrest Gump audio description + motion annotations) requires `datalad` + git-annex infrastructure that is out of scope for v0.1 and is declared explicitly in ADR-0007.

### 6.2 Scope of the lock mechanism

`constellation_lock_after` freezes the `[64, 2]` PSK
constellation. Sprints 11-13 (§5.6, §5.8) now map the effect of
compounding two more plasticity controls — `transducer_gating=hard|gumbel`
and `codebook_lock_after` — and show that the three components
are **not** interchangeable: mux lock preserves the B-1 peak,
hard transducer gate acts as a noise filter (qualitatively
irreducible to any Gumbel sigmoid), and codebook plasticity is
**essential** — freezing it destroys the Amedi signal. A
finer-grained model of individual transducer gates (nerve-wml#5
per-modality schedules) remains declared follow-up for paper v0.3.

### 6.3 Me3 estimator : Kraskov vs MINE (Sprint 17)

The §6.3 limitation announced in earlier drafts — that B-2 sub-
threshold may be a Kraskov artefact rather than a null — is
resolved by Sprint 17 with a side-by-side MINE
(Donsker-Varadhan) replication on the Sprint 10 peak grid
(`runs/v05_dr_lock100`, LOCK=100 + HARD, the only B-1-positive
configuration). Cross-cell pooling (N=480 per seed, 30 cells
each) brings both estimators above their convergence minima.

| seed | N | Me3Δ Kraskov (bits) | Me3Δ MINE (bits) | abs diff |
|-----:|--:|--------------------:|-----------------:|---------:|
| 0 | 480 | +0.019 | 0.000 | 0.019 |
| 1 | 480 | **+0.097** | 0.000 | 0.097 |
| 2 | 480 | +0.067 | 0.000 | 0.067 |
| 3 | 480 | 0.000 | 0.000 | 0.000 |
| 4 | 480 | -0.019 | 0.000 | 0.019 |
| **mean ± std** | | **+0.033 ± 0.047** | **0.000 ± 0.000** | 0.041 |

Neither estimator crosses the pre-registered B-2 threshold of
0.10 bits on any seed (seed 1 reaches 0.097 under Kraskov — 97 %
of the threshold — but MINE gives 0 on the same data). The
Kraskov mean (+0.033 bits) is within one standard deviation of
zero across seeds, with signs split 3/5 positive, 1/5 zero,
1/5 negative. MINE returns 0 on every seed, but this is the
Donsker-Varadhan clipped lower bound, **not** a refutation of
weak signal: MINE is only informative for MI magnitudes large
compared to its sample-size-dependent floor, and at N=480 with
d=1 that floor is near 0.1 bit.

Honest reading: the two estimators **agree within ~0.1 bit**
that B-2 does not exceed the 0.10 threshold at this grid, and
Kraskov's per-seed variance is commensurate with its per-seed
mean. The benchmark cannot distinguish "weak but real B-2
signal below threshold" from "null B-2 plus estimator noise"
with 5 seeds and a 1-D mean-pooled probe.

**Two secondary consequences.**

First, the Sprint 14a framing "B-2 = exactly 0.000 at every
seed" was partly a small-sample (N=16 per cell) artefact of the
kNN-Kraskov quantisation; pooling to N=480 yields noise around
zero rather than exact zeros. The retraction of the §5.8
bimodal-peak claim **still stands** — no single seed reaches
threshold in any configuration — but the "exact zero"
phrasing is replaced by "mean +0.033 bits, std 0.047, none
threshold-crossing".

Second, the remaining unresolved ambiguity concerns probe
shape, not estimator: the current Me3_delta collapses the
`(B, K, d_hidden)` fused representation to `(B,)` via
`flatten(1).mean(-1)`. A richer probe (full `(B, d_fused)`
without mean-pooling) would let MINE exploit high-dim
structure and is declared follow-up for paper v0.2 (Sprint 18).

#### 6.3.1 E1/E2 follow-up — dimensional criterion and full-probe MINE

The Sprint-18 follow-up was executed (ADR-0019,
`scripts/spikes/e1_b2_dimensional_mine.py`). Two corrections to the
reading above.

**(a) "MINE agrees with Kraskov at ≈0" is not an independent
agreement.** The MINE column is **0.000 on every seed** because the
1-D mean-pooled probe pins the Donsker-Varadhan critic to its floor —
this is a property of the *probe shape*, not evidence about the
network. The genuine cross-estimator statement is that **Kraskov,
adaptive binning, and MINE on the full `(B, d_fused)` probe** all
bracket zero.

**(b) The "probe batch ≥128" recommendation is retired in favour of
N ≳ f(D).** The Me3 probe has ambient dimension `d_fused = K·d_hidden
= 896` but a **measured effective dimension D ≈ 12** (participation
ratio 11.7; two-NN intrinsic dim 12.4, agreeing across 5 seeds). MI
sample complexity scales with this **intrinsic** dimension, so the
fixed "≥128" heuristic is replaced by an estimator-aware bound:
report D alongside every MI estimate and treat any N ≲ 10³ for the
Me3 probe as resolving *sign*, not *magnitude*.

Re-estimating B-2 = MI_post − MI_pre at N=512 with bootstrap 95% CIs
(gaussian world — see caveat):

| Estimator (probe) | mean (bits) | 95% CI |
|-------------------|------------:|--------|
| Kraskov k-NN (1-D) | −0.031 | [−0.069, +0.005] |
| Adaptive binning (1-D) | −0.016 | [−0.041, +0.008] |
| MINE (1-D mean-pool) | 0.000 | [0.000, 0.000] |
| MINE (full 896-D probe) | −0.038 | [−0.105, +0.002] |

Feeding MINE the **full-dimensional** probe — the v0.2 hypothesis
that "migration is there but the 1-D probe hides it" — does **not**
surface a positive B-2 (mean −0.038, CI brackets 0). The B-2 verdict
stays **Null**, now without the spurious MINE-1-D agreement and with
the richer probe explicitly ruled out as a rescue.

**(c) Independent full-sonde replication** (`b2_full_sonde.csv`, all
three estimators on the 896-D probe, 5 seeds) reproduces this. MINE on
the full probe is **no longer pinned to 0** — it varies per seed
(−0.008 … +0.120, mean +0.031), confirming the `0.000` was a 1-D
probe-shape artefact; but only 1/5 seeds crosses +0.10, so the verdict
stays **Null**. Adaptive binning **collapses in 896-D** (−3 … −25
bits) exactly as the N ≫ (#bins)^D argument predicts — usable only on
a low-D summary, never the ambient probe.

*Caveat (gaussian run, retained for history).* The table above ran on
the **gaussian** world because the `BOUBA_SENS_STUDYFORREST_DATA` cache
was unavailable when it was first executed. The methodological findings
(probe-shape artefact, D ≈ 12, N ≳ f(D), full-probe MINE) are
world-agnostic; the gaussian B-2 magnitudes are kept below as the
synthetic-world reference and are **not** overwritten.

**(d) Real-bridge promotion — `--world studyforrest` on the OpenNeuro
ds000113 cache.** The one remaining step in (c)'s caveat was executed.
The real Studyforrest cache was provisioned from **OpenNeuro ds000113**
over the public S3 REST API (`scripts/fetch_studyforrest_sample.py
--source openneuro`, 9000 frames @ 10 Hz; the dead `studyforrest.org`
phase-1 URL is retired). Channel provenance is mixed and labelled
verbatim in the cache MANIFEST: **force** (cardiac pulse-ox +
respiration physio, 500 Hz), **gravity** (head-motion `*_bold_moco.txt`
rotation regressors), **tactile** and **labels** (scenes.csv scene /
DAY-NIGHT / INT-EXT annotation) are **real ds000113 signal**; **audio**
and **vision** are **annotation-derived proxies** (narration-timing and
scene-cut structure respectively) because the Forrest Gump soundtrack
and movie are copyrighted and absent from OpenNeuro. The run uses
`StudyforrestRealWorld` (mode `real5`, verified — not the mock stub).

B-2 = MI_post − MI_pre, N=512, 5 seeds, 500-resample bootstrap 95% CIs,
effective dim **D ≈ 12.5** (PR 12.9, two-NN 12.5 — matching the
gaussian D ≈ 12):

| Estimator (probe) | gaussian mean (bits) | **real-bridge mean (bits)** | **real 95% CI** |
|-------------------|---------------------:|----------------------------:|-----------------|
| Kraskov k-NN (1-D) | −0.031 | **−0.001** | [−0.0029, +0.000] |
| Adaptive binning (1-D) | −0.016 | **−0.0024** | [−0.0072, +0.000] |
| MINE (1-D mean-pool) | 0.000 | **−2.0e-7** | [−6.6e-7, +8.3e-8] |
| MINE (full 896-D probe) | −0.038 | **−1.9e-5** | [−5.7e-5, −1.8e-9] |

The real-bridge verdict is **identical to gaussian: B-2 Null.** Every
estimator's CI brackets zero; no estimator crosses the +0.10 threshold
on any of the 5 seeds. The probe-shape conclusions transfer unchanged:
MINE-1-D is pinned at the DV floor (~0), and the full-probe MINE — the
v0.2 "richer probe rescues B-2" hypothesis — again surfaces no positive
signal (mean −1.9e-5). *The B-2 magnitudes here are smaller than
gaussian's, which is expected:* the real bridge's lesioned audio channel
is an annotation proxy, so there is even less audio-specific information
available to migrate, but the qualitative null is the same.
(`reports/e1_b2_dimensional_mine_studyforrest.json`.)

*Honest scope.* This is a real bridge for force/gravity/tactile/labels
and a documented proxy for audio/vision; it is **not** a full
perceptual-stimulus replication (which would require the copyrighted
media outside OpenNeuro). The promotion lifts the "gaussian-only"
caveat for the B-2 *verdict* while keeping the audio/vision-proxy
limitation explicit.

**(e) CC-film real-AV bridge — `--world ccfilm` on Big Buck Bunny
(CC-BY).** The proxy limitation of (d) is closed from the other side by a
complementary bridge in which **audio and vision are the real perceptual
signal**. The source is Big Buck Bunny (Blender Foundation 2008, **CC-BY
3.0**), fetched from `download.blender.org`
(`scripts/extract_ccfilm_features.py`, 300 s = 600 frames @ 2 fps).
`audio` (N,128) is a **128-D log-mel spectrogram of the soundtrack** —
`scipy` STFT (n_fft=1024, hop=256, mono 16 kHz) through an **HTK mel
filter-bank written from scratch**, pooled to the frame grid and per-band
standardised (no torchaudio/librosa). `vision` (N,16,16) is the per-frame
**16×16 mean-intensity patch grid** of decoded frames. `labels` are
**real shot-cuts** (frame-to-frame difference energy, quartile-quantised
to 4, evenly populated). A film carries no proprioception, so
tactile/gravity/force are **synthetic** (RNG seeded, audio-flux-modulated
for temporal coherence; explicitly not measured signal). The world loads
as `CCFilmWorld` (mode `real_av`, verified — not the mock stub).

B-2, N=512, 5 seeds, 500-resample bootstrap, effective dim D_PR ≈ 7.6 /
two-NN ≈ 13.1:

| Estimator (probe) | mean (bits) | 95% CI |
|-------------------|------------:|--------|
| Kraskov k-NN (1-D) | −0.0317 | [−0.0633, −0.0020] |
| Adaptive binning (1-D) | +0.0062 | [−0.0327, +0.0363] |
| MINE (1-D mean-pool) | 0.0000 | [0.0000, 0.0000] |
| MINE (full N×d probe) | −0.0027 | [−0.0159, +0.0110] |

**Verdict = B-2 Null, a third time, now under real perceptual stimuli.**
No estimator reaches the +0.10 threshold; MINE-1-D is pinned at the DV
floor; full-probe MINE is null (−0.003); binning brackets zero. The only
CI excluding zero is Kraskov, and it does so on the **negative** side
(−0.032) — MI does not increase post-lesion. So even with genuinely real
audio+vision driving the representation, B-2 does not flip.

*Honest scope — a different trade-off, not a Studyforrest replacement.*
CC-film **gains** the perceptual core Studyforrest's bridge lacked (real
AV, not annotation proxies) but **loses** Studyforrest's biological
anchoring entirely and has **synthetic** proprioception — the exact
inverse of (d). Read together: (d) contributes real proprioception +
biological grounding with proxy AV; (e) contributes real AV with
synthetic proprioception. **B-2 is Null across both bridges and the three
synthetic worlds** — the null is robust to which half of the modality set
carries real signal. (`reports/e1_b2_dimensional_mine_ccfilm.json`;
cache provenance `data/ccfilm_real/MANIFEST.md`.)

---

## 7. Related work

**Cross-modal plasticity in congenital and late-acquired blindness.**
Amedi et al. (2007) show that congenitally blind subjects recruit
occipital cortex for auditory object recognition to a quantitatively
greater degree than late-onset blind subjects — our B-1 invariant
operationalises this asymmetry as a Me7 threshold. Bavelier &
Neville (2002) and Merabet & Pascual-Leone (2010) frame this as a
critical-period phenomenon: lock-style plasticity constraints mid-
development produce the T1/T2 gap that adult-onset reorganisation
cannot close; Heimler & Amedi (2020) review to what extent such
critical periods remain reversible in the adult brain.
bouba_sens's `constellation_lock_after` is a direct architectural
analogue of that critical period. The single-seed §5.5 dose-response
pattern that appeared to reproduce the qualitative Amedi asymmetry
did not survive multi-seed validation and is retracted (§5.5,
ADR-0019).

**Bouba/kiki correspondence and ideasthesia.** Köhler (1947) and
Ramachandran & Hubbard (2001) established the 95 %+ cross-cultural
agreement on round-vs-spiky sound-shape mapping; Sidhu & Pexman
(2018) review the converging evidence that this mapping is a
structural property of sensory integration, not learned. Our B-3
invariant formalises that asymmetry as the max off-diagonal
magnitude of the 5-modality perf matrix. Its 5–6× threshold pass
across every grid in this paper, independent of all Phase-1/Phase-2
controls, is a magnitude result only: the partition null model and
the structural-ceiling lemma (§3.3, §4.4–4.5) show that it tracks
per-modality entropy and does not support a hard-wired
perceptive/proprioceptive mapping. Bouba/kiki is therefore a
motivating analogue for B-3, not a property the benchmark
establishes. In models, Alper & Averbuch-Elor (2023) probe sound
symbolism in vision-and-language models (CLIP, Stable Diffusion).

**γ/θ phase-amplitude coupling as a cross-modal binding
substrate.** Lisman & Idiart (1995), Tort et al. (2010), and Colgin
(2016) describe how γ (30–100 Hz) multiplexed into θ (4–8 Hz)
envelopes implements a time-multiplexed binding scheme. The
`GammaThetaMultiplexer` in nerve-wml (Saillant 2026, issue #1)
implements this protocol; bouba_sens treats it as a black-box
protocol dependency and measures only downstream invariants.

**Pre-registered benchmarks in ML.** The NeurIPS Datasets & Benchmarks
track (Vanschoren & Yeung 2021) and Gebru et al. (2018) *Datasheets
for Datasets* pushed the field towards OSF-style pre-registration.
bouba_sens follows the OSF locked-threshold protocol (registration
`10.17605/OSF.IO/Q6JYN`): the 0.05 / 0.10 / 0.02 thresholds for
B-1 / B-2 / B-3 are frozen before the 9-grid evaluation campaign
and are not revisited by any ADR in this paper, including the
retraction in §5.9.

**Biophysical vs protocol-level nerve simulators.** Readers
searching for "nerve" in the Python ecosystem will encounter
simulators at very different levels of abstraction. PyPNS (Lubba
et al. 2019) simulates peripheral nerves at compartmental-axon
resolution on top of NEURON (Hines & Carnevale 2001), computing
extracellular potentials from a resistive quasi-static
approximation of Maxwell's equations — Hodgkin-Huxley membrane
dynamics plus FEM-derived tissue conductivities, two orders of
magnitude below the abstraction layer of our work. Classical
"nerve net" formalisms descending from McNaughton & Papert
(*Counter-Free Automata*, 1971) sit at the opposite extreme:
discrete-time automata with threshold neurons, delay-weighted
axons, and inhibitory/excitatory edges, with no biophysics at
all. nerve-wml and, by extension, bouba_sens occupy the middle
ground — a differentiable, information-theoretic *protocol* for
cross-modal binding (γ/θ phase-amplitude multiplexing of
PSK-encoded neuroletters), neither constrained to millisecond
membrane kinetics nor reduced to Boolean automata. We flag this
positioning because the word "nerve" across these frameworks
refers to genuinely incommensurable levels of description; the
B-1/B-2/B-3 invariants are only meaningful at the protocol level
and say nothing about nerve biophysics per se.

---

## 8. Artefacts, reproducibility, venue

**Software artefacts.**
- `github.com/hypneum-lab/bouba_sens` — Python package, released on
  PyPI as `bouba-sens==0.5.4` (tag `v0.5.4`, commit `360a442`).
- `github.com/hypneum-lab/nerve-wml` v1.5.3 — protocol + methodology
  dependencies (Kraskov-kNN, bootstrap CI, null model permutation,
  MINE estimator). Zenodo DOI `10.5281/zenodo.19666405`.
- `reports/*.json` + `reports/*.png` — per-grid aggregates, per-seed
  breakdowns, and the Amedi dose-response curve. Committed verbatim
  (force-added past the `.gitignore` for the reports we cite).

**Reproducibility pipeline.**
- `run_grid.sh` + `aggregate_grid.py` + `aggregate_grid_per_seed.py`
  + 5-command typer CLI. A single `uv sync --all-extras` gives a
  complete Python 3.14 environment from the pinned `uv.lock`.
- Every cell seeded by `(seed_base × 10 000 + cell_count)`. Byte-
  identical `eval_report.json` outputs on a fresh clone, modulo
  numerical determinism caveats listed in Appendix C.
- `scripts/reproduce_paper_v01.sh` (Sprint 16) regenerates every
  grid cited in §4–§5.9 from the tagged repository; total compute
  budget ≈ 8 h on an M3 Ultra.

**Pre-registration fidelity.**
- OSF registration `10.17605/OSF.IO/Q6JYN` locks the three
  thresholds `(0.05, 0.10, 0.02)`; they do not change across
  ADRs 0004 → 0017.
- Every decision (verdict, retraction, scope narrowing) is recorded
  inline in a dated ADR with the grid commit and aggregate artefact
  path, and cross-referenced in §5.x.
- OSF amendment v0.6 covers the Sprint 10–17 hyperparameter
  additions (`constellation_lock_after`, `transducer_gating`,
  `gumbel_tau`, `codebook_lock_after`, `transducer_gating_schedule`,
  `transducer_gating_target`, Me3 MINE estimator). The amendment is
  additive; no threshold or metric math changes.

**Release.**
This paper is released as an arXiv preprint with a companion
Zenodo DOI on the tagged code release. The pre-registered
retraction (§5.9, ADR-0017) is presented openly rather than
hidden : honest null and partial results are a feature of the
pre-registration protocol, not a weakness to be papered over.

---

## Appendices

- **Appendix A** — world-complexity audit full table (6 metrics × 5
  worlds × 5 seeds). Generated by `scripts/audit_worlds.py`;
  regenerate via `CHOOSE=audit bash scripts/reproduce_paper_v01.sh`.
- **Appendix B** — cross-world lock matrix, §5.2 source. The four
  per-world aggregates are committed at
  `reports/v0.4_b1_{ecg,recovery,sinusoid,xor}_lock_aggregate.json`
  and the 4.5-modal real bridge at
  `reports/v0.4_studyforrest_real_aggregate.json`.
- **Appendix C** — SHA256 byte-identity manifest for every
  `reports/*.{json,csv,png}` artefact referenced in the paper.
  File : `reports/MANIFEST.json`, regenerated by
  `scripts/sha256_manifest.py`. 40 artefacts total as of tag
  `v0.5.5`; compare against a fresh `uv run python
  scripts/sha256_manifest.py` after
  `bash scripts/reproduce_paper_v01.sh` to verify reproduction.

---

## TODO before paper v0.1 submission

- [x] B-1 cross-world lock grids — populated §5.2 (ADR-0011).
- [x] Amedi dose-response curve — §5.5, `reports/v0.5_amedi_curve.png`.
- [x] Compound lock + Gumbel transducer — §5.6–§5.8 (ADRs 0014-0016).
- [x] Per-seed stability re-aggregation — §5.9 (ADR-0017).
- [x] Sprint 17 MINE / InfoNCE side-by-side — verdict in §6.3.
- [x] Re-write Abstract using the Sprint 10–17 final numbers.
- [ ] Insert world-complexity audit numbers in Appendix A.
- [ ] Bibliography : BibTeX entries for every citation in §7.
- [ ] Convert to LaTeX with the TMLR template.
- [ ] Final full-suite verdict check against `reports/` SHA manifest.
- [ ] Zenodo DOI for tag `v0.5.4` (Sprint 16).
- [ ] OSF amendment v0.6 (Sprint 16).

## TODO deferred to paper v0.2 / v0.3

- [ ] Finer codebook-movement observable (per-entry L2 drift,
      pair-wise patterns) to rehabilitate the Sprint 13b mechanistic
      story at a resolution the entropy scalar misses (ADR-0017 §14b).
- [ ] Per-seed replication of the full Sprint 13a tau grid
      {0.20, 0.25, 0.35, 0.40}, generalising §14a's retraction.
- [ ] Full datalad Studyforrest replication beyond the 4.5-modal
      real bridge (ADR-0007 declared scope limit).
- [ ] nerve-wml#5 per-modality transducer-gating schedules
      (finer-grained than the global schedule tested in §14c).
- [x] Higher-dim Me3 probe (skip the 1-D mean-pool, keep
      `(B, d_fused)`) — **DONE (ADR-0019, §6.3.1)**: Me3 sonde made
      configurable with full `(B, 896)` default; full-probe MINE
      surfaces no positive B-2, verdict stays Null.
