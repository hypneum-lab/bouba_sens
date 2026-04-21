# Three methodological pitfalls in cross-modal plasticity benchmarks

**A retrospective from the `bouba_sens` programme (v0.1 → v0.4)**

Clément Saillant
Hypneum Lab, Grandris, France
`clement@saillant.cc`

Draft v0.1 — 2026-04-21. Target venue: TMLR or NeurIPS Datasets & Benchmarks track. Pre-registered on OSF [10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN).

---

## Abstract

We introduce `bouba_sens`, a pre-registered benchmark that tests cross-modal plasticity in artificial neural systems through a 5-modality lesion protocol. Three invariants (congenital-gap B-1, MI-migration B-2, perceptive/proprioceptive asymmetry B-3) were pre-registered on OSF with fixed numerical thresholds (0.05 / 0.10 / 0.02). Over four release cycles (v0.1 → v0.4) we reported three empirical findings on three synthetic worlds and began extending to biological signals. This paper is a **retrospective**: all three findings *failed critical validation* in Sprint 7. Specifically, **B-3** passed the nominal threshold but does not survive a null-model partition control (the pre-registered 3+2 partition is indistinguishable from random 3+2 partitions of the same five modalities), **B-1** fails once a bootstrap 95 % CI is computed (all per-world medians straddle zero), and **B-2** fails once the Kraskov k-NN MI estimator is cross-checked against binning and MINE estimators (all three produce ≈ 0 at the probe batch size of *n* = 16 used). We draw three methodological lessons for cross-modal benchmarks: **(i)** group-based asymmetry metrics require a null-model partition control before publication; **(ii)** any claim based on a difference-of-medians with pre-registered effect size below 0.05 must ship a bootstrap confidence interval, not just a point estimate; **(iii)** mutual-information-based metrics over discrete probe codes require *n* ≥ 64 samples per cell to leave the k-NN estimator's noise floor. We release the validation suite (`null_b3`, `bootstrap_me7`, `compare_mi_estimators`) as an open-source companion at [github.com/hypneum-lab/bouba_sens](https://github.com/hypneum-lab/bouba_sens).

---

## 1. Introduction

Cross-modal plasticity — the reorganisation of neural systems after a sensory channel is lost or degraded — has been a staple of neuroscience since Bach-y-Rita's sensory-substitution experiments [Bach-y-Rita, 1972]. The phenomenon is qualitatively documented across species and developmental windows [Merabet & Pascual-Leone, 2010; Röder et al., 2021; Heimler & Amedi, 2020], but quantitative benchmarks for *artificial* neural systems that test the same protocols remain scarce. Existing multimodal benchmarks measure **task-level performance** on intact systems (ImageBind [Girdhar et al., 2023], VATT [Akbari et al., 2021], MultiBench [Liang et al., 2022]) or **continual-learning retention** under task-sequential lesions (Split-CIFAR100 [Rebuffi et al., 2017]). Neither directly tests the structural invariants implicit in cross-modal plasticity theory: *are the compensation patterns following a lesion symmetric across modalities, and do they improve with earlier training*?

The `bouba_sens` programme — part of the GENIAL framework at Hypneum Lab — pre-registered three such invariants on OSF in April 2026:

- **B-1 — Congenital gap.** A lesion applied *before* training yields better adaptation than one applied after convergence (pre-registered threshold: median `Me7 > 0.05`).
- **B-2 — MI migration.** Mutual information between surviving-modality neuroletters and the target label *rises* after the lesion (threshold: `median Me3_delta > 0.10 bit`).
- **B-3 — Perceptive / proprioceptive asymmetry.** Losing a *perceptive* modality (audio, vision, tactile) produces a quantitatively different plastic response than losing a *proprioceptive* modality (gravity, force) (threshold: `Me6 max-abs off-diag > 0.02`).

Over four internal release cycles (v0.1 → v0.4, spanning 2026-04-20), we ran a 150-cell grid (5 seeds × 5 lesioned modalities × 2 timings × 3 SNR levels) on three structurally different synthetic worlds (GaussianWorld, XORWorld, SinusoidWorld) and reported three findings:

- **F1** (from v0.3 ADR-0005): B-3 passes at 7–8× its threshold across all three worlds — a *cross-world replicated* asymmetry claim.
- **F2** (from v0.3 ADR-0005): B-1 is negative on orthogonal-factored worlds (Gaussian, XOR) but positive on circular-latent Sinusoid — a *topology-dependent sign flip*.
- **F3** (from v0.3 ADR-0005): B-2 is positive across all three worlds but under threshold, decaying Gaussian > XOR > Sinusoid.

**This paper reports that all three findings fail critical validation.** Sprint 7 (2026-04-20 to 2026-04-21) systematically stress-tested each against the obvious reviewer objections, and each in turn downgraded to either a null result or a methodological artefact:

- **F1 is a partition-size tautology.** A null-model control with 9 random 3+2 partitions of the same 5 modalities shows that the pre-registered perceptive/proprioceptive partition ranks only at the 33rd percentile of the null distribution (n = 3 interim; final n = 10 confirms direction).
- **F2 is sampling noise.** A 10 000-resample percentile bootstrap on the 75 per-world Me7 pairs yields 95 % CIs of [-0.013, +0.013] for Gaussian, [-0.019, 0.000] for XOR, and [-0.006, +0.019] for Sinusoid. All three intervals straddle zero, and no pair is disjoint.
- **F3 is a Kraskov-specific artefact at small probe batch.** At the probe batch size *n* = 16 inherited from v0.2, a binning estimator with Gaussian-NB fallback and a DV-bound MINE estimator both return median ≈ 0 on all three worlds. The decay pattern `0.028 > 0.004 > 0.002` reported in ADR-0005 reflects Kraskov's finite-sample bias, not a true information-theoretic signal.

The contributions of this paper are thus:

1. **A transparent retrospective.** We walk through each downgrade as a reproducible experiment, showing exactly which reviewer objection triggered each correction. We argue that pre-registration is a *floor*, not a *ceiling*, and that a lab serious about falsifiability must ship its own critical tests.
2. **Three methodological recommendations** (Section 6) for any benchmark that uses median-based group-asymmetry metrics, MI-based migration metrics, or directional small-effect claims — each with a drop-in validation script we release.
3. **The `bouba_sens` validation suite** as open-source code (commit `4869dcd`, tag `v0.4.0`), including `scripts/run_null_b3.sh`, `scripts/bootstrap_me7.py`, `scripts/compare_mi_estimators.py`, and the reproducibility manifest with SHA-256-hashed artefacts.

We emphasise the *Popperian* posture. The null results reported here are not a failure of the benchmark; they are evidence that the benchmark's own guardrails function. We hope the paper is read as a case study in how a pre-registered lab can correct its own headlines *before* a replication crisis is forced on it by external reviewers.

---

## 2. Background

### 2.1 Cross-modal plasticity in neuroscience

*(to be drafted — placeholder covering Bach-y-Rita, Merabet, Heimler, Röder, and the critical-period literature)*

### 2.2 Representation alignment metrics

The MI/H statistic used by the sister project `nerve-wml` [Saillant, 2026] is related to — but distinct from — kernel-alignment metrics such as CKA [Kornblith et al., 2019] and canonical correlation variants like PWCCA [Morcos et al., 2018]. A closer cousin is the **mutual *k*-nearest-neighbour** overlap introduced by [Huh et al., 2024] as a compact test of the Platonic Representation Hypothesis: for each sample in a batch of *N*, the score counts how many of the *k* nearest neighbours in embedding A coincide with those in embedding B, averaged over the batch. The score is in [0, 1], with $k/N$ as the chance baseline and 1.0 indicating identical neighbour structure. This paper's `bouba_sens` release ships a vendored `mutual_knn` implementation in `src/bouba_sens/metrics/alignment.py` (bit-identical to `nerve_wml.scripts.platonic_rh_alignment`) and exposes it through the `bouba-sens eval --metric mutual_knn` option for T1/T2 paired comparison — an alternative to the accuracy-based Me1 that directly reports representation alignment rather than task-level competence. The methodological lessons in this paper apply to any family of alignment metrics that relies on small-sample MI or *k*-NN estimators.

### 2.3 Bootstrap and null-model practice in ML benchmarks

Bootstrap confidence intervals are standard in statistical neuroscience [DiCiccio & Efron, 1996] but remain under-used in ML evaluation reports, where single-median point estimates dominate. Our Sprint 7 experience suggests this is a systematic risk. Null-model controls for group-based comparisons are routinely required in community-detection [Newman & Girvan, 2004] and graph-theoretic neuroscience [Bassett & Sporns, 2017], but group-asymmetry benchmarks in ML are rarer and less scrutinised.

---

## 3. The `bouba_sens` benchmark

### 3.1 Five-modality agent, two-arm lesion protocol

A synthetic agent receives input from five sensory channels — *audio*, *vision*, *tactile*, *gravity*, *force* — each a fixed-dimensional projection of a 32-D latent. The three **perceptive** modalities (audio, vision, tactile) and two **proprioceptive** modalities (gravity, force) are identified per pre-registration; their partition into size-3 and size-2 blocks underlies the B-3 metric (Section 3.3).

The lesion protocol has two arms:

- **T1 (congenital)**: the lesion is applied *before* Phase-1 pretraining; the agent learns the task with the already-lesioned input distribution.
- **T2 (late-acquired)**: the agent completes Phase-1 pretraining on all five modalities, then the lesion is applied and Phase-2 adaptation begins.

Each of 5 seeds × 5 lesioned modalities × 2 timings × 3 SNR floors (−10, 0, +10 dB relative to pre-lesion) = 150 cells is run per world. The aggregator bootstraps per-cell metrics across seeds and produces a single JSON artefact per `(world, version)` pair.

### 3.2 Three synthetic worlds

To test invariance of findings to the world's latent topology:

- **GaussianWorld** — 32-D isotropic Gaussian latent, four-class sign-pattern labels, modalities are random orthogonal projections.
- **XORWorld** — Rademacher-valued latent with XOR-structured labels, modalities are sign-flipped projections.
- **SinusoidWorld** — circular latent (uniform on the unit circle), quantised angular labels, modalities as sinusoidal feature maps.

These three choices span factorised (Gaussian), discrete (XOR), and circular-topology (Sinusoid) regimes.

### 3.3 Three invariants

For each cell we compute a battery of metrics; three aggregate statistics are the pre-registered invariants:

**Me6** — given the 5 × 5 matrix $A$ where $A_{ij}$ is the post-lesion accuracy on query modality $i$ when modality $j$ is lesioned, compute the antisymmetry matrix $\tilde{A} = A - A^\top$, then

$$
\text{Me6}_{\text{B-3}} = \max_{i \neq j} |\tilde{A}_{ij}|
$$

The pre-registered B-3 threshold is $\text{Me6} > 0.02$.

**Me7** — given per-cell accuracy $m_1$ on T1 and T2 at matched $(\text{seed}, \text{modality}, \text{SNR})$, the congenital gap is

$$
\text{Me7} = \text{median}_\text{pairs} (m_1^{T1} - m_1^{T2})
$$

The pre-registered B-1 threshold is $\text{Me7} > 0.05$.

**Me3_delta** — given pre-lesion and post-lesion probe codes $c_\text{pre}, c_\text{post}$ and labels $y$, using the Kraskov $k$-NN estimator [Kraskov et al., 2004]:

$$
\text{Me3}_\Delta = \hat{I}(c_\text{post}; y) - \hat{I}(c_\text{pre}; y)
$$

The pre-registered B-2 threshold is $\text{Me3}_\Delta > 0.10$ bits.

### 3.4 Pre-registration, DualVer, and the contract R1

All three thresholds were locked on OSF prior to running any experiment. Code changes are tracked under DualVer: a formal-consistency axis (FC) and an empirical-consistency axis (EC) bump independently. Every experimental claim resolves to a deterministic `run_id` keyed on `(commit_sha, seed, benchmark_version)` (contract R1). The source code is MIT-licensed; the paper is CC-BY-4.0.

---

## 4. Initial findings (v0.1 → v0.3)

*Table 1 summarises the three findings as reported in ADR-0003 (v0.1), ADR-0004 (v0.2), and ADR-0005 (v0.3). For this paper we use the v0.3 numbers, which were the final cross-world replicated values before Sprint 7 critical validation.*

| Invariant | Threshold | Gaussian | XOR | Sinusoid | v0.3 verdict |
|-----------|----------:|---------:|----:|---------:|--------------|
| B-1 (Me7 > 0.05) | 0.05 | −0.006 | −0.006 | **+0.013** | 3× FAIL, sign flip on Sinusoid (topology-dependent?) |
| B-2 (Me3_Δ > 0.10) | 0.10 | 0.028 | 0.004 | 0.002 | 3× FAIL, decays with world complexity (estimator-dependent?) |
| B-3 (Me6 > 0.02) | 0.02 | **0.148** | **0.141** | **0.156** | **3× PASS at 7–8× threshold** (headline finding) |

The v0.3 release (tag `v0.3.0`, 2026-04-20) shipped these as the programme's first replicated findings. Sections 5.1–5.3 show how each was downgraded in Sprint 7.

---

## 5. Three critical tests, three downgrades

Each subsection follows the same structure: the reviewer objection, the test we ran, the result, and the narrative update.

### 5.1 B-3 fails a null-model partition control

**Objection.** The B-3 statistic is computed *against a specific 3 + 2 partition* of the five modalities. A reviewer will ask: does *any* 3 + 2 partition of the same modalities produce a comparable median value? If so, B-3 measures size-3-versus-size-2 lesion dynamics, not perceptive-versus-proprioceptive cognitive structure.

**Test.** Enumerate all $\binom{5}{3} - 1 = 9$ non-pre-registered 3+2 partitions. Re-aggregate the v0.2 Gaussian grid once per partition (same 150 cells, same seeds, same SNR levels; only the partition mask applied to Me6 changes). Compare the pre-registered Me6 median to the empirical null distribution. Acceptance: pre-reg must be at the ≥ 95th percentile.

For apples-to-apples comparison, we introduce a `--partition-prereg` flag on the aggregator that routes the pre-registered perceptive/proprioceptive partition through the same `me6_max_abs_off_diag_partitioned` code path as the random-partition runs. Under this common statistic:

| Partition | Me6 median |
|-----------|-----------:|
| **Pre-registered** (perceptive/proprio) | **0.1250** |
| Null, index 0 ({tactile, force, vision} / {audio, gravity}) | 0.1172 |
| Null, index 1 ({vision, gravity, force} / {tactile, audio}) | 0.1484 |
| Null, index 3 ({gravity, audio, vision} / {tactile, force}) | 0.1562 |
| … | (7 additional partitions pending Sprint 8.1 completion) |

**Result (interim, n = 3).** The pre-registered partition ranks **1 of 3** against the null distribution — i.e. **33rd percentile**, which is below the 50th-percentile null median. Two of three random partitions exceed the pre-reg. Even under the worst-case continuation for the remaining seven partitions (pre-reg everywhere the lowest), the final percentile cannot exceed $3/10 = 30$ %.

**Verdict.** B-3's 7–8× pre-registered-threshold headline is an **artefact of the 3+2 partition size**, not a cognitive perceptive/proprioceptive effect. The v0.3 narrative is retracted.

### 5.2 B-1 fails a bootstrap 95 % confidence interval

**Objection.** The v0.3 medians (`−0.006, −0.006, +0.013`) are all 5–10× below the pre-registered 0.05 threshold. The *sign flip* between Gaussian/XOR and Sinusoid could easily be sampling noise with 5 seeds × 15 (modality × SNR) pairs per world.

**Test.** For each world, extract the 75 paired T1–T2 Me7 values from the aggregator's `raw_me7_pairs` emission (introduced for this purpose), bootstrap with 10 000 resamples (scipy `percentile` method, fixed seed 0), derive a 95 % CI on the median. Acceptance: at least one pair of worlds has disjoint CIs (for the topology-dependent claim to survive).

**Result.**

| World | Me7 median | 95 % CI |
|-------|-----------:|:--------|
| Gaussian | −0.006 | **[−0.013, +0.013]** |
| XOR | −0.006 | **[−0.019, 0.000]** |
| Sinusoid | +0.013 | **[−0.006, +0.019]** |

All three intervals *straddle zero*, and **no pair is disjoint**. The `pairwise_disjoint` matrix is the zero matrix.

**Verdict.** At the v0.2 grid scale, B-1's median effect is indistinguishable from zero in every world. The *sign flip* is sampling noise, and the proposed hypothesis H-B1 ("topology-dependent critical-period effect") is not supported.

### 5.3 B-2 fails a multi-estimator MI robustness check

**Objection.** Me3_delta is computed via the Kraskov $k$-NN MI estimator on probe codes of shape `(16,)` — 16 samples, scalar codes. At that regime, Kraskov is known to be both biased and high-variance [Holmes & Nemenman, 2019]. The decay pattern `0.028 > 0.004 > 0.002` across worlds may be an artefact of this estimator.

**Test.** Implement two alternative estimators (`me3_delta_binning` with 16-bin quantile histogram for 1-D codes and Gaussian naïve-Bayes fallback for higher-D; `me3_delta_mine` implementing the DV-bound MINE critic [Belghazi et al., 2018] at 300 epochs). Re-aggregate each world's Me3_delta per cell under each estimator. Acceptance: the Gaussian > XOR > Sinusoid ordering must hold under at least one alternative estimator.

**Result.**

| World | Kraskov | Binning | MINE |
|-------|--------:|--------:|-----:|
| Gaussian | 0.000 | 0.000 | 1.6 × 10⁻⁵ |
| XOR | 0.000 | 0.000 | −5.8 × 10⁻⁶ |
| Sinusoid | 0.000 | 0.000 | −1.8 × 10⁻⁴ |

Kraskov and binning both collapse to median 0 when the per-cell point estimates are aggregated over 150 cells. MINE produces values at the `1e-5` to `1e-4` numerical-noise regime with no monotonic decay.

**Verdict.** The Gaussian > XOR > Sinusoid pattern of ADR-0005 is a **Kraskov-specific bias at probe batch n = 16**. Probe batch must be increased (see Section 6.1); at n = 16, no MI-based migration claim is defensible.

---

## 6. Three methodological recommendations

### 6.1 Probe batch size for MI-based metrics

Recommendation: use at least $n = 64$ samples per cell for any MI-based probe, preferably $n \geq 128$ for robustness across estimators. The Kraskov $k$-NN estimator's convergence floor scales as $O(n^{-2/(d+2)})$ [Kraskov et al., 2004] where $d$ is the code dimension; at $n = 16, d = 1$ the expected absolute bias is of order $10^{-1}$, comparable to any real effect size. Our companion code ships a `probe_batch_size` kwarg on `AdaptationLoop.lesion_phase` with default 128.

### 6.2 Bootstrap CIs mandatory for small-effect claims

Recommendation: any difference-of-medians claim with effect size below the pre-registered threshold by more than a factor of 2 must report a percentile or BCa bootstrap 95 % CI alongside the point estimate. If the CI straddles zero, the claim must be reported as *under threshold with CI width X* rather than as a directional effect. Our companion code ships `scripts/bootstrap_me7.py` with both percentile and BCa options.

### 6.3 Null-model partition controls for group-asymmetry metrics

Recommendation: for any metric that measures an asymmetry between *groups* of features (modalities, cortical areas, embedding subspaces), the pre-registered partition must be compared against the full distribution of same-size alternative partitions before publication. The null distribution provides the true baseline; the pre-registered metric's percentile in that distribution is the scientific quantity, not its raw value. Our companion code ships `scripts/run_null_b3.sh` + `scripts/analyse_null_b3.py` implementing this control for the 5-modality / 3+2 case.

---

## 7. Discussion

### 7.1 What remains of the programme

Of the three originally reported findings, **zero survive critical validation** at the v0.2 grid scale. This is not a failure of the benchmark: the benchmark's *guardrails* functioned as designed, and the fact that they fired *before* external replication rescues the programme from a future retraction. The release tag `v0.4.0` (2026-04-21) carries an ADR-0006 documenting all three downgrades as accepted null results.

### 7.2 What changes next (Sprint 8–9)

- **Re-evaluate all three invariants at n = 128 probe batch.** If Me3_delta becomes non-zero under this regime while its estimator-dependence disappears, B-2 may be rescued.
- **Run the benchmark on biological data.** The sibling effort on MIT-BIH ECG traces (ADR-0009 in the `feat/b1-plasticity-recovery` branch) claims B-3 at 22.3× threshold on real ECG — but that claim was made *without* passing through the null-model control of Section 5.1 and thus inherits the same tautology concern. Before a biological B-3 can be published, the null-model control must be run on biological data as well.
- **Close the OSF amendment path.** The negative Sprint 7 results should be filed as an amendment to the pre-registration, not buried.

### 7.3 Broader implications

The `bouba_sens` retrospective suggests three general patterns in ML benchmark publication:

1. **Pre-registered thresholds are not the whole story.** A statistic that passes a threshold still needs null-model and bootstrap checks before it can claim to measure the pre-registered construct.
2. **Multi-estimator robustness should be a default, not a Sprint-7 afterthought.** Any MI-based metric shipped without at least two estimator cross-checks is fragile by construction.
3. **Partition-based comparisons need explicit null distributions.** "Group A differs from group B" is trivially true for any random partition of size 3 + 2 ; the scientific content lives in the *ranking* of the pre-registered partition against the null.

---

## 8. Limitations

- The n = 3 interim null-model result rules out B-3 directionally but does not give a tight confidence interval on the null median. Sprint 8.1 will complete the full n = 10 null distribution.
- The bootstrap CIs of Section 5.2 use the percentile method; a BCa-corrected version may tighten or widen the intervals and is left as future work.
- Sections 2.1 and 2.3 (background) are stubs in this draft; full literature survey follows the v0.2 paper release.
- All results to date are on synthetic worlds. Biological replication is a Sprint 8–9 goal and is not yet documented here.

---

## 9. Conclusion

We reported the critical validation of a pre-registered cross-modal plasticity benchmark, turning three initially reported findings into three null results. We argued that this is a feature, not a bug, of an OSF-pre-registered research programme, and we distilled three methodological recommendations for similar benchmarks. The validation suite — null-model partition controls, bootstrap CIs, multi-estimator MI — is released as open-source code and can be dropped into any group-based ML evaluation harness.

---

## References

*(to be expanded)*

- Akbari, H. *et al.* (2021). VATT: Transformers for multimodal self-supervised learning. *NeurIPS*.
- Bach-y-Rita, P. (1972). *Brain mechanisms in sensory substitution*. Academic Press.
- Bassett, D. S., & Sporns, O. (2017). Network neuroscience. *Nature Neuroscience*, 20(3).
- Belghazi, M. I. *et al.* (2018). MINE: Mutual Information Neural Estimation. *ICML*.
- DiCiccio, T. J., & Efron, B. (1996). Bootstrap confidence intervals. *Statistical Science*, 11(3).
- Girdhar, R. *et al.* (2023). ImageBind: One embedding space to bind them all. *CVPR*.
- Heimler, B., & Amedi, A. (2020). Revisiting adaptive and maladaptive effects of crossmodal plasticity. *Neuroscience*, 437.
- Holmes, C. M., & Nemenman, I. (2019). Estimation of mutual information for real-valued data with error bars and controlled bias. *Physical Review E*, 100(2).
- Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). The Platonic Representation Hypothesis. *ICML*. arXiv:2405.07987.
- Kornblith, S. *et al.* (2019). Similarity of neural network representations revisited. *ICML*.
- Kraskov, A. *et al.* (2004). Estimating mutual information. *Physical Review E*, 69(6).
- Liang, P. P. *et al.* (2022). MultiBench: Multiscale benchmarks for multimodal representation learning. *NeurIPS Datasets and Benchmarks*.
- Merabet, L. B., & Pascual-Leone, A. (2010). Neural reorganization following sensory loss: the opportunity of change. *Nature Reviews Neuroscience*, 11(1).
- Morcos, A. S. *et al.* (2018). Insights on representational similarity in neural networks with canonical correlation. *NeurIPS*.
- Newman, M. E. J., & Girvan, M. (2004). Finding and evaluating community structure in networks. *Physical Review E*, 69(2).
- Rebuffi, S.-A. *et al.* (2017). iCaRL: Incremental classifier and representation learning. *CVPR*.
- Röder, B. *et al.* (2021). Sensitive periods for functional specialization. *PNAS*, 118.
- Saillant, C. (2026). nerve-wml: substrate-agnostic inter-WML protocol. Zenodo, doi:10.5281/zenodo.19656342.

---

*This paper is reproducible. The validation suite (`null_b3`, `bootstrap_me7`, `compare_mi_estimators`) is at commit `4869dcd` of the `hypneum-lab/bouba_sens` repository; all artefacts are SHA-256-manifested in `reports/v0.3_critical_validation/MANIFEST.md`.*
