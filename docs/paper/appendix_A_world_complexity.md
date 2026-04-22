# Appendix A — World-complexity audit (verbatim numbers)

**Source** : `reports/v0.3_critical_validation/world_complexity_audit.json`,
produced by `src/bouba_sens/audit/world_complexity.py` at tag `v0.4.0`
(ADR-0007, Sprint 7 Task 7.5).
**Scope** : the audit in its current form covers the **three synthetic
worlds** (Gaussian, XOR, Sinusoid). The Studyforrest mock and MIT-BIH
real-ECG worlds do not have an audit JSON — their structural distance
from the synthetic cluster is argued qualitatively from single-number
summaries cited in §2.4 and §4.3 (intrinsic PCA dim 6 / 28, temporal
autocorr ≈ 0.97). Running the 6-metric audit on these two worlds is
left as follow-up (see §6.1 limitations and the TODO v0.2 bloc).

## A.1 Condensed table (one row per world, aggregates over modalities)

| Metric | Gaussian | XOR | Sinusoid |
|--------|---------:|----:|---------:|
| Intrinsic PCA dim (mean over audio+vision+tactile) | 30.00 | 30.00 | 29.00 |
| Support compactness (mean over audio+vision) | 0.176 | 0.176 | 0.170 |
| Pairwise MI (median) | 0.0292 | 0.0243 | 0.0358 |
| Label-conditional entropy | 1.9228 | 0.9770 | 1.8164 |
| Linear separability | 0.891 | 0.502 | 0.871 |
| Temporal autocorrelation | −0.002 | +0.019 | −0.009 |

**Reading.** The three synthetic worlds are tightly clustered on
*modality geometry* (Intrinsic PCA dim 29–30 ±0, support compactness
0.170–0.176 ±0.01) but diverge on *task difficulty* (linear
separability : Gaussian 0.891, Sinusoid 0.871, XOR 0.502). This
matches the pre-registered prediction : B-3 is architecturally
invariant across the cluster (5.5–6.3× threshold), while B-1 depends
on the difficulty gradient (qualitative Amedi peak emerges on
Sinusoid, collapses on Gaussian/XOR).

## A.2 Per-modality breakdown (all 14 metrics, all 3 worlds)

### Intrinsic PCA dimensionality (per modality)

| World | Audio | Vision | Tactile | Gravity | Force |
|-------|------:|-------:|--------:|--------:|------:|
| Gaussian | 30.0 | 30.0 | 30.0 | 3.0 | 6.0 |
| XOR | 30.0 | 30.0 | 30.0 | 3.0 | 6.0 |
| Sinusoid | 29.0 | 30.0 | 30.0 | 3.0 | 6.0 |

IQR = 0 across all cells (deterministic given seed, `random_state=42`).

### Support compactness (per modality)

| World | Audio | Vision | Tactile | Gravity | Force |
|-------|------:|-------:|--------:|--------:|------:|
| Gaussian | 0.234 | 0.117 | 0.938 | 1.000 | 1.000 |
| XOR | 0.234 | 0.117 | 0.938 | 1.000 | 1.000 |
| Sinusoid | 0.219 | 0.120 | 0.938 | 1.000 | 1.000 |

Gravity and Force are degenerate (support = 1.0) because their
intrinsic dimensionality is very low (3 and 6) and the compactness
metric saturates at the unit hypercube edge. This is an expected
audit artefact, not a data issue. Reported here for completeness ;
the §2.4 narrative only uses Audio and Vision.

### Scalar world-level metrics

| Metric | Gaussian | XOR | Sinusoid |
|--------|---------:|----:|---------:|
| Pairwise MI (median) | 0.02923 | 0.02426 | 0.03582 |
| Pairwise MI (IQR) | 0.01777 | 0.00282 | 0.00215 |
| Label-conditional entropy | 1.9228 | 0.9770 | 1.8164 |
| LCE IQR | 0.0279 | 0.0022 | 0.0019 |
| Linear separability | 0.8907 | 0.5021 | 0.8711 |
| Lin-sep IQR | 0.0137 | 0.0342 | 0.0061 |
| Temporal autocorrelation | −0.00174 | +0.01918 | −0.00858 |
| Autocorr IQR | 0.07785 | 0.03344 | 0.02218 |

**Note on IQR of temporal autocorr.** Gaussian's wider IQR (0.078)
reflects the fact that white-noise-like generators produce per-seed
autocorrelation oscillating around 0 within a ±0.04 band ; XOR and
Sinusoid are lower because the generator-side modality transform
damps autocorrelation more uniformly. This is a property of the
*generators*, not the benchmark metric, and does not affect the B-1 /
B-2 / B-3 measurements that average over seeds before thresholding.

## A.3 Reproduction

```bash
cd ~/Projets/bouba_sens
git checkout v0.4.0
uv sync --all-extras
uv run python -m bouba_sens.audit.world_complexity \
  --worlds gaussian xor sinusoid \
  --seeds 5 \
  --out reports/v0.3_critical_validation/world_complexity_audit.json
```

SHA256 of the fresh-clone output should match the MANIFEST entry :
(to be inserted after Clément runs the reproduce script — see §8
reproducibility pipeline).
