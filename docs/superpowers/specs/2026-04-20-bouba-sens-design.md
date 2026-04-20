---
title: "bouba_sens — Cross-Modal Plasticity Benchmark"
version: "0.1 design draft"
date: "2026-04-20"
author: "Clément Saillant (Hypneum Lab)"
status: "Design — awaiting user review before writing-plans transition"
license: "MIT"
repo_target: "github.com/hypneum-lab/bouba_sens (fallback: genial-lab/bouba_sens)"
parent_framework: "Hypneum Lab / GENIAL framework"
depends_on: "nerve-wml >=1.1.4,<1.2"
---

# bouba_sens — A Cross-Modal Plasticity Benchmark

> *"A blind person hears more. We want to know why — and whether we can make a model do the same."*

## 0. Executive summary

`bouba_sens` is a **benchmark-first** research repository under the Hypneum Lab umbrella that studies **cross-modal plasticity** in artificial neural systems. Inspiration: human neuroscience of congenital and late blindness — occipital cortex recruited for audio (Amedi 2007), heightened auditory/tactile sensitivity (Merabet 2010), asymmetric gain/cost of sensory loss (Heimler 2020).

The benchmark exposes a 5-modality agent (audio, vision, tactile, gravity, force) to a two-arm **lesion protocol** (congenital vs late-acquired) with **progressive SNR degradation**, then measures how the surviving modalities reorganise to compensate. It extends the `nerve-wml` protocol (neuroletters, γ/θ multiplexing) with plasticity machinery: adaptive gating, adaptive codebook expansion, and learnable cross-modal transducers.

Three gaps in the current literature are addressed in a single protocol:
1. **No benchmark controls lesion timing** (congenital vs late).
2. **No benchmark treats gravity/force as first-class sensory modalities** alongside vision/audio/tactile.
3. **No architecture couples sound-symbolism (bouba/kiki) with biologically-plausible plastic multimodal learning.**

Phase B (synthetic abstract) targets a publishable v1.0 in ~6 months; Phase A (embodied physics simulation) is a v2.0 satellite ~12 months out.

---

## 1. Vision & contribution

### 1.1 Scientific thesis

Cross-modal plasticity — the reorganisation of neural resources when a sensory channel is lost or degraded — is a **central** but **computationally under-specified** phenomenon. Current multimodal ML benchmarks (MultiBench, ImageBind-eval, McKinzie 2023) test *robustness* to missing modalities but ignore the **developmental timing** and the **information-theoretic dynamics** of compensation.

`bouba_sens` posits that three invariants *should be observable* in any architecture that genuinely models cross-modal plasticity, and builds a benchmark that *empirically tests* them.

### 1.2 Three testable invariants

| Id | Statement | Operationalisation |
|----|-----------|--------------------|
| **B-1** | **Congenital gap**: lesion applied pre-training yields better asymptotic adaptation than lesion applied post-convergence, at the floor SNR of the lesion schedule. | `Me7 = perf_T1 − perf_T2 > 0.05` at SNR_floor, bootstrap IC 95 %. |
| **B-2** | **Cross-modal MI migration**: after lesion, mutual information between surviving-modality neuroletters and the target label *increases significantly*, demonstrating informational compensation (not merely weight reallocation). | `Me3_post(channel ≠ lesioned) − Me3_pre > 0.10 bit`. |
| **B-3** | **Perceptive/proprioceptive asymmetry**: losing exteroceptive modalities (audio/vision/tactile) triggers a quantitatively different plastic response than losing interoceptive modalities (gravity/force). | `Me6` asymmetry matrix max absolute off-diagonal `> 0.02`, with sign structure reproducible across seeds. |

Each invariant has a **corresponding empirical test** (see §7.2) enforced in CI nightly — a regression in B-1/B-2/B-3 blocks merges.

### 1.3 Target venue

Paper-phare cible: **TMLR** (accept-or-reject-with-full-review, fits benchmark contributions) or **NeurIPS Datasets & Benchmarks track**. Fallback: arXiv + Zenodo DOI release.

### 1.4 Paper outline (to be elaborated by writing-plans)

1. Cross-modal plasticity in humans — short neuroscience primer.
2. Gaps in current multimodal ML benchmarks.
3. `bouba_sens` protocol: modalities, lesions, plasticity machinery.
4. Three invariants B-1/B-2/B-3 and their operationalisation.
5. Experimental results on 3 synthetic worlds + bouba/kiki tasks.
6. Discussion: congenital gap quantified, asymmetry index, MI migration.
7. Related work (50 references, see §12).
8. Limitations & open questions.

---

## 2. Architecture (Top-1 + P5)

### 2.1 Block diagram

```
                            ┌────────────────────────┐
                            │   WorldSimulator        │
                            │  (shared latent z_t)    │
                            └──────────┬─────────────┘
                                       │ projects to 5 modalities
         ┌──────────┬──────────┬───────┴─────┬──────────┬──────────┐
         ▼          ▼          ▼             ▼          ▼          ▼
    ┌────────┐ ┌────────┐ ┌────────┐    ┌────────┐ ┌────────┐
    │  AUDIO │ │ VISION │ │TACTILE │    │GRAVITY │ │ FORCE  │
    │ SensoryWML × 5 (subclass of track_w.mlp_wml.MlpWML)        │
    └────┬───┘ └────┬───┘ └────┬───┘    └────┬───┘ └────┬───┘
         │ γ/θ-multiplexed neuroletters (64-code shared alphabet) │
         └──────────┬──────────┬─────────────┬──────────┬──────────┘
                    ▼          ▼             ▼          ▼
                ┌───────────────────────────────────────────┐
                │  CrossModalNerve  (extends nerve_core.protocols.Nerve) │
                │  ├─ PlasticityGate      (P1 — channel α)  │
                │  ├─ AdaptiveCodebook    (P2 — code shift) │
                │  └─ CrossModalTransducer (P3 — subst.)    │
                └──────────────────┬────────────────────────┘
                                   ▼
                         ┌─────────────────────┐
                         │  IntegrationHead    │
                         │  (task-specific)    │
                         └──────────┬──────────┘
                                    ▼
                                action / label ŷ
                                    │
                                    ▼
              θ-error (6 Hz) → AdaptationLoop buffer → P1/P2/P3 updates
```

### 2.2 Five design principles

1. **Sensory specialisation at the edges.** Each `SensoryWML` is a sub-class of `track_w.mlp_wml.MlpWML` with a modality-typed `input_proj`. Training sees them as 5 distinct substrates; the taxonomy "exteroceptive vs interoceptive" emerges from the task.
2. **Shared 64-code alphabet.** Controlled violation of `nerve-wml` invariant **N-5** (local codebook per WML) — documented and scoped to this project. A `CodebookAligner` is *not* used in v0.1; if open question OQ1 resolves toward "local codebooks", an aligner is added in v0.2.
3. **CrossModalNerve as plastic router.** Three orthogonal mechanisms (P1 gating, P2 codebook, P3 transducers) live inside the nerve — not inside the WMLs — to keep sensory cortices modality-local and compensation global.
4. **Lesion injected between Simulator and WMLs.** Model code is agnostic to lesion state; `LesionScheduler` intercepts samples before `SensoryWML.step`. Consequence: the exact same model binary is reusable across intact, lesioned, and recovered regimes.
5. **Online adaptation via θ-replay buffer.** Post-lesion, backprop continues on a small replay buffer driven by θ-error signals (6 Hz rhythm inherited from nerve-wml). Mitigates catastrophic forgetting in T2 arm (risk R8).

### 2.3 Extension axes deliberately excluded from v0.1

- Hebbian / STDP plasticity (P4): deferred to v2 "biologically plausible" satellite.
- Predictive coding local rules: same deferral.
- Real neuromorphic hardware deployment (Loihi/Akida): out of scope.
- Real robot hardware: out of scope until v2.0 Phase A validates via simulation.

---

## 3. Components & public interfaces

All public APIs use Python 3.12+ syntax, `from __future__ import annotations`, frozen dataclasses, and strict typing (mypy + pyright on pre-commit).

### 3.1 WorldSimulator (§3.1)

```python
from dataclasses import dataclass
from typing import Protocol
from torch import Tensor

@dataclass(frozen=True)
class WorldSample:
    z:       Tensor   # shared latent, (B, D_z)
    audio:   Tensor   # (B, T_audio)   1-D spectrogram
    vision:  Tensor   # (B, H, W)      grayscale
    tactile: Tensor   # (B, N_taxels)
    gravity: Tensor   # (B, 3)         normalized g-vector
    force:   Tensor   # (B, 6)         wrench (3F + 3τ)
    label:   Tensor   # (B,)           task target

class WorldSimulator(Protocol):
    def sample(self, batch_size: int, seed: int) -> WorldSample: ...
    def modality_dims(self) -> dict[str, tuple[int, ...]]: ...
```

v0.1 implementations: `GaussianWorld`, `XORWorld`, `SinusoidWorld` — all sharing a single latent `z`, projecting to 5 modalities via orthogonally-factorised maps (enforced by construction; validated by unit test `test_pairwise_mi_below_0.2bit`).

### 3.2 SensoryWML

```python
from track_w.mlp_wml import MlpWML
from typing import Literal

Modality = Literal["audio", "vision", "tactile", "gravity", "force"]

class SensoryWML(MlpWML):
    modality: Modality
    input_proj: nn.Module   # modality-specific encoder

    def step(self, x: Tensor) -> NeuroLetters: ...   # (B, K_letters=64)
```

### 3.3 CrossModalNerve

```python
from nerve_core.protocols import Nerve

class CrossModalNerve(Nerve):
    gates:       PlasticityGate                               # P1
    codebook:    AdaptiveCodebook                             # P2
    transducers: dict[tuple[Modality, Modality], CrossModalTransducer]  # P3

    def fuse(self, letters: dict[Modality, NeuroLetters]) -> NeuroLetters: ...
    def on_lesion(self, modality: Modality, snr_db: float) -> None: ...
    def migration_stats(self) -> MigrationReport: ...
```

### 3.4 LesionScheduler

```python
@dataclass(frozen=True)
class LesionSpec:
    modality: Modality
    mode:     Literal["M1", "M2", "M3", "M4"]
    timing:   Literal["T1", "T2", "T3", "T4"]
    schedule: Callable[[int], float]   # step → SNR_dB

class LesionScheduler:
    def apply(self, sample: WorldSample, step: int) -> WorldSample: ...
```

### 3.5 IntegrationHead

```python
class IntegrationHead(nn.Module):
    def forward(self, fused: NeuroLetters) -> Tensor: ...   # → logits (B, n_classes)
```
Task-specific. v0.1 default: 10-class linear classifier on `WorldSample.label`. Alternative tasks (auto-encoding on `z`, contrastive alignment) parked as OQ2.

### 3.6 AdaptationLoop

```python
class AdaptationLoop:
    def pretrain(self, steps: int) -> Checkpoint: ...                 # Phase 1 (T2 only)
    def lesion_phase(self, lesion: LesionSpec, steps: int) -> AdaptationReport: ...
    def eval_congenital(self) -> EvalReport: ...                      # T1
    def eval_late(self, ckpt_intact: Checkpoint) -> EvalReport: ...   # T2
```
Owns the θ-replay buffer (FIFO, size 1 024 in v0.1 — OQ5), triggers P1/P2/P3 updates per step, logs to `curves.parquet` + `migration.parquet`.

### 3.7 Metric protocol

```python
class Metric(Protocol):
    name: str
    def update(self, report: EvalReport) -> None: ...
    def compute(self) -> dict[str, float]: ...
```
Seven v0.1 implementations: Me1, Me2, Me3, Me6, Me7, Me8, Me9 (see §5.2).

### 3.8 PlasticityRule protocol (extension hook)

```python
class PlasticityRule(Protocol):
    def step(self, nerve: CrossModalNerve, error: Tensor) -> None: ...
```
v0.1 ships only `BackpropRule`. Hebbian/STDP and predictive-coding rules land in v2 satellite (see OQ4).

---

## 4. Lesion protocol (M2 + T3)

### 4.1 SNR schedule (mode M2)

```
SNR(t) = max(SNR_floor, SNR_init + (SNR_floor − SNR_init) · min(1, t / K))
```

- v0.1 defaults: `SNR_init = +20 dB`, `SNR_floor = −20 dB`, `K = 5 000 steps`.
- Noise is additive Gaussian, scaled to local signal magnitude (relative SNR, not absolute) to preserve input statistics.

### 4.2 Two arms (timing T3)

| Arm | Phase 1 — Intact | Phase 2 — Lesion | Phase 3 — Evaluation |
|-----|------------------|-------------------|-----------------------|
| **T1 congenital** | *(skipped — no intact pre-training)* | `10 000` steps, lesion active from t=0 | All channels restored for eval; ablation control applied |
| **T2 late-acquired** | `10 000` steps, all channels clean | `5 000` steps, `LesionScheduler` active, full recovery phase | Same eval suite as T1 |

Each arm produces an `AdaptationReport`: learning curves, per-step MI(channel;label), codebook occupancy trajectory, transducer activation rates.

### 4.3 Adaptation dynamics

During Phase 2, standard backprop on:
- **PlasticityGate**: channel-wise attentional weights `α ∈ ℝ⁵` (softmax-normalised).
- **AdaptiveCodebook**: soft-assignment projection (Gumbel-softmax, temperature schedule 1.0 → 0.1).
- **CrossModalTransducer**: 20 paired MLPs (5×4 directed edges), activated only when `gates[source] < θ = 0.1` *and* `gates[target] > 0.3`.

### 4.4 Baselines (Me8)

Each (arm × modality × SNR) cell run in three regimes:
- **(a) intact**: no lesion — performance ceiling.
- **(b) robust-only**: lesion active, P1/P2/P3 **frozen** — plasticity-free floor.
- **(c) random-rewire**: P1/P2/P3 replaced by fixed random weights — chance control.

### 4.5 Statistical replication (Me9)

- `5 seeds × 5 modalities × 2 timings × 3 SNR points (floor, −10, +10) = 150 runs/version`.
- IC 95 % via bootstrap (1 000 resamples) per cell.
- All runs archived: `AdaptationReport` pickles + `metrics.parquet` + `migration.parquet`.

---

## 5. Evaluation harness & metrics

### 5.1 Run layout

```
runs/{date}_{config}_{seed}/
├── config.yaml           # LesionSpec + architecture hparams + schedule
├── metadata.json         # git SHA, python version, hardware, wall-time
├── intact_ckpt.pt        # Phase 1 (T2 only)
├── adapted_ckpt.pt       # Phase 2 final
├── curves.parquet        # per-step loss, accuracy, MI(code_*;label), …
├── migration.parquet     # per-step codebook distribution, gate α, transducer activations
├── eval_report.json      # Me1..Me9 summary values
└── stdout.log / stderr.log
```

### 5.2 Metric catalogue

| Id | Metric | Definition / estimator | Window |
|----|--------|------------------------|--------|
| **Me1** | Accuracy post-adaptation | `mean(argmax(logits) == label)` on 1 000-sample hold-out | end Phase 2 |
| **Me2** | Recovery curve AUC | `trapz(accuracy_t)` over Phase 2 steps | full Phase 2 |
| **Me3** | MI(code_channel; label) | Kraskov kNN estimator over discretised (Voronoi 64-cell) code space | pre / t=1 000 / end |
| **Me6** | Asymmetry index | 5×5 matrix `[i,j] = perf(lesion=i, query=j) − perf(lesion=j, query=i)` | final |
| **Me7** | Congenital gap | `perf_T1 − perf_T2` at SNR_floor, per modality | final |
| **Me8** | Baselines | intact / robust-only / random-rewire | final |
| **Me9** | Seed variance | bootstrap IC 95 % over ≥ 5 seeds | all cells |

Optional v0.2: Me4 (codebook KL shift), Me5 (transducer activation rate).

### 5.3 Command-line interface

```bash
bouba-sens sim      --world gaussian --size 100k --out data/world_v1.parquet
bouba-sens train    --config configs/v0.1_intact.yaml --seed 0
bouba-sens lesion   --ckpt runs/intact_seed0 --spec configs/t2_audio_m2.yaml
bouba-sens eval     --run  runs/2026-04-25_v0.1_seed0 --metrics Me1,Me2,Me3,Me6,Me7
bouba-sens aggregate --glob "runs/v0.1_*" --out reports/v0.1_summary.html
```

### 5.4 HTML report (`report.py`)

Jinja2 template producing `reports/v0.1_summary.html` with:
- Me1/Me2 recap table per (config × modality)
- Me6 asymmetry matrix as interactive heatmap (plotly)
- Me2 recovery curves overlaid for T1 vs T2 per modality
- Me7 congenital gap bar chart
- Me9 IC 95 % table (all metrics, all cells)

---

## 6. Repository structure & stack

### 6.1 Tree

```
bouba_sens/
├── pyproject.toml              # uv + setuptools-scm; depends on nerve-wml >=1.1.4,<1.2
├── README.md                   # thesis + quickstart + 8 priority refs
├── LICENSE                     # MIT
├── CITATION.cff                # aligned with Hypneum Lab template
├── SECURITY.md                 # aligned with Hypneum Lab template
├── .github/workflows/
│   ├── ci.yml                  # lint + mypy + pytest + 1-seed smoke test
│   └── full-benchmark.yml      # nightly 5-seed run (manual trigger)
├── src/bouba_sens/
│   ├── __init__.py
│   ├── world/                  # §3.1
│   │   ├── base.py
│   │   ├── gaussian.py
│   │   ├── xor.py
│   │   └── sinusoid.py
│   ├── sensory.py              # SensoryWML + modality encoders
│   ├── nerve.py                # CrossModalNerve, PlasticityGate, AdaptiveCodebook, Transducer
│   ├── lesion.py               # LesionScheduler, LesionSpec, SNR schedules
│   ├── head.py                 # IntegrationHead
│   ├── loop.py                 # AdaptationLoop, θ-replay buffer
│   ├── metrics/                # §5.2
│   │   ├── performance.py      # Me1, Me2
│   │   ├── mi_migration.py     # Me3 (+ Me4, Me5 optional)
│   │   ├── asymmetry.py        # Me6
│   │   ├── congenital.py       # Me7
│   │   └── baselines.py        # Me8
│   ├── cli.py                  # typer-based CLI
│   └── report.py               # Jinja2 HTML report generator
├── configs/
│   ├── v0.1_intact.yaml
│   ├── t1_audio_m2.yaml … t2_force_m2.yaml   # 5 × 2 = 10 base configs
│   └── v0.1_full_grid.yaml
├── tests/
│   ├── unit/                   # 1:1 with src/ modules
│   ├── property/               # hypothesis — B-1/B-2/B-3 contracts
│   ├── integration/            # end-to-end smoke (1 seed, 100 steps)
│   └── empirical/              # asserts on real run results (nightly CI)
├── docs/
│   ├── superpowers/specs/2026-04-20-bouba-sens-design.md    ← this file
│   ├── superpowers/plans/                                    ← writing-plans output
│   ├── architecture.md
│   ├── protocol.md
│   └── references.bib
├── scripts/
│   ├── repro_v0.1.sh           # reproduces all v0.1 results end-to-end
│   └── validate_refs.py        # revalidates 5 flagged refs (Loakman, Peeters, Genesis, V-JEPA, Ji 2024)
└── papers/
    └── paper1/                 # LaTeX sources, figures, refs.bib
```

### 6.2 Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Python | 3.14 | Hypneum Lab default |
| Packaging | uv + setuptools-scm | Hypneum Lab default, fast |
| DL | PyTorch ≥ 2.5 | Direct compatibility with nerve-wml transducers |
| Typing | mypy strict + pyright pre-commit | API discipline |
| Testing | pytest + hypothesis + pytest-xdist | Unit + property + parallel |
| Lint/format | ruff + ruff-format | Speed + Hypneum alignment |
| CLI | typer + rich | UX consistent with nerve-wml CLI |
| Storage | pyarrow/parquet + orjson | Fast, portable, diffable |
| Viz | matplotlib + plotly + jinja2 | Static + interactive HTML |
| Config | hydra | Reproducibility |
| CI | GitHub Actions | Hypneum infra |

### 6.3 Required exposure from `nerve-wml`

Verified 2026-04-20 against local clone v0.1.0 at `~/Documents/Projets/nerve-wml`.
The package exposes 7 top-level modules (`nerve_core`, `track_p`, `track_w`, `bridge`,
`harness`, `interpret`, `neuromorphic`) — **not** a unified `nerve_wml.*` namespace.
Installed as a local path source (`[tool.uv.sources]` in `pyproject.toml`):

```toml
[tool.uv.sources]
nerve-wml = { path = "../nerve-wml" }
```

| Symbol | Status | Notes |
|--------|--------|-------|
| `nerve_core.protocols.Nerve` | ✓ runtime-checkable Protocol — verified | exposes `GAMMA_HZ`, `THETA_HZ`, `ALPHABET_SIZE` constants |
| `nerve_core.neuroletter.Neuroletter` | ✓ frozen dataclass — verified (singular, not `NeuroLetters`) | transport metadata only; `src`/`dst`/`timestamp` are NOT carried on the γ/θ signal |
| `track_w.mlp_wml.MlpWML` | ✓ `nn.Module` subclass — verified | — |
| `track_p.transducer.Transducer` | ✓ `nn.Module` subclass — verified (generic, not `CrossSubstrateTransducer`) | shape convention `[B] long → [B] long`; `hard: bool`, `tau: float` switch |
| `track_p.multiplexer.GammaThetaMultiplexer` + `GammaThetaConfig` | 🟡 **PENDING** — draft PR tracking in [nerve-wml#1](https://github.com/hypneum-lab/nerve-wml/issues/1) | see contract below |

**Revised γ/θ multiplexer contract** (agreed in nerve-wml#1 design review, 2026-04-20):

- `forward(codes: Tensor[B, K] long, *, theta_phase_offset: float = 0.0) → carrier: Tensor[B, T] float32` with `T = sample_rate_hz // theta_hz`
- `demodulate(carrier: Tensor[B, T], *, hard: bool = True) → Tensor[B, K] long` (Gumbel-softmax when `hard=False`, matches `Transducer` convention)
- Constants sourced from `Nerve.GAMMA_HZ / THETA_HZ / ALPHABET_SIZE` — no duplication
- Config object: `@dataclass(frozen=True) GammaThetaConfig` (6 hyperparams)
- Gaussian PAC envelope (differentiable, physiologically plausible per Harris & Gong 2026)
- Role encoding: out-of-band second channel (preserves full 64-code alphabet; deferred to bouba_sens v0.2)
- No `Neuroletter` round-trip — the multiplexer operates on code tensors, not transport objects

**Sprint 1 blocker tracking**: the γ/θ multiplexer is the only remaining API gap. The 4 verified symbols unblock `SensoryWML`, `CrossModalNerve` structural scaffolding, and protocol-based testing. Mocking the multiplexer behind a local `Protocol` is acceptable in Sprint 1 until the draft PR lands.

### 6.4 Location & GitHub

- Local: `~/Documents/Projets/bouba_sens`
- GitHub: **`hypneum-lab/bouba_sens`** — public, `main`, pushed 2026-04-20 (commit `f9c65df`).
- Visibility: public from first commit.
- Default branch: `main`.
- License: MIT.
- Topics: `cross-modal-plasticity`, `multimodal-benchmark`, `embodied-ai`, `hypneum-lab`, `pytorch`.

---

## 7. Phases, testing, risks, open questions

### 7.1 Phases & milestones

| Phase | Duration | Scope | Go/no-go criterion |
|-------|----------|-------|---------------------|
| **v0.1** — MVP abstract | 6 weeks | 5 channels Phase-B synthetic, Top-1+P5, M2+T3, Me1+Me2+Me8+Me9, GaussianWorld only | Non-trivial recovery curve + Δcongenital > 5 % on ≥ 1 modality |
| **v0.2** — Full benchmark | +10 weeks | 150 runs (5×2×3×5), add Me3+Me6+Me7, 3 worlds, HTML report | B-1 and B-2 hold with IC 95 % ; OSF pre-registration |
| **v1.0** — Paper + release | +8 weeks | Bouba/kiki tasks, M1+T1 / M4+T4 appendix extensions, paper TMLR draft, Zenodo DOI, tag `v1.0` | Paper submitted + CI full-benchmark green + internal peer review pass |
| **v2.0** — Embodied (Phase A) | +24 weeks | MuJoCo or Genesis, 1 morphology, invariants transfer, satellite paper | Measurable Δ between v1.0 and v2.0 on ≥ 2 invariants |

### 7.2 Testing strategy — 4 levels

1. **Unit** (`tests/unit/`, > 90 % coverage). Each class in §3 tested in isolation with mocks for `WorldSimulator` and `nerve_wml` internals.
2. **Property-based** (`tests/property/`, hypothesis). Architectural invariants:
   - `CrossModalNerve.fuse(letters)` always returns shape `(B, 64)` regardless of how many channels are lesioned.
   - `LesionScheduler` commutes with batch concatenation.
   - `PlasticityGate` weights always sum to 1 post-softmax.
3. **Integration smoke** (`tests/integration/`, < 2 min). End-to-end pipeline on 100 steps, 1 seed, tiny config — verifies nothing blows up.
4. **Empirical** (`tests/empirical/`, nightly GitHub Actions).
   - `test_B1_congenital_gap()` — `Me7 > 0.05` on audio M2 lesion.
   - `test_B2_mi_migration()` — `Me3_post − Me3_pre > 0.10 bit` on healthy-channel code.
   - `test_B3_asymmetry_nonzero()` — `np.abs(Me6_matrix).max() > 0.02`.
   - Failures block merges (scientific regression gate).

### 7.3 Risks & mitigations

| # | Risk | P | Impact | Mitigation |
|---|------|:-:|:------:|------------|
| R1 | nerve-wml v1.2 breaks interfaces before bouba_sens v0.2 | M | high | Pin 1.1.4, joint issue tracker, CI compat matrix |
| R2 | M2+T3 protocol reveals **no** congenital gap (B-1 fail) | M | very high | Fallback to `M1+T1+T2+T3` + finer SNR grid; OSF pre-reg held until after v0.1 |
| R3 | 5 modalities explode compute (> 48 h/config on KXKM-AI) | H | medium | MVP on 3 modalities (audio, vision, gravity) then scale-up; bench on CILS first |
| R4 | Gravity/force signals too correlated → no informative gap | M | high | World simulator enforces orthogonal z-factorisation; pairwise MI < 0.2 bit in unit tests |
| R5 | Bouba/kiki stimuli (Peeters 2023 unverified) unavailable | M | medium | Generate synthetic stimuli + 3-pilot MTurk validation, budget +500 € |
| R6 | TMLR rejects paper | M | low | Plan B: NeurIPS D&B. Plan C: arXiv + Zenodo DOI |
| R7 | `hypneum-lab` rename not completed at `gh repo create` time | H | low | Create under `genial-lab`, transfer post-rename via `gh repo transfer` (lossless) |
| R8 | Catastrophic forgetting during Phase 2 (T2) | H | high | θ-replay buffer + EWC-like regulariser during `lesion_phase` (Kirkpatrick 2017) |

### 7.4 Open questions (deliberately unresolved, to trigger spikes)

- **OQ1** — Shared 64-code alphabet across SensoryWMLs, or local codebooks + `CodebookAligner` in the nerve? *Resolve week 2 via 1-day spike.*
- **OQ2** — v0.1 IntegrationHead task: 10-class classification on `label`, or auto-encoding / predicting `z`? *Resolve at end of OQ1 spike.*
- **OQ3** — Transducers pre-trained during Phase 1 (warm start), or from scratch at lesion onset? *Depends on whether R2 fallback triggers.*
- **OQ4** — Ship a `PlasticityRule` hook in v0.1 (future-proof for Hebbian) or defer to v2? *Default: defer.*
- **OQ5** — `θ-replay buffer`: FIFO or prioritised (Schaul 2015)? *Default FIFO in v0.1; prioritise in v0.2 if R8 materialises.*

---

## 8. Transition to `writing-plans`

Once this design is user-approved, the `writing-plans` skill produces a sprint-level implementation plan. Expected shape:

- **Sprint 0 (week 1)**: scaffolding, CI pipeline, `pyproject.toml`, verified `nerve-wml` imports, OQ1 spike.
- **Sprint 1 (weeks 2-3)**: `WorldSimulator.gaussian`, `SensoryWML` + modality encoders, associated unit tests.
- **Sprint 2 (weeks 4-5)**: `CrossModalNerve` + `LesionScheduler` + `AdaptationLoop` + integration smoke.
- **Sprint 3 (week 6)**: `Me1`/`Me2`/`Me8`/`Me9` + `cli.py` + first end-to-end v0.1 run; v0.1 go/no-go decision.

---

## 9. References

### 9.1 Priority references to cite in README + paper v0.1

1. **Amedi 2007** — DOI `10.1038/nn1912`. Biological justification for meta-modality.
2. **Heimler 2020** — DOI `10.1016/j.neuroscience.2019.12.048`. Source of the asymmetry-index framing.
3. **Röder 2021** — DOI `10.1073/pnas.2004321118`. Empirical basis for the congenital gap.
4. **Alper 2023** — arXiv `2310.16781`. Bouba/kiki in VLMs — mandatory VLM baseline.
5. **Girdhar 2023 (ImageBind)** — arXiv `2305.05665`. Reference architecture to beat.
6. **Ma 2022** — arXiv `2204.05454`. Missing-modality methodology.
7. **Liang 2022 (MultiBench)** — arXiv `2107.07502`. Benchmark comparator.
8. **Keller 2018** — DOI `10.1016/j.neuron.2018.10.003`. Predictive-coding skeleton of the nerve.

### 9.2 Full bibliography

See `docs/references.bib` (to be populated in Sprint 0). Six thematic buckets:

- **Neuroscience of blindness & cross-modal plasticity**: Amedi 2007, Striem-Amit 2012, Merabet 2010, Bavelier 2002, Röder 2021, Sadato 1996, Thinus-Blanc 1997, Heimler 2020.
- **Bouba/kiki & sound-symbolism**: Ramachandran 2001, Sidhu 2018, Ćwiek 2022, Alper 2023, Loakman 2024 *(revalidate)*, Peeters 2023 *(revalidate)*, Nielsen 2013, Dingemanse 2015.
- **Multi-modal learning & cross-modal transfer**: Jaegle 2021 (Perceiver), Guzhov 2022 (AudioCLIP), Girdhar 2023 (ImageBind), Zhang 2023 (Meta-Transformer), Ma 2022, Recasens 2023 (Zorro), Han 2023, Liang 2024.
- **Embodied/sensorimotor AI & active inference**: Friston 2010, O'Regan 2001, Brohan 2023 (RT-2), Driess 2023 (PaLM-E), Makoviychuk 2021 (Isaac Gym), Genesis 2024 *(revalidate)*, Todorov 2012 (MuJoCo), Sferrazza 2024, Yang 2024 (V-JEPA) *(revalidate)*.
- **Neuromorphic & biologically plausible plasticity**: Bi 1998, Rao 1999, Keller 2018, Millidge 2022, Payeur 2021, Eshraghian 2023, Ji 2024 *(revalidate)*, Hoel 2021.
- **Benchmarks & evaluation**: Goyal 2017, Liang 2022, Hendrycks 2019 (ImageNet-C), McKinzie 2023, Yeh 2024, Olsson 2022, Conmy 2023, Belrose 2023.

Flagged for revalidation before first commit citing them: Loakman 2024, Peeters 2023, Genesis 2024, Yang 2024 (V-JEPA), Ji 2024. Script `scripts/validate_refs.py` to be written in Sprint 0.

### 9.3 Research gaps addressed (one-line summary)

1. **No benchmark controls lesion timing** → addressed by M2+T3 arms (§4.2) and invariant B-1 (§1.2).
2. **Gravity/force under-represented as first-class modalities** → addressed by 5-modality Top-1 architecture (§2) and invariant B-3 (§1.2).
3. **Weak link between sound-symbolism and plastic multimodal architectures** → addressed by bouba/kiki tasks in v0.2 + adaptive codebook (§2.2 principle 2, §3.3).

---

## 10. Changelog

| Date | Version | Change |
|------|---------|--------|
| 2026-04-20 | 0.1-draft | Initial design, output of `superpowers:brainstorming`. Awaits user review and `writing-plans` transition. |
