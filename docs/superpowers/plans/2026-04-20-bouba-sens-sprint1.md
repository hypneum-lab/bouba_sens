# bouba_sens — Sprint 1 Implementation Plan (Weeks 2-3: WorldSimulator + SensoryWML)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill in the `src/bouba_sens/world/` and `src/bouba_sens/sensory.py` modules scaffolded in Sprint 0. Land three `WorldSimulator` implementations (`GaussianWorld`, `XORWorld`, `SinusoidWorld`) sharing the same latent `z`, plus the `SensoryWML` family (one subclass of `track_w.mlp_wml.MlpWML` per modality). End state: a 5-modality world can be sampled, each modality has a typed encoder, each `SensoryWML.step(x)` returns 64-code neuroletters that go through a shared `GammaThetaMultiplexer` with zero Sprint 2 dependencies.

**Architecture:** Pure PyTorch, nerve-wml `master@77efb4d` (multiplexer now MERGED — no mocking). OQ1 resolved to "shared codebook" per ADR-0001 (Sprint 0 c740864) — this plan bakes that decision in.

**Tech Stack:** Python 3.14, uv, PyTorch ≥ 2.5, pytest + hypothesis + pytest-xdist (parallel seed sweeps), ruff + mypy + pyright, hydra.

**Parent spec:** `docs/superpowers/specs/2026-04-20-bouba-sens-design.md` §3.1 (`WorldSimulator`) and §3.2 (`SensoryWML`).

**Sprint 1 scope:** Tasks 1.1 → 1.10. Sprint 2 (`CrossModalNerve` + `LesionScheduler` + `AdaptationLoop`) is deliberately out of scope and will be planned once Sprint 1 closes with green CI and the three invariants (§1.2 B-1/B-2/B-3) have testable Me3/Me7 harness hooks scaffolded from the real SensoryWML emissions.

**Compute target:** GrosMac for scaffolding + unit tests (lightweight); Studio reserved for training runs from Sprint 3 onward. All Sprint 1 tests must pass under `uv run pytest` on GrosMac.

---

## File structure touched in Sprint 1

```
bouba_sens/
├── src/bouba_sens/
│   ├── world/
│   │   ├── base.py             [Task 1.1]  WorldSample + WorldSimulator
│   │   ├── gaussian.py         [Task 1.2]  GaussianWorld impl
│   │   ├── xor.py              [Task 1.3]  XORWorld impl
│   │   └── sinusoid.py         [Task 1.4]  SinusoidWorld impl
│   ├── sensory.py              [Tasks 1.6-1.8]  SensoryWML + input_proj + step()
│   └── encoders/
│       ├── __init__.py         [Task 1.7]  (new subpackage)
│       ├── audio.py            [Task 1.7]  Conv1D on spectrogram
│       ├── vision.py           [Task 1.7]  Conv2D stack
│       ├── tactile.py          [Task 1.7]  MLP on taxel vector
│       ├── gravity.py          [Task 1.7]  MLP on 3-vector
│       └── force.py            [Task 1.7]  MLP on 6-wrench
├── tests/unit/
│   ├── test_world_base.py      [Task 1.1]
│   ├── test_world_gaussian.py  [Task 1.2]
│   ├── test_world_xor.py       [Task 1.3]
│   ├── test_world_sinusoid.py  [Task 1.4]
│   ├── test_world_orthogonality.py  [Task 1.5]  pairwise MI < 0.2 bit
│   ├── test_sensory_base.py    [Task 1.6]
│   ├── test_encoders.py        [Task 1.7]  5 modality input_proj shape tests
│   └── test_sensory_step.py    [Task 1.8]
├── tests/integration/
│   └── test_five_modality_emission.py  [Task 1.9]
└── CHANGELOG.md                [Task 1.10] release notes for v0.0.2-sprint1
```

---

## Tasks

### Task 1.1 — `WorldSample` + `WorldSimulator` Protocol

- [ ] In `src/bouba_sens/world/base.py`, define the `WorldSample` frozen dataclass with the seven fields from spec §3.1 (`z`, `audio`, `vision`, `tactile`, `gravity`, `force`, `label`).
- [ ] Define the `WorldSimulator` `Protocol` (`runtime_checkable`) with `sample(batch_size, seed) -> WorldSample` and `modality_dims() -> dict[str, tuple[int, ...]]`.
- [ ] Export both from `src/bouba_sens/world/__init__.py`.

**TDD acceptance:**
- `test_world_base.py::test_worldsample_is_frozen_dataclass` — asserts immutability via `FrozenInstanceError`.
- `test_world_base.py::test_worldsimulator_is_runtime_checkable` — dummy class implementing the protocol satisfies `isinstance(..., WorldSimulator)`.

### Task 1.2 — `GaussianWorld`

- [ ] In `src/bouba_sens/world/gaussian.py`, implement `GaussianWorld`: shared latent `z ∈ ℝ^{B×D_z}` sampled from `N(0, I)`, then projected to each modality via a **fixed orthogonal** matrix (Gram-Schmidt-orthogonalised at init, frozen).
- [ ] `modality_dims()` returns: `audio=(128,)`, `vision=(16,16)`, `tactile=(32,)`, `gravity=(3,)`, `force=(6,)`. `D_z = 32`.
- [ ] `label` = `(z[:, 0] > 0).long() + 2 * (z[:, 1] > 0).long()` (4-class). Deterministic from seed.
- [ ] Seeded via local `torch.Generator` (MlpWML convention — never pollute global RNG).

**TDD acceptance:**
- `test_world_gaussian.py::test_sample_shape` — all 7 fields have expected shapes.
- `test_world_gaussian.py::test_same_seed_reproducible` — two `sample(..., seed=42)` calls return identical tensors.
- `test_world_gaussian.py::test_different_seeds_diverge` — seeds 0 vs 1 give different `z`.
- `test_world_gaussian.py::test_label_is_deterministic_from_z` — given `z`, label is fully determined.

### Task 1.3 — `XORWorld`

- [ ] In `src/bouba_sens/world/xor.py`, implement `XORWorld`: latent `z ∈ {−1, +1}^{B×D_z}` (Rademacher sampled), modality projections same orthogonal-factored scheme as `GaussianWorld`.
- [ ] `label` = parity of first 2 z-dims: `(z[:, 0] * z[:, 1] > 0).long()` (2-class).

**TDD acceptance:**
- `test_world_xor.py::test_z_is_rademacher` — all `z` values ∈ {−1, +1}.
- `test_world_xor.py::test_label_matches_parity` — label correctly reflects sign product.

### Task 1.4 — `SinusoidWorld`

- [ ] In `src/bouba_sens/world/sinusoid.py`, implement `SinusoidWorld`: latent `z = [sin(2πt/T), cos(2πt/T), …]` for a batch of `t ∈ [0, T)` sampled uniformly. Modality projections same orthogonal-factored scheme.
- [ ] `label` = 4-bin quantisation of `z[:, 0]` (4-class).

**TDD acceptance:**
- `test_world_sinusoid.py::test_z_on_unit_circle_pairs` — first two dims satisfy `z[:,0]² + z[:,1]² ≈ 1`.

### Task 1.5 — Orthogonality invariant test

- [ ] In `tests/unit/test_world_orthogonality.py`, implement `test_pairwise_mi_below_0.2bit` (mentioned in spec §3.1): for each of the 3 worlds, draw a large batch (`N ≥ 8192`), compute pairwise mutual information (via `sklearn.feature_selection.mutual_info_regression` or equivalent) between every pair of modalities *after excluding their shared-z lineage*, assert MI < 0.2 bit per pair.
- [ ] Parametrise over all 3 worlds via `@pytest.mark.parametrize`.

**TDD acceptance:**
- Test runs in < 10 s on GrosMac (use `pytest-xdist -n 3` for parallel seeds).
- Threshold `< 0.2 bit` matches the spec literal.

### Task 1.6 — `SensoryWML` base class

- [ ] In `src/bouba_sens/sensory.py`, define `Modality = Literal["audio", "vision", "tactile", "gravity", "force"]` and `SensoryWML(MlpWML)` subclass per spec §3.2.
- [ ] Fields: `modality: Modality`, `input_proj: nn.Module` (injected at `__init__`), and a reference to a **shared** `GammaThetaMultiplexer` (constructor accepts `mux: GammaThetaMultiplexer`).
- [ ] Constructor signature: `SensoryWML(id: int, modality: Modality, input_proj: nn.Module, mux: GammaThetaMultiplexer, *, seed: int | None = None)` — id forwarded to `MlpWML`, input_proj drives modality-specific projection, mux is the *shared* multiplexer instance per §2.2 principle 2 (shared 64-code alphabet).

**TDD acceptance:**
- `test_sensory_base.py::test_sensorywml_is_mlpwml_subclass` — MRO includes `MlpWML`.
- `test_sensory_base.py::test_shared_mux_not_duplicated` — two `SensoryWML` instances constructed with the same `mux` share constellation identity (`id(s1.mux.constellation) == id(s2.mux.constellation)`).

### Task 1.7 — Modality-specific encoders

- [ ] New subpackage `src/bouba_sens/encoders/` with 5 modules, each exposing one `nn.Module`:
  - `audio.py::AudioEncoder` — 1-D conv stack on `(B, T_audio=128)` spectrogram → `(B, d_hidden=128)`.
  - `vision.py::VisionEncoder` — 2-D conv stack on `(B, H=16, W=16)` grayscale → `(B, d_hidden=128)`.
  - `tactile.py::TactileEncoder` — MLP on `(B, N_taxels=32)` → `(B, d_hidden=128)`.
  - `gravity.py::GravityEncoder` — MLP on `(B, 3)` → `(B, d_hidden=128)`.
  - `force.py::ForceEncoder` — MLP on `(B, 6)` → `(B, d_hidden=128)`.
- [ ] All encoders use `init_scale=0.1` convention (matches `MlpWML` + `Transducer`).

**TDD acceptance:**
- `test_encoders.py::test_encoder_shapes` parametrised over 5 modalities: input of modality's native shape → output `(B, 128)`.
- `test_encoders.py::test_encoder_preserves_global_rng` — constructor does not pollute `torch.get_rng_state()`.

### Task 1.8 — `SensoryWML.step()` emits neuroletters

- [ ] Implement `SensoryWML.step(x: Tensor) -> Tensor`:
  1. `h = self.input_proj(x)` — modality → `(B, d_hidden)`.
  2. `codes = self.output_head(h).argmax(-1)` — `(B, K=symbols_per_theta)` long.
  3. `carrier = self.mux.forward(codes)` — `(B, T)` float32.
  4. Return `carrier`.
- [ ] Note: `SensoryWML` returns the **carrier** not raw codes, so downstream `CrossModalNerve.fuse` (Sprint 2) consumes the DSP signal directly. The 64-code alphabet is shared via the common `mux.constellation` parameter.

**TDD acceptance:**
- `test_sensory_step.py::test_step_returns_carrier_shape` — output is `(B, mux._t_grid.numel())`.
- `test_sensory_step.py::test_step_end_to_end_differentiable` — `loss = sensory.step(x).pow(2).mean()`; `loss.backward()`; assert `mux.constellation.grad` is non-zero AND `input_proj.*.weight.grad` is non-zero (both paths carry gradient).

### Task 1.9 — Integration smoke test (5 modalities coherent)

- [ ] In `tests/integration/test_five_modality_emission.py`:
  1. Build a `GaussianWorld`, sample a batch (`B=8`).
  2. Build a shared `GammaThetaMultiplexer(seed=0)`.
  3. Build 5 `SensoryWML` instances, one per modality, with their native encoder.
  4. Call `wml.step(sample.<modality>)` for each, collect 5 carriers.
  5. Demodulate each carrier via `mux.demodulate(carrier)`.
  6. Assert shape `(B, K)` long for each, values ∈ `[0, 64)`.
- [ ] Demod returns integers because `hard=True` default — soft mode is Sprint 2 `CrossModalNerve.fuse` path.

**TDD acceptance:**
- Test runs in < 5 s on GrosMac.
- All 5 carriers have identical shape `(8, 175)`.
- All demodulated codes are valid alphabet indices.

### Task 1.10 — Sprint 1 close

- [ ] Update `CHANGELOG.md` with the Sprint 1 summary (added WorldSimulator × 3, SensoryWML + 5 encoders, 8 test files).
- [ ] Commit, push, tag `v0.0.2-sprint1` on `main`.
- [ ] Update `MEMORY.md` index entry for `project_bouba_sens_sprint*` to reflect Sprint 1 closure.

**Acceptance criteria:**
- `uv run pytest` green (all unit + integration tests pass).
- `uv run ruff check src tests` clean.
- `uv run mypy src tests` no new errors.
- `git tag -l | grep v0.0.2-sprint1` returns the tag.

---

## Out of scope (Sprint 2 or later)

- `CrossModalNerve.fuse` + `PlasticityGate` + `AdaptiveCodebook` + `CrossModalTransducer` (Sprint 2).
- `LesionScheduler` + the M2/T3 protocol (Sprint 2).
- `IntegrationHead` + 10-class classifier (Sprint 3).
- `AdaptationLoop` + θ-replay buffer (Sprint 2-3 split per spec).
- Training on Studio (Sprint 3+).
- Metric harness (Me1/Me2/Me3/Me6/Me7/Me8/Me9) (Sprint 3).

## Risks / checkpoints

- **R-sprint1-1**: PyTorch 2.11 on Studio vs ≥ 2.5 local — pin nerve-wml via `[tool.uv.sources]` local path, lock `uv.lock` commit so Studio reproducibility matches GrosMac. Verified 2026-04-20 via Studio smoke test (`T=175 samples`).
- **R-sprint1-2**: Orthogonality test (Task 1.5) may fail if the Gram-Schmidt orthogonalisation loses precision in fp32. Fallback: compute the projections in fp64 at init, cast weights to fp32 after.
- **R-sprint1-3**: The 5 modality encoders may over-parametrise for the 4-class Gaussian label. Keep `d_hidden=128` small enough that no single encoder can memorise — OQ1 resolution (shared codebook) depends on this.

## Parent spec sections cited

- §1.2 three invariants B-1/B-2/B-3 — Sprint 1 lays down the emission path these invariants eventually operate on.
- §2.1 block diagram — Sprint 1 ships the left half (WorldSimulator → SensoryWMLs → carriers). Sprint 2 wires the right half.
- §2.2 principle 2 — shared 64-code alphabet via a single `GammaThetaMultiplexer` instance passed to all SensoryWMLs (per OQ1 ADR-0001).
- §3.1 + §3.2 — public interfaces implemented literally.

## Related context

- nerve-wml `master@77efb4d` — multiplexer shipped, `py.typed` covered, `from track_p import GammaThetaMultiplexer, GammaThetaConfig, AWGN` verified on both GrosMac and Studio (2026-04-20).
- `docs/adr/0001-codebook-sharing.md` — OQ1 resolution: shared codebook wins (Sprint 0 spike `c740864`).
