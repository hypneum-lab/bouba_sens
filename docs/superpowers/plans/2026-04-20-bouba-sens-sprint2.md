# bouba_sens — Sprint 2 Implementation Plan (Weeks 4-5: CrossModalNerve + Lesion + Adaptation)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the Sprint 0 skeletons `nerve.py`, `lesion.py`, `head.py`, `loop.py`. Land `CrossModalNerve` (P1 gate + P2 codebook + P3 transducers + `fuse`), the M2/T3 lesion protocol, a minimal `IntegrationHead`, and an `AdaptationLoop` that can run Phase 1 (pretrain) + Phase 2 (lesion) for 100 steps each without NaN. End state: the right half of the spec §2.1 block diagram is wired and the integration smoke demonstrates that lesion-driven backprop reaches every plasticity mechanism. Sprint 3 scope (metrics harness + CLI + v0.1 full run) stays out.

**Architecture:** Pure PyTorch, reuses Sprint 1's `SensoryWML.step()` carrier output as fuse input. nerve-wml `master@77efb4d` multiplexer + soft-demod (`hard=False`, Gumbel-softmax) provide the gradient-flowing channel the θ-replay loss needs.

**Tech Stack:** Python 3.14, uv, PyTorch ≥ 2.5, pytest + hypothesis + pytest-xdist, ruff + mypy + pyright, hydra. No new runtime deps.

**Parent spec:** `docs/superpowers/specs/2026-04-20-bouba-sens-design.md` §3.3–§3.6 + §4.

**Sprint 2 scope:** Tasks 2.1 → 2.12. Sprint 3 (Me*/CLI/v0.1 run) plan will be written after Sprint 2 closes.

**Compute target:** GrosMac for scaffolding + unit tests. Studio only for the Task 2.11 smoke once the pipeline clears unit gates (100-step Phase 1 + 100-step Phase 2 is cheap enough for GrosMac, but Studio is the canonical reproducibility machine).

---

## File structure touched in Sprint 2

```
bouba_sens/
├── src/bouba_sens/
│   ├── nerve.py                   [Tasks 2.1-2.5]
│   ├── lesion.py                  [Tasks 2.6-2.7]
│   ├── head.py                    [Task 2.8]
│   ├── loop.py                    [Tasks 2.9-2.10]
│   └── __init__.py                [Task 2.12]  bump to v0.0.3 + re-exports
├── tests/unit/
│   ├── test_nerve_plasticity_gate.py   [Task 2.1]
│   ├── test_nerve_adaptive_codebook.py [Task 2.2]
│   ├── test_nerve_transducer.py        [Task 2.3]
│   ├── test_nerve_fuse.py              [Tasks 2.4-2.5]
│   ├── test_lesion_spec.py             [Task 2.6]
│   ├── test_lesion_scheduler.py        [Task 2.7]
│   ├── test_head.py                    [Task 2.8]
│   └── test_loop.py                    [Tasks 2.9-2.10]
├── tests/integration/
│   └── test_phase1_phase2_smoke.py     [Task 2.11]
└── CHANGELOG.md                   [Task 2.12]  v0.0.3-sprint2 entry
```

---

## Tasks

### Task 2.1 — PlasticityGate (P1)

- [ ] In `src/bouba_sens/nerve.py`, implement `PlasticityGate(nn.Module)` with a single learnable vector `alpha ∈ ℝ⁵` passed through `softmax` to produce channel weights.
- [ ] Forward: `gate(letters_dict) -> dict[Modality, Tensor]` — multiplies each modality carrier by its softmax alpha scalar.
- [ ] Init: `alpha = torch.zeros(5)` (uniform softmax = 0.2 per channel).

**TDD acceptance:**
- `test_plasticity_gate_uniform_at_init` — all 5 gate outputs have the same scale at init (within fp32 eps).
- `test_plasticity_gate_softmax_sums_to_one` — `gate.weights()` sums to 1.0.
- `test_plasticity_gate_gradient_flows` — loss on gated output produces non-zero grad on `alpha`.

### Task 2.2 — AdaptiveCodebook (P2)

- [ ] In `nerve.py`, implement `AdaptiveCodebook(nn.Module)` wrapping an `nn.Parameter[alphabet_size, d_hidden]` (init from the shared multiplexer constellation as starting point, then freely adapt).
- [ ] Forward: `codebook(soft_codes: Tensor[B, K, A]) -> Tensor[B, K, d_hidden]` — soft-assignment projection.
- [ ] Temperature schedule scaffold: accept `tau` kwarg (default 1.0), to be annealed 1.0 → 0.1 by `AdaptationLoop` (Task 2.10). Gumbel-softmax applied before the matmul.

**TDD acceptance:**
- `test_codebook_output_shape` — `(B, K, A) → (B, K, d_hidden)` for `B=4, K=7, A=64, d_hidden=128`.
- `test_codebook_tau_controls_entropy` — at tau=0.01 the assignment is near-onehot; at tau=10 it is near-uniform.
- `test_codebook_gradient_flows_to_param` — backprop reaches the codebook parameter.

### Task 2.3 — CrossModalTransducer (P3)

- [ ] In `nerve.py`, implement `CrossModalTransducer(nn.Module)` — a single directed-edge MLP with 2 hidden layers, gated activation by a scalar mask that external callers flip 0/1 per step.
- [ ] Forward: `transducer(x: Tensor[B, d_hidden], *, active: bool) -> Tensor[B, d_hidden]` — identity when `active=False`, full forward when `active=True`.
- [ ] The class holds its own `(source, target)` `Modality` pair for logging.

**TDD acceptance:**
- `test_transducer_identity_when_inactive` — `active=False` returns input unchanged.
- `test_transducer_changes_output_when_active` — `active=True` produces a different tensor.
- `test_transducer_gradient_flows_when_active`.
- `test_transducer_pair_is_stored` — `t.source` and `t.target` match constructor args.

### Task 2.4 — CrossModalNerve class + fuse()

- [ ] In `nerve.py`, implement `CrossModalNerve(Nerve)` (Nerve from `nerve_core.protocols`; duck-typed per §2.2).
- [ ] Assembles:
  - `self.gates`: `PlasticityGate` (P1)
  - `self.codebook`: `AdaptiveCodebook` (P2)
  - `self.transducers`: `nn.ModuleDict` indexed by `"{src}->{dst}"` strings over the 20 directed pairs (5×5 minus 5 self-loops).
- [ ] `fuse(letters: dict[Modality, Tensor]) -> Tensor` — takes 5 `(B, T)` carriers (output of `SensoryWML.step`), applies:
  1. Per-modality soft demod via `mux.demodulate(carrier, hard=False, tau=...)` → `(B, K, A)` soft codes per modality.
  2. Project each modality's soft codes through the adaptive codebook → `(B, K, d_hidden)`.
  3. For each directed `(src, dst)` pair, run the transducer gated on `active = gates[src] < 0.1 AND gates[dst] > 0.3` (per spec §4.3).
  4. Gate-weighted sum across modalities → `(B, K, d_hidden)` fused representation.
  5. Re-encode via mux constellation's pseudo-inverse → output carrier or codes (see integration contract in §3.5).
- [ ] For Sprint 2 MVP: return the fused `(B, K, d_hidden)` tensor directly; `IntegrationHead` consumes it. Re-encoding to carrier is deferred to Sprint 3 if ever needed.

**TDD acceptance:**
- `test_fuse_shape` — takes 5 carriers `(B, 175)`, returns `(B, K=7, d_hidden=128)`.
- `test_fuse_respects_gate_weights` — raising `gates.alpha[audio]` and zeroing others produces a fused output biased toward audio's contribution.
- `test_fuse_end_to_end_differentiable` — backprop reaches (a) mux constellation, (b) input_proj of every SensoryWML, (c) PlasticityGate.alpha, (d) AdaptiveCodebook.param, (e) at least one active CrossModalTransducer.

### Task 2.5 — on_lesion + migration_stats

- [ ] `CrossModalNerve.on_lesion(modality, snr_db)` — logs a lesion event, optionally re-initialises the affected gate's alpha with a small negative bias (documented).
- [ ] `CrossModalNerve.migration_stats() -> MigrationReport` — dataclass with per-modality gate values, codebook entropy, transducer activation counts. Used by `AdaptationLoop` (Task 2.10).

**TDD acceptance:**
- `test_on_lesion_updates_internal_log` — after `on_lesion("audio", -20)`, the stats report reflects the event.
- `test_migration_stats_shape` — returned dataclass has the expected fields.

### Task 2.6 — LesionSpec + M2 SNR schedule

- [ ] In `src/bouba_sens/lesion.py`, implement `LesionSpec` frozen dataclass matching spec §3.4.
- [ ] Implement `m2_snr_schedule(step: int, *, snr_init: float = 20.0, snr_floor: float = -20.0, k: int = 5000) -> float` per spec §4.1.

**TDD acceptance:**
- `test_lesionspec_is_frozen` — mutation raises `FrozenInstanceError`.
- `test_m2_starts_at_snr_init` — `m2(step=0)` ≈ `snr_init`.
- `test_m2_converges_to_floor` — `m2(step=10*K)` ≈ `snr_floor`.

### Task 2.7 — LesionScheduler.apply()

- [ ] Implement `LesionScheduler(spec: LesionSpec)` with an `.apply(sample: WorldSample, step: int) -> WorldSample` that adds modality-local Gaussian noise scaled to the running SNR (spec §4.1).
- [ ] Noise is additive and *relative* to the modality's signal magnitude (so spectral statistics are preserved): `noise_std = signal_std × 10^(-SNR(t) / 20)`.

**TDD acceptance:**
- `test_apply_modifies_only_lesioned_modality` — non-lesioned modalities are returned byte-identical.
- `test_apply_respects_snr_schedule` — higher step → more noise (lower SNR → larger `(lesioned - original).std()`).
- `test_apply_preserves_label` — the label field is never modified.

### Task 2.8 — IntegrationHead

- [ ] In `src/bouba_sens/head.py`, implement `IntegrationHead(nn.Module)` — Sprint 2 MVP is a 10-class linear classifier over the fused `(B, K, d_hidden)` representation (pooled as `mean(dim=1)` → `(B, d_hidden)` → Linear → `(B, 10)`).
- [ ] Accepts `n_classes` kwarg (default 10) so Sprint 3 can swap labels without changing the interface.

**TDD acceptance:**
- `test_head_output_shape` — `(B, K, d_hidden) → (B, n_classes)`.
- `test_head_gradient_flows`.

### Task 2.9 — AdaptationLoop.pretrain()

- [ ] In `src/bouba_sens/loop.py`, implement `AdaptationLoop` holding: world, 5 SensoryWMLs, shared mux, CrossModalNerve, IntegrationHead, an optimiser over all parameters.
- [ ] `pretrain(steps: int) -> Checkpoint` — runs a simple cross-entropy training loop. Uses `WorldSimulator.sample` → 5 carriers → `fuse` → head → CE loss on `sample.label`. Logs per-step loss and accuracy.
- [ ] Checkpoint is a `@dataclass` containing all module state_dicts.

**TDD acceptance:**
- `test_pretrain_100_steps_no_nan` — loss is finite at every step.
- `test_pretrain_loss_decreases` — `loss[-10:].mean() < loss[:10].mean()` on a 100-step run.
- `test_pretrain_checkpoint_roundtrip` — saving + loading preserves exact parameter values.

### Task 2.10 — lesion_phase + θ-replay buffer

- [ ] `AdaptationLoop.lesion_phase(lesion: LesionSpec, steps: int) -> AdaptationReport` — Phase 2 loop:
  - Lesion applied to sample before SensoryWML.step via `LesionScheduler`.
  - `CrossModalNerve.on_lesion` called once at t=0.
  - θ-replay buffer: FIFO list of up to 1 024 pre-lesion fused tensors (pulled from the intact checkpoint). At each Phase-2 step, compose batch as `[fresh_samples_under_lesion, replay_samples]`. Replay gets a θ-phase weight (alpha bias) to simulate 6 Hz rhythm.
  - `migration_stats()` captured every 100 steps → appended to `AdaptationReport`.
- [ ] `AdaptationReport` is a `@dataclass` with `loss_curve`, `accuracy_curve`, `gate_trajectory`, `codebook_entropy_trajectory`, `transducer_activation_trajectory`.

**TDD acceptance:**
- `test_lesion_phase_100_steps_no_nan` — same robustness test under lesion.
- `test_lesion_phase_populates_migration_report` — `report.gate_trajectory` has ≥ 1 entry.
- `test_replay_buffer_fifo_behavior` — after 2 × buffer_size samples, the oldest entry has been evicted.

### Task 2.11 — Integration gate: Phase 1 + Phase 2 smoke

- [ ] In `tests/integration/test_phase1_phase2_smoke.py`:
  1. Build `GaussianWorld(seed=0)`, shared `GammaThetaMultiplexer`, 5 SensoryWMLs, `CrossModalNerve`, `IntegrationHead`, `AdaptationLoop`.
  2. Run `loop.pretrain(steps=100)` → `ckpt_intact`.
  3. Build `LesionSpec(modality="audio", mode="M2", timing="T2", schedule=m2_snr_schedule)`.
  4. Run `loop.lesion_phase(spec, steps=100)` → `report`.
  5. Assert: no NaN anywhere, final accuracy > chance (0.25 for 4-class Gaussian), report contains non-empty trajectories, gate_trajectory shows audio gate DECREASING (lesion compensates away from audio).

**TDD acceptance:**
- Single `pytest` invocation under 60 s on GrosMac.
- `report.gate_trajectory[-1]["audio"] < report.gate_trajectory[0]["audio"]` by at least 10 %.

### Task 2.12 — Sprint 2 close

- [ ] Bump `src/bouba_sens/__init__.py` version to `0.0.3`.
- [ ] Update `CHANGELOG.md` with Sprint 2 entry (12 tasks).
- [ ] Commit, push, tag `v0.0.3-sprint2` on `main`.
- [ ] Update memory `project_bouba_sens_sprint0_2026_04_20.md` with Sprint 2 closure (or migrate to `project_bouba_sens_sprint2_2026_XX_XX.md` if scope warrants).

**Acceptance criteria:**
- `uv run pytest` green (all unit + integration tests pass, target ≥ 70/70).
- `uv run ruff check src tests` clean.
- `uv run mypy src tests` no new errors.
- `git tag -l | grep v0.0.3-sprint2` returns the tag.
- Studio smoke check: `ssh studio 'cd ~/Projets/bouba_sens && git pull && uv run pytest tests/integration/'` green.

---

## Out of scope (Sprint 3 or later)

- Metric harness (Me1/Me2/Me3/Me6/Me7/Me8/Me9) — Sprint 3.
- `cli.py` typer/rich interface — Sprint 3.
- Jinja2 HTML report (`report.py`) — Sprint 3.
- Configs for all 10 (timing × modality) variants — Sprint 3.
- Empirical tests (`tests/empirical/`) — Sprint 3+.
- Nightly `full-benchmark.yml` activation — Sprint 3.
- Any real training beyond the 100+100 step smoke — Sprint 3 runs on Studio.

## Risks / checkpoints

- **R-sprint2-1** — **Fuse gradient plumbing.** The sprint's hardest gate is Task 2.4: backprop must reach 5 independent input_proj encoders, the shared constellation, the gate alpha, the codebook param, and at least one transducer. If any path is blocked by an argmax, use the Sprint 1 softmax-bridge pattern OR switch to `mux.demodulate(hard=False)` which already Gumbel-softmaxes per Q2 arbitration.
- **R-sprint2-2** — **θ-replay buffer memory.** A 1024-entry buffer of `(B=32, K=7, d_hidden=128)` fused tensors is ~11 MB. Fine on GrosMac. Document explicitly as a cap.
- **R-sprint2-3** — **Lesion noise scaling.** Per-modality signal magnitude must be estimated running (EMA), otherwise early-step noise dominates. Default EMA α=0.05.
- **R-sprint2-4** — **SensoryWML.step "1 code replicated across K slots" MVP caveat from Sprint 1.** In Sprint 2 `fuse`, replace with `mux.demodulate(hard=False)` which actually gives independent soft codes per slot. This is the natural fix — plan implements it in Task 2.4 step 1.

## Related context

- nerve-wml `master@77efb4d` — multiplexer MERGED, soft demod (Gumbel-softmax) ready for the fuse gradient path.
- bouba_sens `main@c90d6a5`, tag `v0.0.2-sprint1` — Sprint 1 closed with 53 tests, left half of §2.1 wired.
- ADR-0001 — shared codebook; every CrossModalNerve instance references the same `mux.constellation` as its SensoryWMLs.
- Spec §3.3–§3.6 + §4 — literal source of the public interfaces and the M2/T3 protocol.
