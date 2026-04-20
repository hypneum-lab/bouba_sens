# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-04-20 (Sprint 3 close — feature-complete v0.1 design)

### Added

- Sprint 3 plan (`docs/superpowers/plans/2026-04-20-bouba-sens-sprint3.md`).
- Seven v0.1 metrics covering the three B-1/B-2/B-3 invariants:
  - `me1_accuracy` + `me2_recovery_auc` over `AdaptationReport.accuracy_curve`.
  - `me3_mi` + `me3_delta` via sklearn Kraskov kNN estimator.
  - `me6_asymmetry` + `me6_max_abs_off_diag` on the n×n lesion-query
    perf matrix.
  - `me7_congenital_gap` scalar on T1 / T2 perfs.
  - `me8_baselines` + `freeze_nerve_plasticity` +
    `random_rewire_nerve` for the three §4.4 regimes.
  - `me9_bootstrap` CI 95 % via `scipy.stats.bootstrap`.
- `metrics/__init__.py` `Metric` Protocol + `EvalReport` dataclass
  re-exporting all 11 callables.
- CLI (spec §5.3): `sim`, `train`, `lesion`, `eval`, `aggregate`
  via typer with parquet + pickle + JSON round-trips.
- `render_html` static-table summary per spec §5.4 (5 section
  headers; interactive plotly deferred to Sprint 4).
- Integration gate `tests/integration/test_v01_smoke_run.py`:
  4.78 s single-seed end-to-end on GrosMac, all metrics finite,
  Me1 > 0.25.
- `scripts/v01_smoke.sh` CLI wrapper chaining train -> lesion
  -> eval -> aggregate.
- ADR-0002 `docs/adr/0002-v01-go-no-go.md` — GO decision.

### Verified

- 115/115 tests pass under `uv run pytest` on GrosMac.
- `uv run ruff check src tests` clean.
- `uv run mypy src tests` clean.
- CLI round-trip: `bouba-sens sim | train | lesion | eval | aggregate`
  all exit 0.

### Still deferred to Sprint 4

- Full 150-run grid on Studio + Me9 bootstrap IC aggregation.
- Interactive plotly heatmap for Me6 and recovery curves for Me2.
- Hydra config-driven experiments (10-config timing × modality grid).
- `tests/empirical/` asserts on real run results (nightly CI).
- `full-benchmark.yml` activation.

### Note on versioning

`v0.1.0` is the first non-sprint tag (Sprint-0/1/2 were
`v0.0.2-sprint1` / `v0.0.3-sprint2`). Marks feature-completeness of
the v0.1 design — code + metrics + CLI + report. Paper v0.1 draft
remains a separate Sprint 5 concern.

## [0.0.3] — 2026-04-20 (Sprint 2 close)

### Added

- Sprint 2 plan (`docs/superpowers/plans/2026-04-20-bouba-sens-sprint2.md`).
- `PlasticityGate` — channel-wise softmax over 5 modalities (P1, Task 2.1).
- `AdaptiveCodebook` — Gumbel-softmax-capable projection,
  `nn.Parameter[A, d_hidden]`, optionally seeded from
  `mux.constellation` (P2, Task 2.2).
- `CrossModalTransducer` — directed-edge MLP with 0/1 gating
  (P3, Task 2.3); 20 instances in `nn.ModuleDict` keyed
  `"{src}->{dst}"`.
- `CrossModalNerve` — assembles P1+P2+P3, implements `fuse()`
  via soft demod (`mux.demodulate(hard=False)`), codebook
  projection, per-pair transducer activation per spec §4.3,
  and gate-weighted sum (Task 2.4). Plus `on_lesion()` and
  `migration_stats() -> MigrationReport` (Task 2.5).
- `LesionSpec` frozen dataclass + `m2_snr_schedule` per spec
  §3.4/§4.1 (Task 2.6).
- `LesionScheduler.apply()` — SNR-scaled additive Gaussian
  noise on the targeted modality, returns a new frozen
  `WorldSample` (Task 2.7).
- `IntegrationHead` — minimal `n_classes` linear classifier
  over mean-pooled `(B, K, d_hidden)` (Task 2.8).
- `AdaptationLoop.pretrain()` Phase 1 + `.lesion_phase()`
  Phase 2 with `AdaptationReport` trajectories and FIFO
  theta-replay buffer (Tasks 2.9-2.10).
- Integration acceptance gate `tests/integration/test_phase1_
  phase2_smoke.py` — 100 + 100 steps, no NaN, accuracy beats
  chance, audio gate drops ≥10 % from pre-lesion baseline
  (Task 2.11).

### Verified

- 76/76 tests pass under `uv run pytest` on GrosMac.
- `uv run ruff check src tests` clean.
- `uv run mypy src tests` clean.
- Sprint 2 acceptance gate runs in ~2 s.

### Still deferred to Sprint 3+

- Metric harness Me1/Me2/Me3/Me6/Me7/Me8/Me9.
- `cli.py` typer/rich interface.
- Jinja2 HTML report (`report.py`).
- Empirical tests (`tests/empirical/`).
- Full configs grid for (timing × modality) variants.
- Training runs on Studio.

### Note on versioning

Sprint 1 was tagged `v0.0.2-sprint1` but the `_version.py`
was not bumped from `0.0.1`. Sprint 2 jumps the package
version to `0.0.3` to honour the git tag semantics going
forward.

## [0.0.2] — 2026-04-20 (Sprint 1 close)

### Added

- Sprint 1 plan (`docs/superpowers/plans/2026-04-20-bouba-sens-sprint1.md`).
- `WorldSample` frozen dataclass + `@runtime_checkable` `WorldSimulator`
  Protocol (Task 1.1).
- `GaussianWorld` — N(0, I_32) latent, 5 orthogonally-projected modalities,
  4-class sign-pattern label (Task 1.2).
- `XORWorld` — Rademacher latent, 2-class parity label (Task 1.3).
- `SinusoidWorld` — circular-latent (sin, cos)-on-unit-circle with Gaussian
  noise on remaining 30 dims, 4-class quantisation label (Task 1.4).
- `test_pairwise_mi_below_0_2_bit` parametrised across all 3 worlds —
  verifies the orthogonal-factored projection scheme keeps pairwise
  modality MI under 0.2 bit (Task 1.5, spec §3.1 invariant).
- `SensoryWML(MlpWML)` — modality-typed subclass with shared-mux identity
  via `object.__setattr__` bypass to prevent `nn.Module` double-registration
  (Task 1.6; ADR-0001 shared-codebook).
- 5 modality-specific encoders (`AudioEncoder`, `VisionEncoder`,
  `TactileEncoder`, `GravityEncoder`, `ForceEncoder`) all emitting
  `(B, d_hidden=128)` and preserving the global `torch.get_rng_state()`
  (Task 1.7).
- `SensoryWML.step(x)` — encodes modality tensors into a PAC carrier via
  the shared multiplexer; differentiable softmax bridge restores gradient
  to `input_proj` through the otherwise-argmax-blocked path (Task 1.8).
- Integration smoke gate `tests/integration/test_five_modality_emission.py`
  — 5 modalities × shared mux → 5 carriers × demod → valid code indices
  (Task 1.9).

### Changed

- `track_p.multiplexer` dep → nerve-wml `master@77efb4d` (PR #2 merged,
  all Q1-Q5 arbitrated) with `GammaThetaMultiplexer`, `GammaThetaConfig`,
  `AWGN`, `HardwareJitterNoise`, `NoiseModel` exports + `py.typed`.
- §6.3 of the design spec updated to reflect the shipped multiplexer API
  (commit `cbab3e3`).

### Verified

- 53/53 tests pass under `uv run pytest` on GrosMac (Python 3.14.3, torch
  post-sync).
- All ruff + mypy pre-commit hooks green.
- Studio smoke import verified 2026-04-20: `T=175 samples` bin-aligned.

### Still deferred to Sprint 2+

- `CrossModalNerve.fuse` + `PlasticityGate` + `AdaptiveCodebook` +
  `CrossModalTransducer` (Sprint 2).
- `LesionScheduler` + M2/T3 SNR protocol (Sprint 2).
- `IntegrationHead` + 10-class classifier (Sprint 3).
- `AdaptationLoop` + θ-replay buffer (Sprint 2–3 split per spec).
- Metric harness (Me1/Me2/Me3/Me6/Me7/Me8/Me9) (Sprint 3).

## [0.0.1] — 2026-04-20

### Added

- Initial design spec (`docs/superpowers/specs/2026-04-20-bouba-sens-design.md`).
- Sprint 0 plan (`docs/superpowers/plans/2026-04-20-bouba-sens-sprint0.md`).
- Package skeleton `src/bouba_sens/` with module stubs for sensory, nerve,
  lesion, head, loop, cli, report, world/*, metrics/*.
- CLI entrypoint `bouba-sens version`.
- Local dependency on `nerve-wml` v0.1.0 (sibling workspace clone; not yet on
  PyPI) — pinned via `[tool.uv.sources]`.
- Smoke test `tests/smoke/test_nerve_wml_api.py` verifying the 4 required
  public symbols (spec §6.3): `nerve_core.protocols.Nerve`,
  `nerve_core.neuroletter.Neuroletter`, `track_w.mlp_wml.MlpWML`,
  `track_p.transducer.Transducer`. The 5th expected symbol
  (`GammaThetaMultiplexer`) is tracked as an upstream gap (issue #1).
- Pre-commit hooks (ruff v0.15.11, mypy v1.20.1, hygiene hooks).
- GitHub Actions: `ci.yml` (lint + mypy + unit/smoke tests on Python 3.14)
  and `full-benchmark.yml` (nightly stub, activated in Sprint 3).
- OQ1 spike script (`scripts/spikes/oq1_codebook.py`) — execution deferred
  to Studio (M3 Ultra) per compute-routing directive.
- ADR-0001 stub (`docs/adr/0001-codebook-sharing.md`, status = Proposed /
  execution pending).
- Hydra-style config skeleton (`configs/v0.1_intact.yaml`) with Me9 in
  `metrics.enabled` and a conservative `codebook_shared=true` default.
- LICENSE (MIT), README, CITATION.cff, SECURITY.md.

### Deferred to Sprint 1+

- OQ1 spike execution (requires Studio).
- All implementation bodies — every module stub raises `NotImplementedError`.
- Empirical tests (`tests/empirical/`) and property tests (`tests/property/`).
