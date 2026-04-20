# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
