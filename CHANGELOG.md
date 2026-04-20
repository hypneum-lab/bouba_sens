# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Pending

- **ADR-0001 decision** — OQ1 spike (`scripts/spikes/oq1_codebook.py`) must be
  executed on Studio (M3 Ultra) per the project compute-routing directive.
  Once `out/oq1_results.json` is produced, update the "Decision" section of
  `docs/adr/0001-codebook-sharing.md` with empirical numbers and flip the
  index status from *Proposed* to *Accepted*.
- **Git tag `v0.0.1-sprint0`** — deferred until ADR 0001 decision lands,
  since the Sprint 0 exit criteria require a filled-in decision block.

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
