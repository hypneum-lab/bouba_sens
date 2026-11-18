# Testing

## Framework

pytest (plus `pytest-xdist`, `pytest-cov` and `hypothesis`, all in the `dev` extra of `pyproject.toml`).
`pyproject.toml` sets `testpaths = ["tests"]` and `addopts = "-ra --strict-markers --strict-config"`, so
an undeclared marker is an error. The marker list in `pyproject.toml` is the source of truth.

## Test layout

| Directory | Role | Marker | Run in CI |
|---|---|---|---|
| `tests/unit` | Per-module tests of `src/bouba_sens` and of the scripts in `scripts/`. | none | yes |
| `tests/smoke` | Fast sanity checks: package imports, the CLI prints the package version, the `nerve-wml` API symbols the project relies on. | none (`smoke` is declared but not applied) | yes |
| `tests/property` | Reserved for `hypothesis` property tests. Holds only `__init__.py` today, so it collects no tests. | none | no |
| `tests/integration` | End-to-end runs of the pipeline: paired-run CLI wiring, five-modality emission, Phase 1 + Phase 2 smoke, v0.1 smoke run. | none | no |
| `tests/empirical` | Structural checks on the grid aggregate artifacts under `reports/`. | none (`empirical` is declared but not applied) | no (only `full-benchmark.yml`, manual dispatch on a self-hosted runner) |
| `tests/golden` | No tests. Holds `loop_default_main.json`, read by `tests/unit/test_loop_default_regression.py`. | none | n/a (used by the unit tests) |

CI (`.github/workflows/ci.yml`) runs exactly `uv run pytest tests/unit tests/smoke -v`.

## Markers

Declared in `[tool.pytest.ini_options]` of `pyproject.toml`:

| Marker | Meaning | Applied to a test today |
|---|---|---|
| `slow` | Slow tests (deselect with `-m "not slow"`). | no |
| `empirical` | Nightly tests that run real benchmarks. | no |
| `smoke` | Fast end-to-end sanity checks. | no |

No test carries these markers yet, so `-m smoke` or `-m empirical` currently selects nothing. Select
by directory instead.

## Running

```bash
uv run pytest tests/unit tests/smoke        # what CI runs
uv run pytest tests/unit                    # unit bucket
uv run pytest tests/smoke                   # smoke bucket
uv run pytest tests/integration             # integration bucket
uv run pytest tests/empirical               # empirical bucket (needs reports/v0.1_aggregate.json or reports/v0.2_aggregate.json)
uv run pytest tests/property                # property bucket (no tests yet)
uv run pytest tests/ -v -k <name>           # filter by name
uv run pytest --collect-only -q | tail -1   # number of collected tests
```

`tests/golden` has no command of its own; its file is exercised by the unit bucket.

## Conventions

- Test files: `test_*.py` or `*_test.py`
- Mirror module structure: `tests/<module>/test_<thing>.py`
- One assertion focus per test when practical
- Fixtures in `conftest.py`, not duplicated across files

## Mocking

- Prefer real implementations
- Mock only at boundaries: network, time, randomness, filesystem

## Anti-patterns

- Don't mock what you can test directly
- Don't test implementation details (private methods, internal state)
- Don't share mutable state between tests
- Don't write tests that pass when production code is broken
