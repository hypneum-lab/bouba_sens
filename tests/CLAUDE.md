# Testing — bouba_sens

## Framework

pytest with `--strict-markers` and `--strict-config` (see
`pyproject.toml [tool.pytest.ini_options]`).

## Test directories

| Directory | Role | Marker | Run in CI |
|-----------|------|--------|-----------|
| `unit/` | Fast, isolated unit tests for individual functions and classes | (none / `slow` for a few) | Yes |
| `property/` | Hypothesis property tests for algebraic invariants | (none) | Yes |
| `smoke/` | Fast end-to-end sanity checks (imports, CLI, nerve-wml API) | `smoke` | Yes |
| `integration/` | Multi-component integration tests; may load real data | (none) | No |
| `empirical/` | Full benchmark cells; require a GPU box and real data | `empirical` | No |
| `golden/` | Bit-exact regression data for `test_loop_default_regression.py` | (none — golden data, not tests) | Yes (via unit/) |

The three markers declared in `pyproject.toml`:
- `slow` — deselect with `-m "not slow"`
- `empirical` — nightly tests that run real benchmarks
- `smoke` — fast end-to-end sanity checks

## Commands per bucket

```bash
# CI command (unit + property + smoke):
uv run pytest tests/unit tests/property tests/smoke -v -rs

# Default loop regression (golden, arm64 mac only for hash checks):
uv run pytest tests/unit/test_loop_default_regression.py -v -rs

# Property tests only:
uv run pytest tests/property -v

# Integration (not in CI):
uv run pytest tests/integration -v

# Empirical (GPU box, real data):
uv run pytest -m empirical -v
```

## Conventions

- Test files: `test_*.py`
- Mirror module structure: `tests/<bucket>/test_<thing>.py`
- One assertion focus per test when practical
- Fixtures in `conftest.py`, not duplicated across files
- **Tolerances are named constants** — never use magic numbers for
  numeric comparisons (e.g. `ATOL: float = 1e-6` at module level)
- Pre-registered thresholds (B-1 ≥ 0.05, B-2 ≥ 0.10, B-3 ≥ 0.02) must
  not be changed in a phase-0 PR — open a `type:correction` issue first

## Mocking

- Prefer real implementations
- Mock only at boundaries: network, time, randomness, filesystem

## Anti-patterns

- Do not mock what you can test directly
- Do not test implementation details (private methods, internal state)
- Do not share mutable state between tests
- Do not write tests that pass when production code is broken
