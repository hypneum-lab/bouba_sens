# Contributing to bouba_sens

Pre-registered cross-modal plasticity benchmark of Hypneum Lab. Thresholds
B-1 >= 0.05, B-2 >= 0.10, B-3 >= 0.02 are locked (OSF amendment v0.6) and
are not up for discussion in a PR. Honest, multi-seed reporting first.

## Set up and run

```bash
# nerve-wml is consumed from a sibling checkout ([tool.uv.sources]).
git clone https://github.com/hypneum-lab/nerve-wml.git ../nerve-wml
git -C ../nerve-wml checkout v1.7.0          # same pin as .github/workflows/ci.yml
git clone https://github.com/hypneum-lab/bouba_sens.git
cd bouba_sens
uv sync --all-extras                          # Python 3.14
uv run pre-commit install
uv run pytest tests/unit tests/smoke          # what CI runs
uv run pytest tests/property tests/integration
uv run pytest -m empirical                    # real grids, needs a GPU box
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy src
```

Test tiers: `tests/{unit,property,smoke,integration,empirical}`; markers
`slow`, `empirical`, `smoke` (`--strict-markers`).

## Evidence contract

- Every number in the paper, README or an ADR traces to a file under
  `reports/` produced by a script in `scripts/` (never hand-edited).
  `reports/MANIFEST.json` (via `scripts/sha256_manifest.py`) pins their
  SHA-256; do not regenerate it to hide a drift, open a `type:correction`.
- Grids are run with `scripts/run_grid.sh` / `reproduce_paper_v01.sh`;
  seeds are explicit; single-seed or under-seeded claims are not
  publishable (portfolio rule since the Q3+ retract, ADR-0019).
- ADRs in `docs/adr/NNNN-*.md` (bump, never reuse); real vs synthetic
  worlds banner-tagged in every figure.

## Commits and branches

- Subject <= 50 characters, body lines <= 72, no underscore in the scope,
  conventional prefix (`feat:`, `fix:`, `docs:`, `test:`).
- No `--no-verify`; pre-commit runs ruff, ruff-format, mypy (strict, `src/`)
  and actionlint.
- No AI attribution trailer. Branches: `feat/`, `fix/`, `docs/`, `n12/`,
  `q3/` (multi-seed reruns).
- Ship-impacting or paper-facing changes get an independent critic review.

## Authorship and AI tools

- Byline for publications: single author, Clément Saillant (PI decision
  2026-10-25, roadmap S1). Contributors are credited in §Acknowledgments
  of each publication and listed in `CONTRIBUTORS.md`.
- LLMs are tools, never co-authors nor in trailers; say so in the PR.

## Issues and license

Use the org templates (`hypothese`, `correction`); new issues carry
`a-valider-pi` plus `phase:*`, `type:*`, `axe:bouba-sens`, `hote:*`.
Bugs: minimal reproducer + environment. Contributions are MIT (LICENSE).
