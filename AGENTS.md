# AGENTS.md

Guidance for AI coding agents (Claude Code, Aider, Cursor, etc.) working in this repo.

## Project

`bouba_sens` — TMLR-track benchmark for the bouba/kiki cross-modal plasticity phenomenon, developed under the **GENIAL framework** at **Hypneum Lab**. Studies how a 5-modality agent (audio, vision, tactile, gravity, force) reorganises after channel loss/degradation. Three invariants B-1/B-2/B-3 pre-registered locally (git-dated design spec 2026-04-20; `docs/osf/` amendments never filed on OSF — `10.17605/OSF.IO/Q6JYN` is dream-of-kiki H1–H4 only). HEAD v0.5.11, 32 ADR files (0001–0027, see `docs/adr/README.md`).

## Tech stack

- Language: Python **3.14** (`requires-python = ">=3.14"`)
- Runtime: `uv` (PEP 668)
- Test: `pytest` (+ `pytest-xdist`, `pytest-cov`, `hypothesis`)
- Build: `setuptools` (src layout, packages under `src/bouba_sens/`)
- Deps: `torch>=2.5`, `numpy>=2.0`, `torchvision`, `scikit-learn`, `librosa`, `hydra-core`, `typer`, `matplotlib`, `plotly`, `pyarrow`, `orjson`, **`nerve-wml`** (sibling clone via `[tool.uv.sources]`; CI and uv.lock pin v1.8.1; clone `hypneum-lab/nerve-wml` beside this repo before `uv sync`)
- CLI: `bouba-sens` → `bouba_sens.cli:app`

## Commands

```bash
uv sync                                          # install (resolves local nerve-wml)
uv run python -m pytest tests/ -x                # full
uv run python -m pytest tests/unit/ -x           # fast bucket
uv run python -m pytest tests/smoke/ -x          # smoke
uv run python -m pytest tests/unit/test_loop_default_regression.py  # golden regression
uv run python -m bouba_sens.cli --help
```

## Conventions

- Commits: subject ≤ 50 chars, body ≤ 72, no underscore in scope, no AI attribution, never `--no-verify`.
- Branches: `feat/<name>`, `fix/<name>`, `docs/<name>`, `n12/<name>`, `q3/<name>` (multi-seed reruns).
- ADRs in `docs/adr/NNNN-*.md` — see `docs/adr/README.md` for current index; bump number, never reuse.
- Critic-review mandatory for ship-impacting commits — see `~/.claude/projects/-Users-electron/memory/feedback_critic_before_ship.md` (Sprints 10/11 saved by critic catching a MAJOR contradiction).
- Single-seed/under-seeded qualitative claims are barred from publication — adopted portfolio-wide.

## File layout

- `src/bouba_sens/` — benchmark engine, lesions, metrics, CLI.
- `tests/{unit,property,smoke,integration,empirical,golden}/` — test pyramid; see `tests/CLAUDE.md`.
- `docs/adr/` — 32 ADR files (0001–0027; index `docs/adr/README.md`).
- `docs/paper/` — TMLR draft + §5.5 reformulation drafts.
- `docs/osf/` — pre-registration amendment drafts (git-dated, never filed on OSF).
- `docs/milestones/` — sprint closure notes.
- `docs/superpowers/` — plans / brainstorms.
- `configs/` — grid + world configs.
- `scripts/` — orchestration helpers.
- `reports/` — per-version aggregates, curves, MANIFEST.
- `runs/`, `runs_q3_base_2026_05_11/` — run outputs (gitignored).

## Domain-specific gotchas

- **§5.5 of the paper is under FINAL Retract** (ADR-0019, Q3+ 10-seed). TMLR submission is **BLOCKED** (ADR-0019, ADR-0024; critic-review gate pending). Do not advance toward submission.
- **N12 subgroup replication** is DONE — LOST 0/2 (ADR-0019); the full-retract reformulation (Version C) has landed in §5.5. Do not reintroduce the retracted dose-response claim.
- **B-1 non-test** (decision of 2026-11-07, issue #24, ADR-0024): measured in a regime where the network does not learn. Do not report as a passed or retracted hypothesis.
- **B-2 non-test / not measurable** (decision of 2026-11-07, ADR-0024). Do not silently re-state as "trend" or "Null".
- **B-3**: see CLAUDE.md for current reading (ADR-0021/0023/0027). Do not call it an "architectural invariant" or describe it as a PASS without PI confirmation.
- **nerve-wml is a local sibling dep** via `[tool.uv.sources]` (CI pins v1.8.1). New checkouts must have `../nerve-wml` cloned beside this repo or `uv sync` will fail.
- **Python 3.14 required** — older venvs (3.12/3.13) will resolve incompatible wheels.
- **Numerical reports re-generate from runs/**; do not hand-edit `reports/*.json` or paper figures — re-run the script.
- **Multi-seed (`n>=10`) is the floor for empirical claims** — single-seed Q3 results are why §5.5 is in Retract. Methodology bootstrap CI lives in `nerve-wml` (CI pins v1.8.1).

## When in doubt

- Read `README.md` (full scientific framing), `CLAUDE.md`, and the latest ADR.
- Recent commits: `git log --oneline -20`.
- Memory: `~/.claude/projects/-Users-electron/memory/project_n8n9_verdicts_2026_05_11.md`, `project_bouba_sens_sprint0_2026_04_20.md`.
- Cluster context: `~/CLAUDE.md`.
- Run `uv run python -m pytest tests/unit/ tests/smoke/ -x` before non-trivial commits.
