# AGENTS.md

Guidance for AI coding agents (Claude Code, Aider, Cursor, etc.) working in this repo.

## Project

`bouba_sens` — TMLR-track benchmark for the bouba/kiki cross-modal plasticity phenomenon, developed under the **GENIAL framework** at **Hypneum Lab**. Studies how a 5-modality agent (audio, vision, tactile, gravity, force) reorganises after channel loss/degradation. Three pre-registered invariants B-1/B-2/B-3 (OSF `10.17605/OSF.IO/Q6JYN`, amendment v0.6). HEAD ≈ v0.5.9, ~156 commits, 19 ADRs, 185+ tests.

## Tech stack

- Language: Python **3.14** (`requires-python = ">=3.14"`)
- Runtime: `uv` (PEP 668)
- Test: `pytest` (+ `pytest-xdist`, `pytest-cov`, `hypothesis`)
- Build: `setuptools` (src layout, packages under `src/bouba_sens/`)
- Deps: `torch>=2.5`, `numpy>=2.0`, `torchvision`, `scikit-learn`, `librosa`, `hydra-core`, `typer`, `matplotlib`, `plotly`, `pyarrow`, `orjson`, **`nerve-wml`** (local sibling clone via `[tool.uv.sources]` — not on PyPI yet)
- CLI: `bouba-sens` → `bouba_sens.cli:app`

## Commands

```bash
uv sync                                          # install (resolves local nerve-wml)
uv run python -m pytest tests/ -x                # full
uv run python -m pytest tests/unit/ -x           # fast bucket
uv run python -m pytest tests/smoke/ -x          # smoke
uv run python -m bouba_sens.cli --help
```

## Conventions

- Commits: subject ≤ 50 chars, body ≤ 72, no underscore in scope, no AI attribution, never `--no-verify`.
- Branches: `feat/<name>`, `fix/<name>`, `docs/<name>`, `n12/<name>`, `q3/<name>` (multi-seed reruns).
- ADRs in `docs/adr/NNNN-*.md` — 0004..0019 exist; bump number, never reuse.
- Critic-review mandatory for ship-impacting commits — see `~/.claude/projects/-Users-electron/memory/feedback_critic_before_ship.md` (Sprints 10/11 saved by critic catching a MAJOR contradiction).
- Single-seed/under-seeded qualitative claims are barred from publication — adopted portfolio-wide.

## File layout

- `src/bouba_sens/` — benchmark engine, lesions, metrics, CLI.
- `tests/{unit,property,smoke,integration,empirical}/` — test pyramid.
- `docs/adr/` — 19 ADRs (0004-0019).
- `docs/paper/` — TMLR draft + §5.5 reformulation drafts.
- `docs/osf/` — pre-registration + amendments.
- `docs/milestones/` — sprint closure notes.
- `docs/superpowers/` — plans / brainstorms.
- `configs/` — grid + world configs.
- `scripts/` — orchestration helpers.
- `reports/` — per-version aggregates, curves, MANIFEST.
- `runs/`, `runs_q3_base_2026_05_11/` — run outputs (gitignored).

## Domain-specific gotchas

- **§5.5 of the paper is under FINAL Retract** (ADR-0019, Q3+ 10-seed). TMLR submission is **BLOCKED** until §5.5 is reformulated. Do not advance toward submission while §5.5 is in Retract state.
- **N12 subgroup replication** runs on kx6tm-23 (tactile-floor / force-plus10) and picks the reformulation draft A/B/C — wait for N12 verdict before touching §5.5.
- **B-2 is Null** (Kraskov + MINE agree within 0.1 bit). Do not silently re-state as "trend".
- **B-3 PASSES at 5-6× threshold** and anchors `dream-of-kiki` Stratum 4 — protect this invariant.
- **nerve-wml is a local sibling dep** via `[tool.uv.sources]` (not on PyPI). New checkouts must have `../nerve-wml` cloned beside this repo or `uv sync` will fail.
- **Python 3.14 required** — older venvs (3.12/3.13) will resolve incompatible wheels.
- **Numerical reports re-generate from runs/**; do not hand-edit `reports/*.json` or paper figures — re-run the script.
- **Multi-seed (`n>=10`) is the floor for empirical claims** — single-seed Q3 results are why §5.5 is in Retract. Methodology bootstrap CI lives in `nerve-wml` v1.5.3.

## When in doubt

- Read `README.md` (full scientific framing), `CLAUDE.md`, and the latest ADR.
- Recent commits: `git log --oneline -20`.
- Memory: `~/.claude/projects/-Users-electron/memory/project_n8n9_verdicts_2026_05_11.md`, `project_bouba_sens_sprint0_2026_04_20.md`.
- Cluster context: `~/CLAUDE.md`.
- Run `uv run python -m pytest tests/unit/ tests/smoke/ -x` before non-trivial commits.
