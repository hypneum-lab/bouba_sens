# bouba_sens — Sprint 0 Implementation Plan (Week 1: Scaffolding + OQ1 spike)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap the `bouba_sens` repository with a production-grade Python skeleton, pinned `nerve-wml` integration verified by smoke tests, CI, tooling, and an OQ1 spike (shared vs local codebook) whose outcome is recorded as an ADR. End state: a repo that a second engineer can clone, `uv sync`, `pytest`, and read the Sprint 1 starting line.

**Architecture:** Single-package Python 3.14 library (`src/bouba_sens/`) depending on pinned `nerve-wml>=1.1.4,<1.2`. Sprint 0 lands *no production code* — only empty module skeletons, interface smoke tests, and one experimental spike script in `scripts/spikes/`. All invariants from the spec (§2, §3) are *deferred* to Sprint 1 once OQ1 is resolved.

**Tech Stack:** Python 3.14, uv, PyTorch 2.5, pytest + hypothesis + pytest-xdist, ruff + mypy + pyright, typer + rich, hydra, pyarrow, GitHub Actions.

**Parent spec:** `docs/superpowers/specs/2026-04-20-bouba-sens-design.md`

**Sprint 0 scope:** Tasks 0.1 → 0.10. Sprints 1-3 plans will be written **after** Sprint 0 completes so they can absorb OQ1 resolution and observed nerve-wml API surface.

---

## File structure created in Sprint 0

```
bouba_sens/
├── pyproject.toml                                [Task 0.1]
├── uv.lock                                       [Task 0.1]
├── .python-version                               [Task 0.1]
├── .gitignore                                    [Task 0.1]
├── LICENSE                                       [Task 0.3]
├── README.md                                     [Task 0.3]
├── CITATION.cff                                  [Task 0.3]
├── SECURITY.md                                   [Task 0.3]
├── CHANGELOG.md                                  [Task 0.10]
├── .pre-commit-config.yaml                       [Task 0.4]
├── .github/workflows/
│   ├── ci.yml                                    [Task 0.6]
│   └── full-benchmark.yml                        [Task 0.6]
├── src/bouba_sens/
│   ├── __init__.py                               [Task 0.2]
│   ├── py.typed                                  [Task 0.2]
│   ├── _version.py                               [Task 0.2]
│   ├── sensory.py                                [Task 0.2]
│   ├── nerve.py                                  [Task 0.2]
│   ├── lesion.py                                 [Task 0.2]
│   ├── head.py                                   [Task 0.2]
│   ├── loop.py                                   [Task 0.2]
│   ├── cli.py                                    [Task 0.2]
│   ├── report.py                                 [Task 0.2]
│   ├── world/
│   │   ├── __init__.py                           [Task 0.2]
│   │   ├── base.py                               [Task 0.2]
│   │   ├── gaussian.py                           [Task 0.2]
│   │   ├── xor.py                                [Task 0.2]
│   │   └── sinusoid.py                           [Task 0.2]
│   └── metrics/
│       ├── __init__.py                           [Task 0.2]
│       ├── performance.py                        [Task 0.2]
│       ├── mi_migration.py                       [Task 0.2]
│       ├── asymmetry.py                          [Task 0.2]
│       ├── congenital.py                         [Task 0.2]
│       └── baselines.py                          [Task 0.2]
├── tests/
│   ├── __init__.py                               [Task 0.5]
│   ├── conftest.py                               [Task 0.5]
│   ├── unit/__init__.py                          [Task 0.5]
│   ├── property/__init__.py                      [Task 0.5]
│   ├── integration/__init__.py                   [Task 0.5]
│   ├── empirical/__init__.py                     [Task 0.5]
│   └── smoke/test_nerve_wml_api.py               [Task 0.7]
├── configs/
│   ├── README.md                                 [Task 0.9]
│   └── v0.1_intact.yaml                          [Task 0.9]
├── scripts/
│   └── spikes/
│       └── oq1_codebook_sharing.py               [Task 0.8]
└── docs/
    └── adr/
        └── 0001-codebook-sharing.md              [Task 0.8]
```

**Files NOT touched in Sprint 0:** anything in `src/bouba_sens/` beyond empty stubs, `tests/unit/`, `tests/property/`, `tests/integration/`, `tests/empirical/`, `papers/`. Sprint 1 fills these.

---

## Task 0.1: Initialise pyproject + uv

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `uv.lock` (generated)

- [ ] **Step 1: Write `.python-version`**

```
3.14
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.mypy_cache/
.pytest_cache/
.ruff_cache/
.pyright/
.coverage
.hypothesis/

# uv
.venv/

# Build artifacts
build/
dist/
*.whl

# IDE
.vscode/
.idea/
*.swp

# Run outputs
runs/
reports/
data/
*.parquet
*.pt
wandb/

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "bouba_sens"
version = "0.0.1"
description = "Cross-modal plasticity benchmark — Hypneum Lab"
readme = "README.md"
requires-python = ">=3.14"
license = "MIT"
authors = [{ name = "Clement Saillant", email = "c.saillant@gmail.com" }]
keywords = ["cross-modal-plasticity", "multimodal-benchmark", "embodied-ai", "hypneum-lab"]
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.14",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Intended Audience :: Science/Research",
]

dependencies = [
    "torch>=2.5",
    "numpy>=2.0",
    "pyarrow>=17.0",
    "orjson>=3.10",
    "typer>=0.12",
    "rich>=13.7",
    "hydra-core>=1.3",
    "matplotlib>=3.9",
    "plotly>=5.24",
    "jinja2>=3.1",
    "scikit-learn>=1.5",
    # "nerve-wml>=1.1.4,<1.2",  # uncommented in Task 0.7 after verification
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-xdist>=3.6",
    "pytest-cov>=5.0",
    "hypothesis>=6.112",
    "ruff>=0.6",
    "mypy>=1.11",
    "pyright>=1.1",
    "pre-commit>=3.8",
]

[project.scripts]
bouba-sens = "bouba_sens.cli:app"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --strict-config"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "empirical: nightly tests that run real benchmarks",
    "smoke: fast end-to-end sanity checks",
]

[tool.coverage.run]
source = ["src/bouba_sens"]
branch = true

[tool.ruff]
target-version = "py314"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "N", "SIM", "RUF"]
ignore = ["E501"]  # line-length handled by formatter

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.mypy]
python_version = "3.14"
strict = true
warn_unused_ignores = true
warn_redundant_casts = true
ignore_missing_imports = false

[[tool.mypy.overrides]]
module = ["plotly.*", "matplotlib.*", "hydra.*", "jinja2.*", "pyarrow.*"]
ignore_missing_imports = true

[tool.pyright]
include = ["src", "tests"]
pythonVersion = "3.14"
typeCheckingMode = "strict"
```

- [ ] **Step 4: Run `uv sync --all-extras` to install and generate `uv.lock`**

Run:
```bash
cd /Users/electron/Documents/Projets/bouba_sens
uv sync --all-extras
```
Expected: `Resolved N packages`, `uv.lock` file created, `.venv/` directory created.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore
git commit -m "chore: initialise pyproject.toml with uv + dev dependencies"
```

---

## Task 0.2: Package skeleton (empty modules)

**Files:**
- Create: `src/bouba_sens/__init__.py`
- Create: `src/bouba_sens/py.typed`
- Create: `src/bouba_sens/_version.py`
- Create: `src/bouba_sens/sensory.py`, `nerve.py`, `lesion.py`, `head.py`, `loop.py`, `cli.py`, `report.py`
- Create: `src/bouba_sens/world/{__init__.py,base.py,gaussian.py,xor.py,sinusoid.py}`
- Create: `src/bouba_sens/metrics/{__init__.py,performance.py,mi_migration.py,asymmetry.py,congenital.py,baselines.py}`

- [ ] **Step 1: Create top-level package marker and version**

`src/bouba_sens/py.typed` — empty file.

`src/bouba_sens/_version.py`:
```python
"""Single source of truth for package version."""

__version__ = "0.0.1"
```

`src/bouba_sens/__init__.py`:
```python
"""bouba_sens — Cross-modal plasticity benchmark for the Hypneum Lab."""

from bouba_sens._version import __version__

__all__ = ["__version__"]
```

- [ ] **Step 2: Create top-level module stubs**

For each of `sensory.py`, `nerve.py`, `lesion.py`, `head.py`, `loop.py`, `report.py`, create a stub with the signature of the single public symbol — raising `NotImplementedError` so Sprint 1 TDD can hit the stubs.

`src/bouba_sens/sensory.py`:
```python
"""SensoryWML — modality-specific subclass of nerve_wml.MLPWML.

Placeholder for Sprint 0. Implementation lands in Sprint 1.
"""

from __future__ import annotations

from typing import Literal

Modality = Literal["audio", "vision", "tactile", "gravity", "force"]
MODALITIES: tuple[Modality, ...] = ("audio", "vision", "tactile", "gravity", "force")


class SensoryWML:
    """Per-modality wrapper around nerve_wml.MLPWML. See spec §3.2."""

    def __init__(self, modality: Modality) -> None:
        raise NotImplementedError("Sprint 1 — see docs/superpowers/plans/sprint1")
```

`src/bouba_sens/nerve.py`:
```python
"""CrossModalNerve — plastic router with gating, codebook, transducers. See spec §3.3."""

from __future__ import annotations


class CrossModalNerve:
    """Placeholder for Sprint 0 — implementation in Sprint 2."""

    def __init__(self) -> None:
        raise NotImplementedError("Sprint 2 — see docs/superpowers/plans/sprint2")
```

`src/bouba_sens/lesion.py`:
```python
"""LesionScheduler — injects modality corruption between simulator and WMLs. See spec §3.4."""

from __future__ import annotations


class LesionScheduler:
    """Placeholder for Sprint 0 — implementation in Sprint 2."""

    def __init__(self) -> None:
        raise NotImplementedError("Sprint 2 — see docs/superpowers/plans/sprint2")
```

`src/bouba_sens/head.py`:
```python
"""IntegrationHead — task-specific decoder on fused neuroletters. See spec §3.5."""

from __future__ import annotations


class IntegrationHead:
    """Placeholder for Sprint 0 — implementation in Sprint 2."""

    def __init__(self) -> None:
        raise NotImplementedError("Sprint 2 — see docs/superpowers/plans/sprint2")
```

`src/bouba_sens/loop.py`:
```python
"""AdaptationLoop — owns pretrain/lesion/eval phases and the θ-replay buffer. See spec §3.6."""

from __future__ import annotations


class AdaptationLoop:
    """Placeholder for Sprint 0 — implementation in Sprint 2."""

    def __init__(self) -> None:
        raise NotImplementedError("Sprint 2 — see docs/superpowers/plans/sprint2")
```

`src/bouba_sens/report.py`:
```python
"""HTML report generator — Jinja2 template on curves.parquet + metrics.parquet. See spec §5.4."""

from __future__ import annotations


def render_html(run_dir: str, out_path: str) -> None:
    """Placeholder for Sprint 0 — implementation in Sprint 3."""

    raise NotImplementedError("Sprint 3 — see docs/superpowers/plans/sprint3")
```

- [ ] **Step 3: Create CLI stub that at least prints version**

`src/bouba_sens/cli.py`:
```python
"""Typer-based CLI. See spec §5.3.

Sprint 0 ships only `version` so `bouba-sens version` works after `uv sync`.
"""

from __future__ import annotations

import typer

from bouba_sens._version import __version__

app = typer.Typer(help="bouba_sens — Cross-modal plasticity benchmark")


@app.command()
def version() -> None:
    """Print the package version."""

    typer.echo(f"bouba_sens {__version__}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Create world/ sub-package with protocol stub**

`src/bouba_sens/world/__init__.py`:
```python
"""World simulators — produce coherent 5-modality samples from a shared latent."""

from bouba_sens.world.base import WorldSample, WorldSimulator

__all__ = ["WorldSample", "WorldSimulator"]
```

`src/bouba_sens/world/base.py`:
```python
"""Base protocol + dataclass for world simulators. See spec §3.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class WorldSample:
    """One batch of coherent multi-modal observations sharing a latent z."""

    z: torch.Tensor        # (B, D_z)
    audio: torch.Tensor    # (B, T_audio)
    vision: torch.Tensor   # (B, H, W)
    tactile: torch.Tensor  # (B, N_taxels)
    gravity: torch.Tensor  # (B, 3)
    force: torch.Tensor    # (B, 6)
    label: torch.Tensor    # (B,)


class WorldSimulator(Protocol):
    """Produces a WorldSample batch — implementations in Sprint 1."""

    def sample(self, batch_size: int, seed: int) -> WorldSample: ...

    def modality_dims(self) -> dict[str, tuple[int, ...]]: ...
```

`src/bouba_sens/world/gaussian.py`:
```python
"""GaussianWorld — 5 modalities from a shared z via orthogonal factorised projections.

Placeholder for Sprint 0 — implementation in Sprint 1.
"""

from __future__ import annotations


class GaussianWorld:
    def __init__(self, d_z: int = 8) -> None:
        raise NotImplementedError("Sprint 1 — see docs/superpowers/plans/sprint1")
```

`src/bouba_sens/world/xor.py`:
```python
"""XORWorld — non-linearly factorised 5-modality world.

Placeholder for Sprint 0 — implementation in Sprint 1.
"""

from __future__ import annotations


class XORWorld:
    def __init__(self) -> None:
        raise NotImplementedError("Sprint 1 — see docs/superpowers/plans/sprint1")
```

`src/bouba_sens/world/sinusoid.py`:
```python
"""SinusoidWorld — periodic latent, useful for temporal compensation tests.

Placeholder for Sprint 0 — implementation in Sprint 1.
"""

from __future__ import annotations


class SinusoidWorld:
    def __init__(self) -> None:
        raise NotImplementedError("Sprint 1 — see docs/superpowers/plans/sprint1")
```

- [ ] **Step 5: Create metrics/ sub-package**

`src/bouba_sens/metrics/__init__.py`:
```python
"""Metric implementations — each corresponds to an entry in spec §5.2."""
```

For each of `performance.py`, `mi_migration.py`, `asymmetry.py`, `congenital.py`, `baselines.py`, create:

`src/bouba_sens/metrics/performance.py`:
```python
"""Me1 (accuracy post-adaptation) + Me2 (recovery curve AUC). Spec §5.2."""

from __future__ import annotations


class Me1Accuracy:
    """Placeholder for Sprint 0 — implementation in Sprint 3."""

    def __init__(self) -> None:
        raise NotImplementedError("Sprint 3 — see docs/superpowers/plans/sprint3")


class Me2RecoveryAUC:
    """Placeholder for Sprint 0 — implementation in Sprint 3."""

    def __init__(self) -> None:
        raise NotImplementedError("Sprint 3 — see docs/superpowers/plans/sprint3")
```

Same one-line-placeholder pattern for `mi_migration.py` (Me3), `asymmetry.py` (Me6), `congenital.py` (Me7), `baselines.py` (Me8).

- [ ] **Step 6: Verify the package imports cleanly**

Run:
```bash
uv run python -c "import bouba_sens; print(bouba_sens.__version__)"
uv run bouba-sens version
```
Expected output (both commands):
```
bouba_sens 0.0.1
```

- [ ] **Step 7: Commit**

```bash
git add src/bouba_sens/
git commit -m "feat: package skeleton with stubs for all Sprint 1-3 modules"
```

---

## Task 0.3: README + LICENSE + CITATION + SECURITY

**Files:**
- Create: `LICENSE`
- Create: `README.md`
- Create: `CITATION.cff`
- Create: `SECURITY.md`

- [ ] **Step 1: Write `LICENSE` (MIT)**

```
MIT License

Copyright (c) 2026 Clément Saillant (Hypneum Lab)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Write `README.md`**

```markdown
# bouba_sens

> A benchmark for cross-modal plasticity in artificial neural systems.
> Hypneum Lab — 2026.

`bouba_sens` studies how a 5-modality agent (audio, vision, tactile, gravity,
force) reorganises itself when one sensory channel is lost or degraded —
inspired by the cross-modal cortical recruitment observed in congenital and
late blindness (Amedi 2007, Merabet 2010, Heimler 2020).

## Three testable invariants

- **B-1 — Congenital gap:** lesion pre-training yields better adaptation than
  lesion post-convergence.
- **B-2 — MI migration:** mutual information between surviving-modality
  neuroletters and the target label rises post-lesion.
- **B-3 — Perceptive/proprioceptive asymmetry:** losing vision/audio/tactile
  produces a quantitatively different plastic response than losing
  gravity/force.

## Status

**v0.0.1 — Sprint 0 scaffolding.** No working implementation yet.
Design: `docs/superpowers/specs/2026-04-20-bouba-sens-design.md`.
Plan: `docs/superpowers/plans/2026-04-20-bouba-sens-sprint0.md`.

## Quickstart (once Sprint 1 lands)

```bash
uv sync --all-extras
uv run bouba-sens version
uv run pytest
```

## Dependencies

- Python 3.14
- PyTorch ≥ 2.5
- `nerve-wml >=1.1.4,<1.2` (neuroletters, γ/θ multiplexing — Hypneum Lab)

## Priority references

1. Amedi et al. 2007 — Shape conveyed by visual-to-auditory sensory substitution activates LOC (Nat Neurosci).
2. Heimler & Amedi 2020 — Revisiting adaptive and maladaptive effects of crossmodal plasticity (Neuroscience).
3. Röder et al. 2021 — Sensitive periods for functional specialization (PNAS).
4. Alper & Averbuch-Elor 2023 — Kiki or Bouba? Sound symbolism in VLMs (NeurIPS).
5. Girdhar et al. 2023 — ImageBind: One Embedding Space (CVPR).
6. Ma et al. 2022 — Are Multimodal Transformers Robust to Missing Modality? (CVPR).
7. Liang et al. 2022 — MultiBench (NeurIPS D&B).
8. Keller & Mrsic-Flogel 2018 — Predictive Processing: A Canonical Cortical Computation (Neuron).

## License

MIT. See `LICENSE`.

## Citation

See `CITATION.cff`.
```

- [ ] **Step 3: Write `CITATION.cff`**

```yaml
cff-version: 1.2.0
message: "If you use bouba_sens in academic work, please cite it."
title: "bouba_sens — A Cross-Modal Plasticity Benchmark"
authors:
  - family-names: "Saillant"
    given-names: "Clément"
    affiliation: "Hypneum Lab"
    email: "c.saillant@gmail.com"
version: "0.0.1"
date-released: "2026-04-20"
license: MIT
repository-code: "https://github.com/hypneum-lab/bouba_sens"
keywords:
  - cross-modal plasticity
  - multimodal benchmark
  - embodied AI
  - Hypneum Lab
  - sensory substitution
```

- [ ] **Step 4: Write `SECURITY.md`**

```markdown
# Security Policy

## Supported Versions

As of Sprint 0, `bouba_sens` is pre-release (v0.0.x). Only the `main` branch
is security-supported.

## Reporting a Vulnerability

Please report vulnerabilities privately to `c.saillant@gmail.com` with the
subject line `[bouba_sens security] <summary>`. Do **not** open a public
issue for unpatched vulnerabilities.

We aim to acknowledge reports within 5 business days and to provide an
initial assessment within 15 business days.

## Scope

`bouba_sens` is a research benchmark. Relevant vulnerabilities include:

- Arbitrary code execution via malicious config files (YAML, pickle) ingested
  through `bouba-sens run` / `bouba-sens eval`.
- Privilege escalation through install-time hooks.
- Supply-chain compromise via a pinned dependency.

Out of scope: incorrect research claims, missing features, or performance
issues.
```

- [ ] **Step 5: Commit**

```bash
git add LICENSE README.md CITATION.cff SECURITY.md
git commit -m "docs: add LICENSE (MIT), README, CITATION.cff, SECURITY.md"
```

---

## Task 0.4: Pre-commit + ruff + mypy config

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Write `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: ["--maxkb=500"]
      - id: check-merge-conflict

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        additional_dependencies:
          - torch>=2.5
          - numpy>=2.0
          - typer>=0.12
        args: [--config-file=pyproject.toml]
```

- [ ] **Step 2: Install and activate pre-commit**

Run:
```bash
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```
Expected: `pre-commit installed at .git/hooks/pre-commit`.

- [ ] **Step 3: Run pre-commit on all existing files**

Run:
```bash
uv run pre-commit run --all-files
```
Expected: some auto-fixes on previously committed files (trailing whitespace, EOF). Re-run until all hooks pass green.

- [ ] **Step 4: Run ruff and mypy standalone to double-check**

Run:
```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
```
Expected: `All checks passed` for ruff, and 0 errors from mypy.

- [ ] **Step 5: Commit**

```bash
git add .pre-commit-config.yaml
# also commit any auto-fixes the hooks applied
git add -u
git commit -m "chore: pre-commit config (ruff + mypy + hygiene hooks)"
```

---

## Task 0.5: Test structure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/property/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/empirical/__init__.py`
- Create: `tests/smoke/__init__.py`
- Create: `tests/unit/test_smoke.py`

- [ ] **Step 1: Create empty `__init__.py` files**

Create empty files at:
- `tests/__init__.py`
- `tests/unit/__init__.py`
- `tests/property/__init__.py`
- `tests/integration/__init__.py`
- `tests/empirical/__init__.py`
- `tests/smoke/__init__.py`

- [ ] **Step 2: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures and configuration for bouba_sens tests."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic_seed() -> None:
    """Seed all RNGs before every test to keep failures reproducible."""

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


@pytest.fixture
def small_batch() -> int:
    """Default small batch size for fast unit tests."""

    return 4
```

- [ ] **Step 3: Write `tests/unit/test_smoke.py` — verifies package imports + CLI**

```python
"""Sanity tests that prove the Sprint 0 skeleton actually loads."""

from __future__ import annotations

import subprocess

import bouba_sens


def test_package_imports() -> None:
    assert bouba_sens.__version__ == "0.0.1"


def test_cli_version_runs() -> None:
    """bouba-sens version should exit 0 and print the version."""

    result = subprocess.run(
        ["uv", "run", "bouba-sens", "version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "bouba_sens 0.0.1" in result.stdout


def test_world_sample_dataclass_importable() -> None:
    from bouba_sens.world import WorldSample, WorldSimulator  # noqa: F401


def test_modality_type_constants_are_five() -> None:
    from bouba_sens.sensory import MODALITIES

    assert MODALITIES == ("audio", "vision", "tactile", "gravity", "force")
    assert len(MODALITIES) == 5
```

- [ ] **Step 4: Run the tests**

Run:
```bash
uv run pytest tests/unit/test_smoke.py -v
```
Expected output: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test: scaffold test tree + smoke tests for Sprint 0 skeleton"
```

---

## Task 0.6: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/full-benchmark.yml`

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint-test:
    name: lint + test (py${{ matrix.python }})
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ["3.14"]
    steps:
      - uses: actions/checkout@v4

      - name: Set up uv
        uses: astral-sh/setup-uv@v3
        with:
          python-version: ${{ matrix.python }}
          enable-cache: true

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Ruff lint
        run: uv run ruff check src tests

      - name: Ruff format check
        run: uv run ruff format --check src tests

      - name: Mypy
        run: uv run mypy src

      - name: Pytest (unit + smoke)
        run: uv run pytest tests/unit tests/smoke -v
```

- [ ] **Step 2: Write `.github/workflows/full-benchmark.yml` (stub)**

```yaml
name: Full Benchmark

on:
  workflow_dispatch:
  schedule:
    # 03:17 UTC — off-peak, deterministic cadence
    - cron: "17 3 * * *"

jobs:
  benchmark:
    name: 5-seed v0.1 full grid
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up uv
        uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.14"
          enable-cache: true

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Skip until Sprint 3
        run: |
          echo "Full benchmark harness arrives in Sprint 3."
          echo "For Sprint 0 we just assert the workflow file is syntactically valid."
```

- [ ] **Step 3: Validate workflow YAML locally (optional but cheap)**

Run:
```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); yaml.safe_load(open('.github/workflows/full-benchmark.yml')); print('yaml ok')"
```
Expected: `yaml ok`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/
git commit -m "ci: GitHub Actions workflows (CI + nightly benchmark stub)"
```

---

## Task 0.7: Verify `nerve-wml` public API

**Files:**
- Modify: `pyproject.toml` (un-comment nerve-wml dependency)
- Create: `tests/smoke/test_nerve_wml_api.py`

- [ ] **Step 1: Write the failing import-contract test first**

`tests/smoke/test_nerve_wml_api.py`:
```python
"""Smoke test that verifies the nerve-wml symbols bouba_sens depends on exist.

Spec §6.3 defines the contract. If any import fails, file a nerve-wml issue
*before* un-commenting the production dependency.
"""

from __future__ import annotations

import importlib

import pytest

REQUIRED_SYMBOLS = [
    ("nerve_wml.wml", "MLPWML"),
    ("nerve_wml.nerve", "Nerve"),
    ("nerve_wml.codes", "NeuroLetters"),
    ("nerve_wml.mux", "GammaThetaMultiplexer"),
    ("nerve_wml.transducer", "CrossSubstrateTransducer"),
]


@pytest.mark.parametrize(("module_name", "symbol"), REQUIRED_SYMBOLS)
def test_nerve_wml_public_symbol_exists(module_name: str, symbol: str) -> None:
    module = importlib.import_module(module_name)
    assert hasattr(module, symbol), (
        f"{module_name}.{symbol} is required by bouba_sens. "
        f"If absent, open a nerve-wml issue before pinning the dependency."
    )
```

- [ ] **Step 2: Run the test — expect ModuleNotFoundError because nerve-wml is not installed yet**

Run:
```bash
uv run pytest tests/smoke/test_nerve_wml_api.py -v
```
Expected: every parametrised case FAILS with `ModuleNotFoundError: No module named 'nerve_wml'`.

- [ ] **Step 3: Attempt installing `nerve-wml` from local path (Hypneum mono-workspace-style)**

First check if nerve-wml is a pypi package. Run:
```bash
uv pip index versions nerve-wml 2>&1 | head -5
```
If *not on pypi*, install from the sibling local clone (spec §6.3 contract):
```bash
uv add "nerve-wml @ file:///Users/electron/Documents/Projets/nerve-wml"
```
If *on pypi*, run:
```bash
uv add "nerve-wml>=1.1.4,<1.2"
```

Either way, `pyproject.toml` gets the `nerve-wml` line populated and `uv.lock` pins it.

- [ ] **Step 4: Re-run the test — expect PASS**

Run:
```bash
uv run pytest tests/smoke/test_nerve_wml_api.py -v
```
Expected: all 5 parametrised cases PASS.

If a case FAILS (missing symbol), do NOT un-pin or patch locally. Instead:

1. Open an issue in the nerve-wml repo titled `[API] Expose <module>.<symbol> for bouba_sens` referencing spec §6.3.
2. Record the missing symbol in `docs/adr/0001-codebook-sharing.md` under "nerve-wml API gap observations" (created in Task 0.8).
3. Continue with Task 0.8 — Sprint 1 will pick this back up.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/smoke/test_nerve_wml_api.py
git commit -m "feat: pin nerve-wml dependency + smoke test for required public API"
```

---

## Task 0.8: OQ1 spike — shared vs local codebook

**Goal of this task:** Resolve open question OQ1 from the spec (shared 64-code alphabet vs per-WML local codebook with `CodebookAligner`). Output = a `docs/adr/0001-codebook-sharing.md` ADR with a choice, evidence, and revisit criteria.

**Files:**
- Create: `scripts/spikes/oq1_codebook_sharing.py`
- Create: `docs/adr/0001-codebook-sharing.md`
- Create: `docs/adr/README.md`

- [ ] **Step 1: Create ADR directory + index**

`docs/adr/README.md`:
```markdown
# Architecture Decision Records

ADRs capture decisions whose reversal is expensive. They are numbered
sequentially and never deleted — superseded decisions are marked "Superseded
by ADR-NNNN" at the top.

## Index

| # | Title | Status |
|---|-------|--------|
| 0001 | Codebook sharing between SensoryWMLs | Proposed |
```

- [ ] **Step 2: Write the spike script**

`scripts/spikes/oq1_codebook_sharing.py`:
```python
"""OQ1 spike — shared vs local codebook on a toy 2-modality task.

Runs a miniature controlled experiment to decide whether the 64-code alphabet
should be shared across SensoryWMLs (v0.1 default) or whether per-WML codebooks
plus a CodebookAligner give measurably better adaptation.

Usage:
    uv run python scripts/spikes/oq1_codebook_sharing.py --mode shared
    uv run python scripts/spikes/oq1_codebook_sharing.py --mode local
    uv run python scripts/spikes/oq1_codebook_sharing.py --mode both --report out/oq1.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn, optim


Mode = Literal["shared", "local", "both"]


@dataclass
class SpikeResult:
    mode: str
    final_accuracy: float
    final_loss: float
    steps: int
    seed: int


class ToyTwoModalityTask:
    """Binary classification on two coherent modalities derived from a shared latent."""

    def __init__(self, d_z: int = 8, noise: float = 0.1) -> None:
        self.d_z = d_z
        self.noise = noise
        self.W_a = torch.randn(d_z, 16)
        self.W_v = torch.randn(d_z, 16)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.randn(batch_size, self.d_z)
        audio = z @ self.W_a + self.noise * torch.randn(batch_size, 16)
        vision = z @ self.W_v + self.noise * torch.randn(batch_size, 16)
        label = (z[:, 0] > 0).long()
        return audio, vision, label


class SharedCodebookModel(nn.Module):
    """Both modalities project into a shared 64-code alphabet, fused by mean."""

    def __init__(self, k_codes: int = 64) -> None:
        super().__init__()
        self.audio_enc = nn.Linear(16, k_codes)
        self.vision_enc = nn.Linear(16, k_codes)
        self.head = nn.Linear(k_codes, 2)

    def forward(self, audio: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        a = torch.softmax(self.audio_enc(audio), dim=-1)
        v = torch.softmax(self.vision_enc(vision), dim=-1)
        fused = (a + v) / 2.0
        return self.head(fused)


class LocalCodebookModel(nn.Module):
    """Each modality has its own 64-code alphabet; an aligner learns to match."""

    def __init__(self, k_codes: int = 64) -> None:
        super().__init__()
        self.audio_enc = nn.Linear(16, k_codes)
        self.vision_enc = nn.Linear(16, k_codes)
        self.aligner = nn.Linear(2 * k_codes, k_codes)
        self.head = nn.Linear(k_codes, 2)

    def forward(self, audio: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        a = torch.softmax(self.audio_enc(audio), dim=-1)
        v = torch.softmax(self.vision_enc(vision), dim=-1)
        fused = self.aligner(torch.cat([a, v], dim=-1))
        return self.head(fused)


def train(model: nn.Module, task: ToyTwoModalityTask, steps: int, seed: int) -> SpikeResult:
    torch.manual_seed(seed)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    losses: list[float] = []
    for _ in range(steps):
        audio, vision, label = task.sample(batch_size=128)
        logits = model(audio, vision)
        loss = nn.functional.cross_entropy(logits, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    audio, vision, label = task.sample(batch_size=1024)
    with torch.no_grad():
        preds = model(audio, vision).argmax(dim=-1)
        acc = (preds == label).float().mean().item()
    return SpikeResult(
        mode=type(model).__name__,
        final_accuracy=acc,
        final_loss=losses[-1],
        steps=steps,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["shared", "local", "both"], default="both")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--report", type=Path, default=Path("out/oq1_results.json"))
    args = parser.parse_args()

    task = ToyTwoModalityTask()
    all_results: list[SpikeResult] = []

    for seed in range(args.seeds):
        if args.mode in ("shared", "both"):
            all_results.append(train(SharedCodebookModel(), task, args.steps, seed))
        if args.mode in ("local", "both"):
            all_results.append(train(LocalCodebookModel(), task, args.steps, seed))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps([asdict(r) for r in all_results], indent=2))

    grouped: dict[str, list[float]] = {}
    for r in all_results:
        grouped.setdefault(r.mode, []).append(r.final_accuracy)
    print("\n=== OQ1 Spike Results ===")
    for mode, accs in grouped.items():
        mean = sum(accs) / len(accs)
        std = math.sqrt(sum((a - mean) ** 2 for a in accs) / max(1, len(accs) - 1))
        print(f"{mode:<24s}  acc = {mean:.4f} ± {std:.4f}  (n={len(accs)})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the spike**

Run:
```bash
mkdir -p out
uv run python scripts/spikes/oq1_codebook_sharing.py --mode both --seeds 5 --steps 2000 --report out/oq1_results.json
```
Expected: console output of form
```
=== OQ1 Spike Results ===
SharedCodebookModel       acc = 0.9XXX ± 0.0XXX  (n=5)
LocalCodebookModel        acc = 0.9XXX ± 0.0XXX  (n=5)
```
and a JSON file at `out/oq1_results.json`.

- [ ] **Step 4: Interpret and commit results**

Rule of thumb: if `|shared − local|` ≤ 0.02 (2 % accuracy), shared wins by Occam; if `local > shared + 0.05`, local wins; otherwise re-run with 10 seeds or deeper architecture before deciding.

- [ ] **Step 5: Write ADR 0001**

`docs/adr/0001-codebook-sharing.md`:
```markdown
# ADR-0001 — Codebook sharing between SensoryWMLs

**Status:** Proposed (Sprint 0)
**Date:** 2026-04-20
**Authors:** Clément Saillant
**Related:** Spec `2026-04-20-bouba-sens-design.md` OQ1 + §2.2 principle 2.

## Context

`bouba_sens` couples five `SensoryWML` instances through a `CrossModalNerve`
that routes neuroletters (64-code alphabet) between them. The `nerve-wml`
invariant **N-5** says each WML has a local codebook. For cross-modal
compensation to work, codes must have *comparable semantics* across WMLs —
either by sharing the alphabet outright (violation of N-5) or by adding a
learnable `CodebookAligner` module.

## Decision

<!--
Fill this in after running the spike (Step 3). Template:

**Chosen:** SHARED 64-code alphabet across SensoryWMLs for v0.1.

**Evidence:**
- Spike results from `out/oq1_results.json` (5 seeds × 2000 steps × 2 modes).
- Shared:  acc = <XX.XX>% ± <X.XX>%
- Local:   acc = <XX.XX>% ± <X.XX>%
- Delta: <sign and magnitude>.
-->

## Consequences

- Explicit, documented violation of nerve-wml invariant N-5, scoped to this
  repository. A note is added to `src/bouba_sens/nerve.py` docstring when
  Sprint 2 implements `CrossModalNerve`.
- If v0.2 empirical results (§4.5, §7.3 R4) show compensation degeneracy, we
  revisit and potentially add a `CodebookAligner` — producing a new ADR that
  supersedes this one.

## nerve-wml API gap observations (from Task 0.7)

<!-- Leave empty if all 5 required symbols were present. Otherwise list them
here so the next iteration of the plan picks up the upstream work. -->

## Revisit criteria

Re-open this ADR if any of the following holds:

1. Sprint 2 integration shows > 10 % accuracy degradation vs the spike baseline.
2. An empirical test `test_B2_mi_migration` fails consistently across seeds.
3. A peer review (internal or external) raises the violation of N-5 as a
   correctness concern.
```

Fill in the empty `<XX.XX>` placeholders in the "Decision" block using the numbers printed in Step 3.

- [ ] **Step 6: Update the ADR index**

Edit `docs/adr/README.md` to change row `0001` status from `Proposed` to `Accepted` after filling in the decision block.

- [ ] **Step 7: Commit**

```bash
git add scripts/spikes/oq1_codebook_sharing.py docs/adr/ out/oq1_results.json
git commit -m "spike(oq1): shared vs local codebook decision + ADR 0001"
```

---

## Task 0.9: Configs skeleton (hydra)

**Files:**
- Create: `configs/README.md`
- Create: `configs/v0.1_intact.yaml`

- [ ] **Step 1: Write `configs/README.md`**

```markdown
# Configs

Hydra-style YAML configuration files for bouba_sens experiments.

## Layout

- `v0.1_intact.yaml` — Sprint 0 placeholder with the Sprint 1-ready
  skeleton (world, architecture, lesion = NONE, adaptation, metrics).
- (future) `t1_<modality>_m2.yaml` / `t2_<modality>_m2.yaml` — one config
  per lesion arm, generated in Sprint 2.
- (future) `v0.1_full_grid.yaml` — entrypoint that spawns the 150-run grid
  in Sprint 3.
```

- [ ] **Step 2: Write `configs/v0.1_intact.yaml`**

```yaml
# Sprint-0 intact-run config skeleton. Populated in Sprint 1.

defaults:
  - _self_

world:
  kind: gaussian       # one of: gaussian, xor, sinusoid (Sprint 1)
  d_z: 8
  batch_size: 128

architecture:
  k_letters: 64        # shared alphabet (see ADR-0001)
  hidden: 128
  n_modalities: 5

lesion:
  mode: null           # no lesion in intact runs

adaptation:
  pretrain_steps: 10000
  lesion_steps: 0
  optimizer: adam
  lr: 1.0e-3
  replay_buffer_size: 1024

metrics:
  enabled: [Me1, Me2, Me8, Me9]

run:
  seed: 0
  out_dir: runs/
  precision: float32
```

- [ ] **Step 3: Validate YAML parses**

Run:
```bash
uv run python -c "import yaml; yaml.safe_load(open('configs/v0.1_intact.yaml')); print('ok')"
```
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add configs/
git commit -m "chore: add configs skeleton + v0.1 intact baseline YAML"
```

---

## Task 0.10: CHANGELOG + tag v0.1-sprint0

**Files:**
- Create: `CHANGELOG.md`

- [ ] **Step 1: Write `CHANGELOG.md`**

```markdown
# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] — 2026-04-20

### Added

- Initial design spec (`docs/superpowers/specs/2026-04-20-bouba-sens-design.md`).
- Sprint 0 plan (`docs/superpowers/plans/2026-04-20-bouba-sens-sprint0.md`).
- Package skeleton `src/bouba_sens/` with module stubs for sensory, nerve,
  lesion, head, loop, cli, report, world/*, metrics/*.
- CLI entrypoint `bouba-sens version`.
- Pinned dependency on `nerve-wml >=1.1.4,<1.2`.
- Smoke test `tests/smoke/test_nerve_wml_api.py` verifying the 5 required
  public symbols (spec §6.3).
- Pre-commit hooks (ruff, mypy, hygiene).
- GitHub Actions: CI workflow (lint + mypy + unit/smoke tests) and stubbed
  nightly full-benchmark workflow.
- ADR-0001 — codebook sharing decision (OQ1 resolution).
- Hydra-style config skeleton (`configs/v0.1_intact.yaml`).
- LICENSE (MIT), README, CITATION.cff, SECURITY.md.
```

- [ ] **Step 2: Commit CHANGELOG**

```bash
git add CHANGELOG.md
git commit -m "docs: add CHANGELOG with 0.0.1 Sprint 0 entry"
```

- [ ] **Step 3: Tag the Sprint 0 release**

Run:
```bash
git tag -a v0.0.1-sprint0 -m "Sprint 0 complete — scaffolding + nerve-wml contract + OQ1 ADR"
git tag -l
```
Expected output: `v0.0.1-sprint0` is listed.

- [ ] **Step 4: Verify full test suite is green**

Run:
```bash
uv run pytest tests/ -v
uv run pre-commit run --all-files
uv run mypy src
```
Expected: all tests PASS, pre-commit PASS, mypy 0 errors. **If anything fails, do not proceed to Sprint 1 — fix the breakage and bump to `v0.0.1-sprint0+1`.**

- [ ] **Step 5: Final sanity check — readable by a cold reader**

Run:
```bash
ls -la .
cat README.md | head -30
uv run bouba-sens --help
```
Expected: tree is clean, README explains the project, CLI surfaces `version` as a command.

---

## Sprint 0 exit criteria (go/no-go to Sprint 1)

All of the following MUST be true before the Sprint 1 plan is written:

- [ ] `git tag` lists `v0.0.1-sprint0`.
- [ ] `uv run pytest tests/` exits 0.
- [ ] `uv run pre-commit run --all-files` exits 0.
- [ ] `uv run mypy src` reports 0 errors.
- [ ] `docs/adr/0001-codebook-sharing.md` has a filled-in `## Decision` block.
- [ ] `out/oq1_results.json` exists and contains ≥ 10 rows (5 seeds × 2 modes).
- [ ] `tests/smoke/test_nerve_wml_api.py` passes (or all failures are logged as open nerve-wml issues in the ADR).
- [ ] GitHub repo exists at `hypneum-lab/bouba_sens` (or `genial-lab/bouba_sens` fallback with a post-rename transfer plan noted in `CHANGELOG.md`).

When all boxes are checked, invoke `superpowers:writing-plans` again with
argument `sprint=1` to produce `docs/superpowers/plans/2026-XX-XX-bouba-sens-sprint1.md`.
