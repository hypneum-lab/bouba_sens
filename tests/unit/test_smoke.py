"""Sanity tests that prove the Sprint 0 skeleton actually loads."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import bouba_sens

_PYPROJECT_VERSION = tomllib.loads(
    (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
)["project"]["version"]


def test_package_imports() -> None:
    """`bouba_sens.__version__` derives from pyproject (N4 Task 6, 2026-05-10)."""
    assert bouba_sens.__version__ == _PYPROJECT_VERSION, (
        f"pyproject={_PYPROJECT_VERSION}, package={bouba_sens.__version__}"
    )


def test_cli_version_runs() -> None:
    """`bouba-sens version` should exit 0 and print the pyproject version."""

    result = subprocess.run(
        ["uv", "run", "bouba-sens", "version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert f"bouba_sens {_PYPROJECT_VERSION}" in result.stdout


def test_world_sample_dataclass_importable() -> None:
    from bouba_sens.world import WorldSample, WorldSimulator  # noqa: F401


def test_modality_type_constants_are_five() -> None:
    from bouba_sens.sensory import MODALITIES

    assert MODALITIES == ("audio", "vision", "tactile", "gravity", "force")
    assert len(MODALITIES) == 5
