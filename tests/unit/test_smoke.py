"""Sanity tests that prove the Sprint 0 skeleton actually loads."""

from __future__ import annotations

import subprocess

import bouba_sens


def test_package_imports() -> None:
    assert bouba_sens.__version__ == "0.1.0"


def test_cli_version_runs() -> None:
    """bouba-sens version should exit 0 and print the version."""

    result = subprocess.run(
        ["uv", "run", "bouba-sens", "version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "bouba_sens 0.1.0" in result.stdout


def test_world_sample_dataclass_importable() -> None:
    from bouba_sens.world import WorldSample, WorldSimulator  # noqa: F401


def test_modality_type_constants_are_five() -> None:
    from bouba_sens.sensory import MODALITIES

    assert MODALITIES == ("audio", "vision", "tactile", "gravity", "force")
    assert len(MODALITIES) == 5
