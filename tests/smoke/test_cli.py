"""Smoke test — verify the Typer CLI prints the package version."""

from __future__ import annotations

from typer.testing import CliRunner

from bouba_sens._version import __version__
from bouba_sens.cli import app


def test_cli_version_prints_package_version() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"bouba_sens {__version__}" in result.stdout
