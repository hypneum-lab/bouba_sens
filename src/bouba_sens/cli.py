"""Typer-based CLI. See spec §5.3.

Sprint 0 ships only `version` so `bouba-sens version` works after `uv sync`.
"""

from __future__ import annotations

import typer

from bouba_sens._version import __version__

app = typer.Typer(help="bouba_sens — Cross-modal plasticity benchmark", no_args_is_help=True)


@app.callback()
def _main() -> None:
    """bouba_sens — Cross-modal plasticity benchmark."""


@app.command()
def version() -> None:
    """Print the package version."""

    typer.echo(f"bouba_sens {__version__}")


if __name__ == "__main__":
    app()
