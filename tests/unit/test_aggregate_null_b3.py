"""Task 7.1 — aggregate_grid.py null-partition CLI flag-pair validation."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_both_flags_absent_uses_prereg_partition(tmp_path: Path) -> None:
    """When neither flag is provided, no argparse error about partition flags."""
    runs = tmp_path / "runs"
    runs.mkdir()
    out = tmp_path / "aggregate.json"
    proc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/aggregate_grid.py",
            "--root",
            str(runs),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    # No flag-pair error — any other error (empty grid etc.) is acceptable
    assert "partition-seed" not in proc.stderr
    assert "partition-index" not in proc.stderr


def test_only_partition_seed_without_index_errors(tmp_path: Path) -> None:
    """Specifying only --partition-seed without --partition-index must error."""
    runs = tmp_path / "runs"
    runs.mkdir()
    out = tmp_path / "aggregate.json"
    proc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/aggregate_grid.py",
            "--root",
            str(runs),
            "--out",
            str(out),
            "--partition-seed",
            "0",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "partition" in combined.lower()


def test_only_partition_index_without_seed_errors(tmp_path: Path) -> None:
    """Specifying only --partition-index without --partition-seed must error."""
    runs = tmp_path / "runs"
    runs.mkdir()
    out = tmp_path / "aggregate.json"
    proc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/aggregate_grid.py",
            "--root",
            str(runs),
            "--out",
            str(out),
            "--partition-index",
            "0",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "partition" in combined.lower()
