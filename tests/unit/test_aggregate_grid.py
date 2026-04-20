"""Tasks 4.2 + 4.3 acceptance tests for the grid aggregator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from aggregate_grid import (  # type: ignore[import-not-found]  # noqa: E402
    B1_ME7_THRESHOLD,
    B2_ME3_DELTA_THRESHOLD,
    B3_ME6_THRESHOLD,
    aggregate,
)


def _seed_cell(root: Path, name: str, **metrics: float) -> None:
    cell = root / name
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "eval_report.json").write_text(json.dumps(metrics))


def test_aggregate_packs_cells_and_invariants(tmp_path: Path) -> None:
    # Seed two T2 audio cells (floor + minus10) across 5 seeds.
    for seed in range(5):
        _seed_cell(
            tmp_path,
            f"seed{seed}_T2_audio_floor",
            me1=0.6 + 0.01 * seed,
            me7=0.08 + 0.001 * seed,
            me3_delta=0.15,
            me6_max_abs=0.03,
        )
    result = aggregate(tmp_path)
    assert "cells" in result
    assert "invariants" in result
    assert "thresholds" in result
    assert "t2_audio_floor" in result["cells"]

    b1 = result["cells"]["t2_audio_floor"]["me1"]
    assert b1["ci_low"] <= b1["mean"] <= b1["ci_high"]


def test_aggregate_thresholds_match_spec(tmp_path: Path) -> None:
    """The three invariant thresholds must match spec §1.2 exactly."""
    assert B1_ME7_THRESHOLD == 0.05
    assert B2_ME3_DELTA_THRESHOLD == 0.10
    assert B3_ME6_THRESHOLD == 0.02


def test_aggregate_b1_passes_when_me7_above_threshold(tmp_path: Path) -> None:
    for seed in range(5):
        _seed_cell(tmp_path, f"seed{seed}_T2_audio_floor", me7=0.08)
        _seed_cell(tmp_path, f"seed{seed}_T2_vision_floor", me7=0.07)
    result = aggregate(tmp_path)
    assert result["invariants"]["b1"]["passes"] is True
    assert result["invariants"]["b1"]["median_me7"] > B1_ME7_THRESHOLD


def test_aggregate_b1_fails_when_me7_below_threshold(tmp_path: Path) -> None:
    for seed in range(5):
        _seed_cell(tmp_path, f"seed{seed}_T2_audio_floor", me7=0.01)
    result = aggregate(tmp_path)
    assert result["invariants"]["b1"]["passes"] is False


def test_aggregate_skips_unparseable_dirs(tmp_path: Path) -> None:
    # A non-grid directory in the same root must be silently ignored.
    (tmp_path / "phase1_seed0").mkdir()
    (tmp_path / "phase1_seed0" / "checkpoint.pkl").write_text("dummy")
    _seed_cell(tmp_path, "seed0_T2_audio_floor", me1=0.5)
    result = aggregate(tmp_path)
    assert "t2_audio_floor" in result["cells"]
    assert len(result["cells"]) == 1
