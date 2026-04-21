"""Tasks 4.2 + 4.3 + 5.2 acceptance tests for the grid aggregator."""

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

MODALITIES = ("audio", "vision", "tactile", "gravity", "force")


def _seed_cell(root: Path, name: str, **metrics: float) -> None:
    cell = root / name
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "eval_report.json").write_text(json.dumps(metrics))


def _seed_cell_with_query(
    root: Path, name: str, per_query: dict[str, float], **metrics: float
) -> None:
    cell = root / name
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "eval_report.json").write_text(json.dumps(metrics))
    (cell / "per_query_me1.json").write_text(json.dumps(per_query))


def test_aggregate_packs_cells_and_invariants(tmp_path: Path) -> None:
    for seed in range(5):
        _seed_cell(
            tmp_path,
            f"seed{seed}_T2_audio_floor",
            me1=0.6 + 0.01 * seed,
            me3_delta=0.15,
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


def test_aggregate_b1_passes_when_t1_beats_t2(tmp_path: Path) -> None:
    """Sprint 5: Me7 is now computed from paired T1/T2 me1 across seeds."""
    for seed in range(5):
        _seed_cell(tmp_path, f"seed{seed}_T1_audio_floor", me1=0.70)
        _seed_cell(tmp_path, f"seed{seed}_T2_audio_floor", me1=0.60)
    result = aggregate(tmp_path)
    assert result["invariants"]["b1"]["passes"] is True
    assert result["invariants"]["b1"]["median_me7"] > B1_ME7_THRESHOLD
    assert result["invariants"]["b1"]["cells_counted"] == 5


def test_aggregate_b1_fails_when_t1_equals_t2(tmp_path: Path) -> None:
    for seed in range(5):
        _seed_cell(tmp_path, f"seed{seed}_T1_audio_floor", me1=0.60)
        _seed_cell(tmp_path, f"seed{seed}_T2_audio_floor", me1=0.60)
    result = aggregate(tmp_path)
    assert result["invariants"]["b1"]["passes"] is False


def test_aggregate_b3_passes_on_asymmetric_perf_matrix(tmp_path: Path) -> None:
    """Sprint 5: Me6 is computed by stacking per-query-me1 across the 5 lesioned-modality cells."""
    # Build a single (seed=0, timing=T2, snr=floor) trio with 5 modality cells.
    # Make the matrix clearly asymmetric: column j (lesioned=j) has query j at 1.0,
    # and an off-diagonal row that is non-uniform, so A = perf - perf.T is non-zero.
    perf = {
        "audio": {"audio": 1.0, "vision": 0.2, "tactile": 0.3, "gravity": 0.4, "force": 0.5},
        "vision": {"audio": 0.6, "vision": 1.0, "tactile": 0.3, "gravity": 0.4, "force": 0.5},
        "tactile": {"audio": 0.6, "vision": 0.7, "tactile": 1.0, "gravity": 0.4, "force": 0.5},
        "gravity": {"audio": 0.6, "vision": 0.7, "tactile": 0.8, "gravity": 1.0, "force": 0.5},
        "force": {"audio": 0.6, "vision": 0.7, "tactile": 0.8, "gravity": 0.9, "force": 1.0},
    }
    for lesioned, col in perf.items():
        _seed_cell_with_query(tmp_path, f"seed0_T2_{lesioned}_floor", per_query=col, me1=0.5)
    result = aggregate(tmp_path)
    assert result["invariants"]["b3"]["cells_counted"] == 1
    assert result["invariants"]["b3"]["median_me6_max_abs"] > B3_ME6_THRESHOLD
    assert result["invariants"]["b3"]["passes"] is True


def test_aggregate_skips_unparseable_dirs(tmp_path: Path) -> None:
    (tmp_path / "phase1_seed0").mkdir()
    (tmp_path / "phase1_seed0" / "checkpoint.pkl").write_text("dummy")
    _seed_cell(tmp_path, "seed0_T2_audio_floor", me1=0.5)
    result = aggregate(tmp_path)
    assert "t2_audio_floor" in result["cells"]
    assert len(result["cells"]) == 1


def test_b2_cells_counted_zero_when_me3_delta_missing(tmp_path: Path) -> None:
    """Regression guard: eval_report.json without `me3_delta` must yield b2.cells_counted == 0.

    This locks the failure mode for Sprint 7 micro-fix G1 — if a future
    commit regresses `scripts/run_grid.sh` to emit only Me1/Me2 (dropping
    Me3), the aggregator produces no Me3 values and B-2 must report
    cells_counted == 0 rather than silently passing on an empty median.
    """
    # Seed 3 cells with me1+me2 but NO me3_delta key.
    for seed in range(3):
        _seed_cell(tmp_path, f"seed{seed}_T2_audio_floor", me1=0.60, me2=0.55)
    result = aggregate(tmp_path)
    b2 = result["invariants"]["b2"]
    assert b2["cells_counted"] == 0
    assert b2["median_me3_delta"] == 0.0
    assert b2["passes"] is False
