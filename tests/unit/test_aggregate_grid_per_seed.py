"""Sprint 14a — per-seed aggregator acceptance tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from aggregate_grid_per_seed import (  # type: ignore[import-not-found]  # noqa: E402
    aggregate_per_seed,
)


def _seed_cell(root: Path, name: str, **metrics: float) -> None:
    cell = root / name
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "eval_report.json").write_text(json.dumps(metrics))


def test_per_seed_isolates_each_seed(tmp_path: Path) -> None:
    """Seed 0..3: Me7 = +0.10 (T1-T2 paired). Seed 4: Me7 = -0.05."""
    for seed in range(4):
        _seed_cell(tmp_path, f"seed{seed}_T1_audio_floor", me1=0.70)
        _seed_cell(tmp_path, f"seed{seed}_T2_audio_floor", me1=0.60)
    _seed_cell(tmp_path, "seed4_T1_audio_floor", me1=0.55)
    _seed_cell(tmp_path, "seed4_T2_audio_floor", me1=0.60)

    result = aggregate_per_seed(tmp_path)
    assert set(result["per_seed"]) == {"0", "1", "2", "3", "4"}

    for s in ("0", "1", "2", "3"):
        assert result["per_seed"][s]["b1"]["median_me7"] > 0.05
        assert result["per_seed"][s]["b1"]["passes"] is True
    assert result["per_seed"]["4"]["b1"]["median_me7"] < 0.0
    assert result["per_seed"]["4"]["b1"]["passes"] is False

    agg = result["aggregate"]["b1"]
    assert agg["sign_stability"] == "4/5 positive"
    assert agg["passes_count"] == "4/5"
    assert len(agg["b1_values"]) == 5


def test_aggregate_std_nonzero_when_seeds_disagree(tmp_path: Path) -> None:
    """Regression guard: std must reflect cross-seed variance, not collapse to 0."""
    for seed, (t1, t2) in enumerate([(0.70, 0.60), (0.80, 0.50), (0.65, 0.60)]):
        _seed_cell(tmp_path, f"seed{seed}_T1_audio_floor", me1=t1)
        _seed_cell(tmp_path, f"seed{seed}_T2_audio_floor", me1=t2)
    result = aggregate_per_seed(tmp_path)
    assert result["aggregate"]["b1"]["b1_std"] > 0.0
