"""Unit tests for ``scripts/replay_amedi_seeds.py`` (Q3 wrapper).

The real pipeline is Studio-gated (10-20 h on M3 Ultra), so we mock
the subprocess runner and stage fake aggregate JSON files on disk to
verify the wrapper composes its outputs correctly. See
``docs/milestones/q3-amedi-seeds-2026-05-10.md``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "replay_amedi_seeds.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "replay_amedi_seeds",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["replay_amedi_seeds"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def staged_runner(tmp_path: Path):
    """Yield (runner, reports_dir, runs_root) where runner stages fake
    aggregate JSON files on every aggregate_grid.py call."""
    reports_dir = tmp_path / "reports"
    runs_root = tmp_path / "runs"
    reports_dir.mkdir()
    runs_root.mkdir()

    # Map (seed, lock_after) -> median_me7 to inject.
    me7_by_cell: dict[tuple[int, int], float] = {
        (0, 50): 0.010,
        (0, 75): 0.018,
        (0, 100): 0.025,
        (0, 125): 0.020,
        (0, 150): 0.012,
    }

    def runner(cmd: list[str], **_: Any) -> Any:
        # Detect aggregate_grid.py invocation and stage the JSON.
        if "aggregate_grid.py" in " ".join(cmd):
            out_idx = cmd.index("--out") + 1
            out_path = Path(cmd[out_idx])
            # Parse seed + lock from filename: v0.5_dr_lock{LA}_seed{S}_aggregate.json
            stem = out_path.stem
            lock_token = stem.split("_lock")[1].split("_")[0]
            seed_token = stem.split("_seed")[1].split("_")[0]
            key = (int(seed_token), int(lock_token))
            me7 = me7_by_cell.get(key, 0.0)
            out_path.write_text(
                json.dumps(
                    {
                        "invariants": {"b1": {"median_me7": me7}},
                    }
                )
            )

        class _Result:
            returncode = 0

        return _Result()

    return runner, reports_dir, runs_root


def test_run_single_seed_returns_peak(tmp_path: Path, staged_runner) -> None:
    module = _load_module()
    runner, reports_dir, runs_root = staged_runner

    result = module.run_single_seed(
        seed=0,
        lock_after_values=[50, 75, 100, 125, 150],
        runs_root=runs_root,
        reports_dir=reports_dir,
        studyforrest_data=tmp_path / "data",
        runner=runner,
    )

    assert result["seed"] == 0
    assert result["lock_after_values"] == [50, 75, 100, 125, 150]
    assert result["b1_values"] == [0.010, 0.018, 0.025, 0.020, 0.012]
    assert result["peak_position"] == 100
    assert result["peak_height"] == pytest.approx(0.025)


def test_run_single_lock_after_invokes_grid_and_aggregate(
    tmp_path: Path,
    staged_runner,
) -> None:
    module = _load_module()
    runner, reports_dir, runs_root = staged_runner

    calls: list[list[str]] = []

    def recording_runner(cmd: list[str], **kw: Any) -> Any:
        calls.append(list(cmd))
        return runner(cmd, **kw)

    me7 = module._run_single_lock_after(
        seed=0,
        lock_after=100,
        runs_root=runs_root,
        reports_dir=reports_dir,
        studyforrest_data=tmp_path / "data",
        runner=recording_runner,
    )

    assert me7 == pytest.approx(0.025)
    assert any("run_grid.sh" in " ".join(c) for c in calls)
    assert any("aggregate_grid.py" in " ".join(c) for c in calls)
