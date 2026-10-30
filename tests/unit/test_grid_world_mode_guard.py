"""ADR-0027 — grid scripts: WORLD_SEED_MODE pass-through and resume guard.

The scripts run against a fake ``uv`` that only logs its arguments, so no
cell is computed.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
GRIDS = ("run_grid.sh", "run_grid_probe_v2.sh")
CELL = "seed400_T1_audio_floor"


@pytest.mark.parametrize("script", GRIDS)
def test_grid_script_syntax(script: str) -> None:
    subprocess.run(["bash", "-n", str(SCRIPTS / script)], check=True)


def _fake_env(tmp_path: Path, mode: str | None) -> tuple[dict[str, str], Path]:
    home = tmp_path / "home"
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    log = tmp_path / "uv.log"
    uv = bin_dir / "uv"
    uv.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{log}"\n')
    uv.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("WORLD_SEED_MODE", "OUT_ROOT", "SEEDS")}
    env |= {
        "HOME": str(home),
        "PATH": f"{bin_dir}:{env['PATH']}",
        "OUT_ROOT": str(tmp_path / "grid"),
        "SEEDS": "400",
        "PAR": "2",
    }
    if mode is not None:
        env["WORLD_SEED_MODE"] = mode
    return env, log


def _finished_cell(tmp_path: Path, recorded: str | None) -> None:
    cell = tmp_path / "grid" / CELL
    cell.mkdir(parents=True)
    (cell / "eval_report.json").write_text("{}")
    (cell / "probe_v2_meta.json").write_text('{"probe_seed": 4005000}')
    cfg = "phase: lesion\nseed: 4000001\n"
    if recorded is not None:
        cfg += f"world_seed_mode: {recorded}\nworld_seed: 400\n"
    (cell / "config.yaml").write_text(cfg)


def _run(script: str, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPTS / script)], env=env, cwd=cwd, capture_output=True, text=True
    )


@pytest.mark.parametrize("script", GRIDS)
@pytest.mark.parametrize(("recorded", "mode"), [("cell", None), ("cell", "legacy"), (None, "cell")])
def test_resume_refuses_a_cell_of_another_mode(
    script: str, recorded: str | None, mode: str | None, tmp_path: Path
) -> None:
    env, log = _fake_env(tmp_path, mode)
    _finished_cell(tmp_path, recorded)
    res = _run(script, env, tmp_path)
    assert res.returncode != 0
    assert f"world_seed_mode {recorded or 'legacy'}" in res.stderr
    assert "Use one OUT_ROOT per arm" in res.stderr
    logged = log.read_text() if log.exists() else ""
    assert "bouba-sens lesion" not in logged


@pytest.mark.parametrize("script", GRIDS)
@pytest.mark.parametrize(("recorded", "mode"), [(None, None), (None, "legacy"), ("cell", "cell")])
def test_resume_accepts_a_cell_of_the_same_mode(
    script: str, recorded: str | None, mode: str | None, tmp_path: Path
) -> None:
    env, log = _fake_env(tmp_path, mode)
    _finished_cell(tmp_path, recorded)
    res = _run(script, env, tmp_path)
    assert res.returncode == 0, res.stderr
    lesions = [line for line in log.read_text().splitlines() if "bouba-sens lesion" in line]
    assert len(lesions) == 29  # 30 cells, one already finished
    assert all(CELL not in line for line in lesions)
    if mode == "cell":
        assert all("--world-seed-mode cell --world-seed 400" in line for line in lesions)
    else:
        assert all("--world-seed-mode" not in line for line in lesions)


@pytest.mark.parametrize("script", GRIDS)
def test_preflight_checks_every_seed_in_out_root(script: str, tmp_path: Path) -> None:
    """A finished cell of another mode stops the grid before any cell runs.

    SEEDS="5 0": seed 5 comes first and has nothing done; seed 0 has one
    finished legacy cell. In cell mode no lesion (and no phase 1) may run.
    """
    env, log = _fake_env(tmp_path, "cell")
    env["SEEDS"] = "5 0"
    cell = tmp_path / "grid" / "seed0_T1_audio_floor"
    cell.mkdir(parents=True)
    (cell / "eval_report.json").write_text("{}")
    (cell / "probe_v2_meta.json").write_text('{"probe_seed": 5000}')
    (cell / "config.yaml").write_text("phase: lesion\nseed: 1\n")
    res = _run(script, env, tmp_path)
    assert res.returncode != 0
    assert "world_seed_mode legacy" in res.stderr
    logged = log.read_text() if log.exists() else ""
    assert "bouba-sens lesion" not in logged
    assert "bouba-sens train" not in logged
