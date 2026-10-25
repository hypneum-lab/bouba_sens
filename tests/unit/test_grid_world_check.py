"""Every B-1 grid output records its world class (PI decision 2026-10-25)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bouba_sens.cli import app

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from aggregate_grid import (  # type: ignore[import-not-found]  # noqa: E402
    WorldCheckError,
    check_world,
)

REAL = "StudyforrestRealWorld"


def _out(root: Path, name: str, meta: dict | None) -> None:
    d = root / name
    d.mkdir(parents=True)
    if meta is not None:
        (d / "metadata.json").write_text(json.dumps(meta))


def _grid(root: Path, world_class: str | None = REAL) -> None:
    meta = {} if world_class is None else {"world_class": world_class}
    _out(root, "phase1_seed4901", meta)
    for name in ("seed4901_T2_audio_floor", "seed4901_T1_vision_plus10"):
        _out(root, name, meta)


def test_check_accepts_a_grid_on_the_real_world(tmp_path: Path) -> None:
    _grid(tmp_path)
    (tmp_path / "unrelated_dir").mkdir()
    assert check_world(tmp_path, REAL) == 3


@pytest.mark.parametrize(
    ("world_class", "match"),
    [(None, "no world_class"), ("StudyforrestWorld", "expected"), ("GaussianWorld", "expected")],
)
def test_check_refuses_missing_or_other_world(
    tmp_path: Path, world_class: str | None, match: str
) -> None:
    _grid(tmp_path, world_class)
    with pytest.raises(WorldCheckError, match=match):
        check_world(tmp_path, REAL)


def test_check_refuses_missing_metadata_and_empty_grid(tmp_path: Path) -> None:
    with pytest.raises(WorldCheckError, match="no grid outputs"):
        check_world(tmp_path, REAL)
    _out(tmp_path, "seed4901_T2_audio_floor", None)
    with pytest.raises(WorldCheckError, match="no metadata"):
        check_world(tmp_path, REAL)


def test_cli_writes_world_class_and_snr_trajectory(tmp_path: Path) -> None:
    runner = CliRunner()
    p1 = tmp_path / "phase1_seed9001"
    r = runner.invoke(app, ["train", "--steps", "2", "--seed", "9001", "--out", str(p1)])
    assert r.exit_code == 0, r.output
    assert json.loads((p1 / "metadata.json").read_text())["world_class"] == "GaussianWorld"
    cell = tmp_path / "seed9001_T2_audio_floor"
    r = runner.invoke(
        app,
        ["lesion", "--ckpt", str(p1), "--steps", "60", "--seed", "90010", "--k-steps", "50",
         "--snr-floor", "-20", "--out", str(cell)],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    meta = json.loads((cell / "metadata.json").read_text())
    assert meta["world_class"] == "GaussianWorld"
    traj = meta["snr_trajectory_db"]
    assert len(traj) == 60 and traj[0] == 20.0
    assert traj[50] == pytest.approx(-20.0) and traj[59] == pytest.approx(-20.0)
    assert traj[49] > -20.0
    assert check_world(tmp_path, "GaussianWorld") == 2
