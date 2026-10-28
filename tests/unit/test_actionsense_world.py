"""ActionSenseWorld real-data bridge: shapes, alignment, CLI dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import typer

from bouba_sens.cli import _build_world
from bouba_sens.world.actionsense import ActionSenseWorld


def _write_cache(path: Path, n: int = 256, vision_rows: int | None = None) -> Path:
    torch.save(torch.randn(n, 128), path / "audio.pt")
    torch.save(torch.rand(vision_rows or n, 16, 16), path / "vision.pt")
    torch.save(torch.randn(n, 32), path / "tactile.pt")
    torch.save(torch.randn(n, 3), path / "gravity.pt")
    torch.save(torch.randn(n, 6), path / "force.pt")
    torch.save(torch.randint(0, 4, (n,), dtype=torch.long), path / "labels.pt")
    return path


def test_actionsense_mode(tmp_path: Path) -> None:
    world = ActionSenseWorld(seed=0, data_dir=_write_cache(tmp_path))
    assert world.mode == "actionsense"


def test_actionsense_sample_shapes(tmp_path: Path) -> None:
    world = ActionSenseWorld(seed=0, data_dir=_write_cache(tmp_path))
    sample = world.sample(batch_size=16, seed=42)
    assert sample.audio.shape == (16, 128)
    assert sample.vision.shape == (16, 16, 16)
    assert sample.tactile.shape == (16, 32)
    assert sample.gravity.shape == (16, 3)
    assert sample.force.shape == (16, 6)
    assert sample.label.shape == (16,)
    assert sample.label.dtype == torch.long


def test_actionsense_rejects_misaligned(tmp_path: Path) -> None:
    _write_cache(tmp_path, n=256, vision_rows=120)
    with pytest.raises(ValueError, match=r"vision=120"):
        ActionSenseWorld(seed=0, data_dir=tmp_path)


def test_build_world_actionsense_missing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BOUBA_SENS_ACTIONSENSE_DATA", str(tmp_path))
    with pytest.raises(typer.BadParameter, match="actionsense cache missing"):
        _build_world("actionsense", 0)


def test_build_world_actionsense_valid_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cache(tmp_path, n=64)
    torch.save(torch.zeros(64, dtype=torch.long), tmp_path / "subject.pt")
    torch.save({"time_s": torch.zeros(64, dtype=torch.float64)}, tmp_path / "rows.pt")
    monkeypatch.setenv("BOUBA_SENS_ACTIONSENSE_DATA", str(tmp_path))
    world = _build_world("actionsense", 0)
    assert isinstance(world, ActionSenseWorld)
    assert world.sample(batch_size=8, seed=1).audio.shape == (8, 128)
