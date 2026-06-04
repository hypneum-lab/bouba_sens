"""ADR-0019 §E1.3d tests — CCFilmWorld real-AV bridge shape + alignment."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bouba_sens.world.ccfilm import CCFilmWorld


@pytest.fixture
def fake_ccfilm_dir(tmp_path: Path) -> Path:
    """Tiny cache matching the real extraction contract."""
    n = 256
    torch.save(torch.randn(n, 128), tmp_path / "audio.pt")
    torch.save(torch.rand(n, 16, 16), tmp_path / "vision.pt")
    torch.save(torch.randn(n, 32), tmp_path / "tactile.pt")
    torch.save(torch.randn(n, 3), tmp_path / "gravity.pt")
    torch.save(torch.randn(n, 6), tmp_path / "force.pt")
    torch.save(torch.randint(0, 4, (n,), dtype=torch.long), tmp_path / "labels.pt")
    return tmp_path


def test_ccfilm_mode_is_real(fake_ccfilm_dir: Path) -> None:
    world = CCFilmWorld(seed=0, data_dir=fake_ccfilm_dir)
    assert world.mode == "real_av"


def test_ccfilm_sample_shapes(fake_ccfilm_dir: Path) -> None:
    world = CCFilmWorld(seed=0, data_dir=fake_ccfilm_dir)
    sample = world.sample(batch_size=16, seed=42)
    assert sample.audio.shape == (16, 128)
    assert sample.vision.shape == (16, 16, 16)
    assert sample.tactile.shape == (16, 32)
    assert sample.gravity.shape == (16, 3)
    assert sample.force.shape == (16, 6)
    assert sample.label.shape == (16,)
    assert sample.label.dtype == torch.long


def test_ccfilm_proprio_nonzero(fake_ccfilm_dir: Path) -> None:
    """tactile/gravity/force are SYNTHETIC but must still be non-zero
    (unlike the Studyforrest stub's zeroed channels)."""
    world = CCFilmWorld(seed=0, data_dir=fake_ccfilm_dir)
    sample = world.sample(batch_size=32, seed=0)
    assert sample.tactile.abs().sum() > 0
    assert sample.gravity.abs().sum() > 0
    assert sample.force.abs().sum() > 0


def test_ccfilm_temporal_contiguity(fake_ccfilm_dir: Path) -> None:
    world = CCFilmWorld(seed=0, data_dir=fake_ccfilm_dir)
    sample = world.sample(batch_size=8, seed=123)
    cache = torch.load(fake_ccfilm_dir / "audio.pt")
    for start in range(cache.shape[0] - 8):
        if torch.allclose(cache[start], sample.audio[0]):
            for offset in range(8):
                assert torch.allclose(
                    cache[(start + offset) % cache.shape[0]], sample.audio[offset]
                )
            return
    raise AssertionError("audio sample did not match any contiguous cache slice")


def test_ccfilm_reproducibility(fake_ccfilm_dir: Path) -> None:
    a = CCFilmWorld(seed=0, data_dir=fake_ccfilm_dir).sample(batch_size=4, seed=7)
    b = CCFilmWorld(seed=0, data_dir=fake_ccfilm_dir).sample(batch_size=4, seed=7)
    assert torch.equal(a.audio, b.audio)
    assert torch.equal(a.label, b.label)


def test_ccfilm_modality_dims_contract(fake_ccfilm_dir: Path) -> None:
    world = CCFilmWorld(seed=0, data_dir=fake_ccfilm_dir)
    assert world.modality_dims() == {
        "audio": (128,),
        "vision": (16, 16),
        "tactile": (32,),
        "gravity": (3,),
        "force": (6,),
    }
