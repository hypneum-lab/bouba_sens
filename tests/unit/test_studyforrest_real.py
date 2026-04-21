"""Task 9.3 tests — StudyforrestRealWorld 5-modal shape + alignment."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bouba_sens.world.studyforrest import StudyforrestRealWorld


@pytest.fixture
def fake_5modal_dir(tmp_path: Path) -> Path:
    """Synthesise a tiny 5-modal cache that matches the real extraction contract."""
    n = 512
    torch.save(torch.randn(n, 128), tmp_path / "audio.pt")
    torch.save(torch.randn(n, 16, 16), tmp_path / "vision.pt")
    torch.save(torch.randn(n, 32), tmp_path / "tactile.pt")
    torch.save(torch.randn(n, 3), tmp_path / "gravity.pt")
    torch.save(torch.randn(n, 6), tmp_path / "force.pt")
    torch.save(torch.randint(0, 4, (n,), dtype=torch.long), tmp_path / "labels.pt")
    return tmp_path


def test_real_sample_shapes(fake_5modal_dir: Path) -> None:
    world = StudyforrestRealWorld(seed=0, data_dir=fake_5modal_dir)
    sample = world.sample(batch_size=16, seed=42)
    assert world.mode == "real5"
    assert sample.audio.shape == (16, 128)
    assert sample.vision.shape == (16, 16, 16)
    assert sample.tactile.shape == (16, 32)
    assert sample.gravity.shape == (16, 3)
    assert sample.force.shape == (16, 6)
    assert sample.label.shape == (16,)
    assert sample.label.dtype == torch.long


def test_real_no_modality_is_zero(fake_5modal_dir: Path) -> None:
    """Unlike the stub, the 5-modal variant must emit non-zero tensors
    for tactile / gravity / force."""
    world = StudyforrestRealWorld(seed=0, data_dir=fake_5modal_dir)
    sample = world.sample(batch_size=32, seed=0)
    assert sample.tactile.abs().sum() > 0
    assert sample.gravity.abs().sum() > 0
    assert sample.force.abs().sum() > 0


def test_real_temporal_contiguity(fake_5modal_dir: Path) -> None:
    """Consecutive samples in the batch must correspond to consecutive
    timecodes — preserves the lag-1 autocorr signal."""
    world = StudyforrestRealWorld(seed=0, data_dir=fake_5modal_dir)
    sample = world.sample(batch_size=8, seed=123)
    cache = torch.load(fake_5modal_dir / "audio.pt")
    # Find which slice of the cache the sample came from by matching row 0.
    for start in range(cache.shape[0] - 8):
        if torch.allclose(cache[start], sample.audio[0]):
            for offset in range(8):
                assert torch.allclose(
                    cache[(start + offset) % cache.shape[0]], sample.audio[offset]
                )
            return
    raise AssertionError("audio sample did not match any contiguous cache slice")


def test_real_reproducibility(fake_5modal_dir: Path) -> None:
    a = StudyforrestRealWorld(seed=0, data_dir=fake_5modal_dir).sample(batch_size=4, seed=7)
    b = StudyforrestRealWorld(seed=0, data_dir=fake_5modal_dir).sample(batch_size=4, seed=7)
    assert torch.equal(a.audio, b.audio)
    assert torch.equal(a.gravity, b.gravity)
    assert torch.equal(a.label, b.label)


def test_real_modality_dims_contract(fake_5modal_dir: Path) -> None:
    """Same modality_dims as GaussianWorld so the SensoryWML + encoders wire unchanged."""
    world = StudyforrestRealWorld(seed=0, data_dir=fake_5modal_dir)
    dims = world.modality_dims()
    assert dims == {
        "audio": (128,),
        "vision": (16, 16),
        "tactile": (32,),
        "gravity": (3,),
        "force": (6,),
    }
