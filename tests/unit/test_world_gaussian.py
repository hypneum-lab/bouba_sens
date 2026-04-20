"""Task 1.2 acceptance tests for `bouba_sens.world.gaussian.GaussianWorld`."""

from __future__ import annotations

import torch

from bouba_sens.world import WorldSimulator
from bouba_sens.world.gaussian import GaussianWorld


def test_gaussian_world_implements_protocol() -> None:
    assert isinstance(GaussianWorld(seed=0), WorldSimulator)


def test_sample_shape() -> None:
    world = GaussianWorld(seed=0)
    sample = world.sample(batch_size=8, seed=42)
    assert sample.z.shape == (8, 32)
    assert sample.audio.shape == (8, 128)
    assert sample.vision.shape == (8, 16, 16)
    assert sample.tactile.shape == (8, 32)
    assert sample.gravity.shape == (8, 3)
    assert sample.force.shape == (8, 6)
    assert sample.label.shape == (8,)
    assert sample.label.dtype == torch.long


def test_same_seed_reproducible() -> None:
    world = GaussianWorld(seed=0)
    a = world.sample(batch_size=4, seed=42)
    b = world.sample(batch_size=4, seed=42)
    assert torch.equal(a.z, b.z)
    assert torch.equal(a.audio, b.audio)
    assert torch.equal(a.label, b.label)


def test_different_seeds_diverge() -> None:
    world = GaussianWorld(seed=0)
    a = world.sample(batch_size=16, seed=0)
    b = world.sample(batch_size=16, seed=1)
    assert not torch.equal(a.z, b.z)


def test_label_is_deterministic_from_z() -> None:
    """label = (z[:,0]>0).long() + 2*(z[:,1]>0).long()."""
    world = GaussianWorld(seed=0)
    sample = world.sample(batch_size=32, seed=123)
    expected = (sample.z[:, 0] > 0).long() + 2 * (sample.z[:, 1] > 0).long()
    assert torch.equal(sample.label, expected)


def test_modality_dims_returns_exact_shapes() -> None:
    world = GaussianWorld(seed=0)
    assert world.modality_dims() == {
        "audio": (128,),
        "vision": (16, 16),
        "tactile": (32,),
        "gravity": (3,),
        "force": (6,),
    }


def test_constructor_does_not_pollute_global_rng() -> None:
    torch.manual_seed(1234)
    anchor = torch.randn(1).item()
    GaussianWorld(seed=7)
    GaussianWorld(seed=8)
    torch.manual_seed(1234)
    assert torch.randn(1).item() == anchor
