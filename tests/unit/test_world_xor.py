"""Task 1.3 acceptance tests for `bouba_sens.world.xor.XORWorld`."""

from __future__ import annotations

import torch

from bouba_sens.world import WorldSimulator
from bouba_sens.world.xor import XORWorld


def test_xor_world_implements_protocol() -> None:
    assert isinstance(XORWorld(seed=0), WorldSimulator)


def test_z_is_rademacher() -> None:
    """All latent values must be ∈ {-1, +1} (Rademacher)."""
    world = XORWorld(seed=0)
    sample = world.sample(batch_size=64, seed=42)
    unique = torch.unique(sample.z).tolist()
    assert sorted(unique) == [-1.0, 1.0]


def test_label_matches_parity() -> None:
    """label[b] = 1 iff z[b, 0] and z[b, 1] have the same sign."""
    world = XORWorld(seed=0)
    sample = world.sample(batch_size=128, seed=7)
    expected = (sample.z[:, 0] * sample.z[:, 1] > 0).long()
    assert torch.equal(sample.label, expected)


def test_label_is_binary() -> None:
    world = XORWorld(seed=0)
    sample = world.sample(batch_size=256, seed=0)
    unique = torch.unique(sample.label).tolist()
    assert set(unique).issubset({0, 1})


def test_sample_shape() -> None:
    world = XORWorld(seed=0)
    sample = world.sample(batch_size=4, seed=42)
    assert sample.z.shape == (4, 32)
    assert sample.audio.shape == (4, 128)
    assert sample.vision.shape == (4, 16, 16)
    assert sample.tactile.shape == (4, 32)
    assert sample.gravity.shape == (4, 3)
    assert sample.force.shape == (4, 6)
    assert sample.label.shape == (4,)


def test_same_seed_reproducible() -> None:
    world = XORWorld(seed=0)
    a = world.sample(batch_size=8, seed=42)
    b = world.sample(batch_size=8, seed=42)
    assert torch.equal(a.z, b.z)


def test_constructor_does_not_pollute_global_rng() -> None:
    torch.manual_seed(1234)
    anchor = torch.randn(1).item()
    XORWorld(seed=11)
    torch.manual_seed(1234)
    assert torch.randn(1).item() == anchor
