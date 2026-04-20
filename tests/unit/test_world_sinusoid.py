"""Task 1.4 acceptance tests for `bouba_sens.world.sinusoid.SinusoidWorld`."""

from __future__ import annotations

import torch

from bouba_sens.world import WorldSimulator
from bouba_sens.world.sinusoid import SinusoidWorld


def test_sinusoid_world_implements_protocol() -> None:
    assert isinstance(SinusoidWorld(seed=0), WorldSimulator)


def test_z_on_unit_circle_pairs() -> None:
    """First two latent dims must satisfy z[:,0]^2 + z[:,1]^2 ~= 1."""
    world = SinusoidWorld(seed=0)
    sample = world.sample(batch_size=256, seed=42)
    radii_sq = sample.z[:, 0] ** 2 + sample.z[:, 1] ** 2
    assert torch.allclose(radii_sq, torch.ones_like(radii_sq), atol=1e-5)


def test_sample_shape() -> None:
    world = SinusoidWorld(seed=0)
    sample = world.sample(batch_size=8, seed=1)
    assert sample.z.shape == (8, 32)
    assert sample.audio.shape == (8, 128)
    assert sample.vision.shape == (8, 16, 16)
    assert sample.tactile.shape == (8, 32)
    assert sample.gravity.shape == (8, 3)
    assert sample.force.shape == (8, 6)
    assert sample.label.shape == (8,)


def test_label_is_four_class() -> None:
    """Label must fall in {0, 1, 2, 3} — 4-bin quantisation of z[:, 0]."""
    world = SinusoidWorld(seed=0)
    sample = world.sample(batch_size=512, seed=0)
    unique = set(torch.unique(sample.label).tolist())
    assert unique.issubset({0, 1, 2, 3})
    assert unique, "label tensor is empty"


def test_same_seed_reproducible() -> None:
    world = SinusoidWorld(seed=0)
    a = world.sample(batch_size=8, seed=42)
    b = world.sample(batch_size=8, seed=42)
    assert torch.equal(a.z, b.z)


def test_constructor_does_not_pollute_global_rng() -> None:
    torch.manual_seed(1234)
    anchor = torch.randn(1).item()
    SinusoidWorld(seed=3)
    torch.manual_seed(1234)
    assert torch.randn(1).item() == anchor
