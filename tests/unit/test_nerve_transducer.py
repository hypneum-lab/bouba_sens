"""Task 2.3 acceptance tests for `bouba_sens.nerve.CrossModalTransducer`."""

from __future__ import annotations

import torch

from bouba_sens.nerve import CrossModalTransducer


def test_transducer_identity_when_inactive() -> None:
    """active=False must return the input unchanged (bit-exact identity)."""
    t = CrossModalTransducer(source="audio", target="vision", seed=0)
    x = torch.randn(3, 128)
    y = t(x, active=False)
    assert torch.equal(x, y)


def test_transducer_changes_output_when_active() -> None:
    t = CrossModalTransducer(source="audio", target="vision", seed=0)
    x = torch.randn(3, 128)
    y = t(x, active=True)
    assert not torch.equal(x, y)
    assert y.shape == x.shape


def test_transducer_gradient_flows_when_active() -> None:
    t = CrossModalTransducer(source="tactile", target="force", seed=0)
    x = torch.randn(2, 128)
    y = t(x, active=True)
    y.pow(2).mean().backward()
    # Gradient must reach both fc1 and fc2 weights.
    assert t.fc1.weight.grad is not None
    assert t.fc2.weight.grad is not None
    assert t.fc1.weight.grad.abs().sum().item() > 0.0
    assert t.fc2.weight.grad.abs().sum().item() > 0.0


def test_transducer_pair_is_stored() -> None:
    t = CrossModalTransducer(source="gravity", target="tactile")
    assert t.source == "gravity"
    assert t.target == "tactile"


def test_transducer_preserves_global_rng() -> None:
    torch.manual_seed(1234)
    anchor = torch.randn(1).item()
    CrossModalTransducer(source="audio", target="vision", seed=7)
    CrossModalTransducer(source="vision", target="audio", seed=8)
    torch.manual_seed(1234)
    assert torch.randn(1).item() == anchor
