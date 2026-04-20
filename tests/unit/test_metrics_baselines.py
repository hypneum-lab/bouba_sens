"""Task 3.5 acceptance tests for Me8 baselines."""

from __future__ import annotations

import torch
from track_p.multiplexer import GammaThetaMultiplexer

from bouba_sens.metrics.baselines import (
    freeze_nerve_plasticity,
    me8_baselines,
    random_rewire_nerve,
)
from bouba_sens.nerve import CrossModalNerve


def test_me8_returns_three_regime_values() -> None:
    out = me8_baselines(intact=0.9, robust_only=0.4, random_rewire=0.25)
    assert out == {"intact": 0.9, "robust_only": 0.4, "random_rewire": 0.25}


def test_freeze_nerve_plasticity() -> None:
    """After freezing, every nerve parameter has requires_grad=False."""
    mux = GammaThetaMultiplexer(seed=0)
    nerve = CrossModalNerve(mux, seed=0)
    # Sanity: at least one param has grad enabled before freeze.
    assert any(p.requires_grad for p in nerve.parameters())
    freeze_nerve_plasticity(nerve)
    for p in nerve.parameters():
        assert not p.requires_grad


def test_random_rewire_overwrites_and_freezes() -> None:
    mux = GammaThetaMultiplexer(seed=0)
    nerve = CrossModalNerve(mux, seed=0)
    # Snapshot one parameter before rewire.
    gate_before = nerve.gates.alpha.detach().clone()
    random_rewire_nerve(nerve, seed=42)
    # Values changed.
    assert not torch.equal(nerve.gates.alpha, gate_before)
    # All frozen.
    for p in nerve.parameters():
        assert not p.requires_grad


def test_random_rewire_reproducible() -> None:
    """Same seed -> identical rewired state."""
    mux = GammaThetaMultiplexer(seed=0)
    a = CrossModalNerve(mux, seed=0)
    b = CrossModalNerve(mux, seed=0)
    random_rewire_nerve(a, seed=7)
    random_rewire_nerve(b, seed=7)
    assert torch.equal(a.gates.alpha, b.gates.alpha)
    assert torch.equal(a.codebook.codebook, b.codebook.codebook)
