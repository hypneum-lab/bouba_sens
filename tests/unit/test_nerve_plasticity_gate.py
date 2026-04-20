"""Task 2.1 acceptance tests for `bouba_sens.nerve.PlasticityGate`."""

from __future__ import annotations

import torch

from bouba_sens.nerve import PlasticityGate
from bouba_sens.sensory import MODALITIES


def _make_letters(batch: int = 2, t: int = 175) -> dict[str, torch.Tensor]:
    """Return a dict[Modality, (B, T)] of all-ones carriers."""
    return {m: torch.ones(batch, t) for m in MODALITIES}


def test_plasticity_gate_uniform_at_init() -> None:
    """At init, every modality must receive the same scale (alpha=0 -> softmax=1/5)."""
    gate = PlasticityGate()
    letters = _make_letters()
    out = gate(letters)
    scales = [out[m].mean().item() for m in MODALITIES]
    # All 5 must be equal within fp32 precision.
    assert max(scales) - min(scales) < 1e-6
    # And equal to 1/5 since input is all-ones.
    assert abs(scales[0] - 0.2) < 1e-6


def test_plasticity_gate_softmax_sums_to_one() -> None:
    gate = PlasticityGate()
    assert abs(gate.weights().sum().item() - 1.0) < 1e-6


def test_plasticity_gate_gradient_flows() -> None:
    """A loss on the gated output must produce non-zero grad on alpha."""
    gate = PlasticityGate()
    letters = _make_letters()
    # Loss weights modalities non-uniformly so softmax asymmetry matters.
    out = gate(letters)
    loss = sum((out[m] * (i + 1)).sum() for i, m in enumerate(MODALITIES))
    loss.backward()
    assert gate.alpha.grad is not None
    assert gate.alpha.grad.abs().sum().item() > 0.0


def test_plasticity_gate_preserves_dict_keys() -> None:
    gate = PlasticityGate()
    letters = _make_letters()
    out = gate(letters)
    assert set(out.keys()) == set(MODALITIES)
