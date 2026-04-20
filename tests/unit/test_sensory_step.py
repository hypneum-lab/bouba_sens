"""Task 1.8 acceptance tests for `SensoryWML.step()`."""

from __future__ import annotations

import torch
from track_p.multiplexer import GammaThetaMultiplexer  # type: ignore[import-not-found]

from bouba_sens.encoders import AudioEncoder
from bouba_sens.sensory import SensoryWML


def _make_audio_sensory() -> tuple[SensoryWML, GammaThetaMultiplexer]:
    mux = GammaThetaMultiplexer(seed=0)
    enc = AudioEncoder()
    wml = SensoryWML(id=0, modality="audio", input_proj=enc, mux=mux, seed=0)
    return wml, mux


def test_step_returns_carrier_shape() -> None:
    """Output must be (B, T) with T = mux._t_grid.numel()."""
    wml, mux = _make_audio_sensory()
    x = torch.randn(4, 128)
    carrier = wml.step(x)
    assert carrier.shape == (4, mux._t_grid.numel())
    assert carrier.dtype == torch.float32


def test_step_end_to_end_differentiable() -> None:
    """Gradient must reach BOTH the shared mux constellation AND the
    modality-specific input_proj weights through the bridge term."""
    wml, mux = _make_audio_sensory()
    x = torch.randn(2, 128)
    loss = wml.step(x).pow(2).mean()
    loss.backward()

    assert mux.constellation.grad is not None
    assert mux.constellation.grad.abs().sum().item() > 0.0, (
        "constellation received no gradient — hard carrier path is broken"
    )

    first_layer_grad_sum = sum(
        p.grad.abs().sum().item() for p in wml.input_proj.parameters() if p.grad is not None
    )
    assert first_layer_grad_sum > 0.0, (
        "input_proj weights received no gradient — softmax bridge is broken"
    )
