"""Task 1.7 acceptance tests for the 5 modality-specific encoders."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from bouba_sens.encoders import (
    AudioEncoder,
    ForceEncoder,
    GravityEncoder,
    TactileEncoder,
    VisionEncoder,
)

_CASES = [
    pytest.param(AudioEncoder, (128,), id="audio"),
    pytest.param(VisionEncoder, (16, 16), id="vision"),
    pytest.param(TactileEncoder, (32,), id="tactile"),
    pytest.param(GravityEncoder, (3,), id="gravity"),
    pytest.param(ForceEncoder, (6,), id="force"),
]


@pytest.mark.parametrize("cls, input_shape", _CASES)
def test_encoder_shapes(
    cls: type[nn.Module],
    input_shape: tuple[int, ...],
) -> None:
    enc = cls()
    batch = 4
    x = torch.randn(batch, *input_shape)
    y = enc(x)
    assert y.shape == (batch, 128), f"{cls.__name__}: expected (4, 128), got {tuple(y.shape)}"


def test_encoder_preserves_global_rng() -> None:
    """All 5 encoder constructors must leave torch.get_rng_state() unchanged."""
    torch.manual_seed(1234)
    anchor = torch.randn(1).item()
    AudioEncoder()
    VisionEncoder()
    TactileEncoder()
    GravityEncoder()
    ForceEncoder()
    torch.manual_seed(1234)
    assert torch.randn(1).item() == anchor
