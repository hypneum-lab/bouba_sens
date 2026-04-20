"""Task 1.1 contract tests for `bouba_sens.world.base`.

Pins the public interface of `WorldSample` (frozen dataclass) and
`WorldSimulator` (runtime-checkable Protocol) per spec §3.1 and
Sprint 1 plan Task 1.1 acceptance criteria.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from bouba_sens.world import WorldSample, WorldSimulator


def test_worldsample_is_frozen_dataclass() -> None:
    """WorldSample must be immutable once constructed."""
    sample = WorldSample(
        z=torch.zeros(2, 32),
        audio=torch.zeros(2, 128),
        vision=torch.zeros(2, 16, 16),
        tactile=torch.zeros(2, 32),
        gravity=torch.zeros(2, 3),
        force=torch.zeros(2, 6),
        label=torch.zeros(2, dtype=torch.long),
    )
    with pytest.raises(FrozenInstanceError):
        sample.label = torch.ones(2, dtype=torch.long)  # type: ignore[misc]


def test_worldsimulator_is_runtime_checkable() -> None:
    """A duck-typed class providing `sample` and `modality_dims` must
    satisfy `isinstance(..., WorldSimulator)` — decorator `@runtime_checkable`
    is required on the Protocol."""

    class _DummyWorld:
        def sample(self, batch_size: int, seed: int) -> WorldSample:
            return WorldSample(
                z=torch.zeros(batch_size, 1),
                audio=torch.zeros(batch_size, 1),
                vision=torch.zeros(batch_size, 1, 1),
                tactile=torch.zeros(batch_size, 1),
                gravity=torch.zeros(batch_size, 3),
                force=torch.zeros(batch_size, 6),
                label=torch.zeros(batch_size, dtype=torch.long),
            )

        def modality_dims(self) -> dict[str, tuple[int, ...]]:
            return {}

    assert isinstance(_DummyWorld(), WorldSimulator)
