"""Base protocol + dataclass for world simulators. See spec §3.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class WorldSample:
    """One batch of coherent multi-modal observations sharing a latent z."""

    z: torch.Tensor  # (B, D_z)
    audio: torch.Tensor  # (B, T_audio)
    vision: torch.Tensor  # (B, H, W)
    tactile: torch.Tensor  # (B, N_taxels)
    gravity: torch.Tensor  # (B, 3)
    force: torch.Tensor  # (B, 6)
    label: torch.Tensor  # (B,)


class WorldSimulator(Protocol):
    """Produces a WorldSample batch — implementations in Sprint 1."""

    def sample(self, batch_size: int, seed: int) -> WorldSample: ...

    def modality_dims(self) -> dict[str, tuple[int, ...]]: ...
