"""Base protocol + dataclass for world simulators. See spec §3.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class WorldSimulator(Protocol):
    """Produces a WorldSample batch — implementations in Sprint 1.

    Decorated with `@runtime_checkable` so consumers can assert
    `isinstance(obj, WorldSimulator)` for duck-typed validators
    (e.g. a world registry or CLI dispatcher).
    """

    def sample(self, batch_size: int, seed: int) -> WorldSample: ...

    def modality_dims(self) -> dict[str, tuple[int, ...]]: ...


# ADR-0027 — fixed probe used to fingerprint a world's content.
WORLD_FINGERPRINT_BATCH = 256
WORLD_FINGERPRINT_SEED = 20_261_030


def world_sha256(
    world: WorldSimulator,
    *,
    batch_size: int = WORLD_FINGERPRINT_BATCH,
    seed: int = WORLD_FINGERPRINT_SEED,
) -> str:
    """SHA-256 of one fixed batch ``world.sample(batch_size, seed)``.

    ADR-0027 — two worlds with the same fingerprint serve the same items
    for this probe; two worlds built with different projections (Gaussian,
    XOR, Sinusoid) or different mock streams differ. Hashes every field of
    the sample with its dtype and shape. Uses only the world's own seeded
    generator, never the global RNG.
    """
    import hashlib

    sample = world.sample(batch_size=batch_size, seed=seed)
    h = hashlib.sha256()
    for name in ("z", "audio", "vision", "tactile", "gravity", "force", "label"):
        t = getattr(sample, name).detach().contiguous().cpu()
        h.update(repr((name, str(t.dtype), tuple(t.shape))).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()
