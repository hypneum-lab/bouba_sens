"""XORWorld — Rademacher latent → 5 orthogonally-projected modalities.

Sprint 1 Task 1.3. Complement to GaussianWorld on a discrete-latent regime
(Rademacher z ∈ {-1, +1}^32) with a 2-class XOR-parity label on the first
two latent dims.
"""

from __future__ import annotations

from typing import cast

import torch

from bouba_sens.world.base import WorldSample


class XORWorld:
    """Rademacher latent z in {-1, +1} of shape (B, 32), 5 projected modalities.

    2-class parity label: `y = (z[:, 0] * z[:, 1] > 0).long()` — distinguishes
    whether the first two latent coordinates have the same sign (y=1) or
    opposite signs (y=0). This is the canonical XOR-separability benchmark.
    """

    D_Z: int = 32

    def __init__(self, *, seed: int = 0) -> None:
        self._seed = seed
        gen = torch.Generator().manual_seed(seed)
        self._proj_audio = self._orthogonal(gen, 128)
        self._proj_vision = self._orthogonal(gen, 16 * 16)
        self._proj_tactile = self._orthogonal(gen, 32)
        self._proj_gravity = self._orthogonal(gen, 3)
        self._proj_force = self._orthogonal(gen, 6)

    def _orthogonal(self, gen: torch.Generator, d_out: int) -> torch.Tensor:
        if d_out <= self.D_Z:
            raw = torch.randn(self.D_Z, d_out, generator=gen, dtype=torch.float64)
            q, _ = torch.linalg.qr(raw, mode="reduced")
            mat = q
        else:
            raw = torch.randn(d_out, self.D_Z, generator=gen, dtype=torch.float64)
            q, _ = torch.linalg.qr(raw, mode="reduced")
            mat = q.T
        return cast(torch.Tensor, mat.to(torch.float32))

    def sample(self, batch_size: int, seed: int) -> WorldSample:
        gen = torch.Generator().manual_seed(seed)
        bits = torch.randint(0, 2, (batch_size, self.D_Z), generator=gen, dtype=torch.int64)
        z = (2 * bits - 1).to(torch.float32)
        audio = z @ self._proj_audio
        vision_flat = z @ self._proj_vision
        vision = vision_flat.reshape(batch_size, 16, 16)
        tactile = z @ self._proj_tactile
        gravity = z @ self._proj_gravity
        force = z @ self._proj_force
        label = (z[:, 0] * z[:, 1] > 0).long()
        return WorldSample(
            z=z,
            audio=audio,
            vision=vision,
            tactile=tactile,
            gravity=gravity,
            force=force,
            label=label,
        )

    def modality_dims(self) -> dict[str, tuple[int, ...]]:
        return {
            "audio": (128,),
            "vision": (16, 16),
            "tactile": (32,),
            "gravity": (3,),
            "force": (6,),
        }
