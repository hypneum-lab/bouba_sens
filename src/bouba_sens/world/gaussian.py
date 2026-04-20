"""GaussianWorld — 5 modalities from a shared z via orthogonal projections.

Sprint 1 Task 1.2 per spec §3.1. Latent z ~ N(0, I) is projected to each
modality through a *fixed* orthogonal matrix (QR-decomposed at init, frozen
thereafter) so that pairwise modality MI is low by construction — verified
by Task 1.5 `test_pairwise_mi_below_0.2bit`.
"""

from __future__ import annotations

from typing import cast

import torch

from bouba_sens.world.base import WorldSample


class GaussianWorld:
    """Gaussian latent world: z ~ N(0, I_{32}) → 5 orthogonally-projected modalities.

    Label is a 4-class code derived from the sign pattern of `z[:, 0:2]`,
    so two latent dimensions carry the class signal and the other 30 are noise.

    Seeded via local `torch.Generator` — never touches the global RNG
    (MlpWML convention, see nerve-wml issue #1).
    """

    D_Z: int = 32

    def __init__(self, *, seed: int = 0) -> None:
        self._seed = seed
        gen = torch.Generator().manual_seed(seed)
        # Orthogonal projections computed in fp64 for precision then cast to
        # fp32 (plan R-sprint1-2 — Gram-Schmidt may drift in fp32 at init).
        self._proj_audio = self._orthogonal(gen, 128)
        self._proj_vision = self._orthogonal(gen, 16 * 16)
        self._proj_tactile = self._orthogonal(gen, 32)
        self._proj_gravity = self._orthogonal(gen, 3)
        self._proj_force = self._orthogonal(gen, 6)

    def _orthogonal(self, gen: torch.Generator, d_out: int) -> torch.Tensor:
        """Return a (D_Z, d_out) orthonormal projection matrix, fp32."""
        if d_out <= self.D_Z:
            raw = torch.randn(self.D_Z, d_out, generator=gen, dtype=torch.float64)
            q, _ = torch.linalg.qr(raw, mode="reduced")
            mat = q  # columns orthonormal
        else:
            raw = torch.randn(d_out, self.D_Z, generator=gen, dtype=torch.float64)
            q, _ = torch.linalg.qr(raw, mode="reduced")
            mat = q.T  # rows orthonormal → after transpose, columns span d_out
        return cast(torch.Tensor, mat.to(torch.float32))

    def sample(self, batch_size: int, seed: int) -> WorldSample:
        gen = torch.Generator().manual_seed(seed)
        z = torch.randn(batch_size, self.D_Z, generator=gen)
        audio = z @ self._proj_audio
        vision_flat = z @ self._proj_vision
        vision = vision_flat.reshape(batch_size, 16, 16)
        tactile = z @ self._proj_tactile
        gravity = z @ self._proj_gravity
        force = z @ self._proj_force
        label = (z[:, 0] > 0).long() + 2 * (z[:, 1] > 0).long()
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
