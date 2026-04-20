"""SinusoidWorld — circular-latent 5-modality world with quantised label.

Sprint 1 Task 1.4. Third v0.1 WorldSimulator: the first two latent
dimensions lie on the unit circle via (sin(phi), cos(phi)) for phi
uniformly sampled in [0, 2pi). The remaining 30 dims are Gaussian noise.
Label is a 4-bin quantisation of z[:, 0] in [-1, 1].
"""

from __future__ import annotations

from typing import cast

import torch

from bouba_sens.world.base import WorldSample


class SinusoidWorld:
    """Circular-latent world: (sin(phi), cos(phi), N(0,I)_30) -> 5 modalities.

    4-class label via 4-bin quantisation of z[:, 0] in [-1, 1]. Tests the
    continuous-low-dim-manifold regime complementary to GaussianWorld's
    isotropic latent and XORWorld's discrete latent.
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
        phi = 2.0 * torch.pi * torch.rand(batch_size, generator=gen)  # (B,)
        circle = torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1)
        noise = torch.randn(batch_size, self.D_Z - 2, generator=gen)
        z = torch.cat([circle, noise], dim=-1).to(torch.float32)

        audio = z @ self._proj_audio
        vision_flat = z @ self._proj_vision
        vision = vision_flat.reshape(batch_size, 16, 16)
        tactile = z @ self._proj_tactile
        gravity = z @ self._proj_gravity
        force = z @ self._proj_force

        # 4-bin quantisation of z[:, 0] in [-1, 1] -> classes {0, 1, 2, 3}.
        bin_idx = ((z[:, 0] + 1.0) * 0.5 * 4.0).floor().clamp(0, 3)
        label = bin_idx.long()

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
