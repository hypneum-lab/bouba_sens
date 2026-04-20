"""VisionEncoder — Conv2D stack on grayscale image."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class VisionEncoder(nn.Module):
    """Maps (B, H=16, W=16) grayscale → (B, d_hidden=128)."""

    def __init__(self, h: int = 16, w: int = 16, d_hidden: int = 128) -> None:
        super().__init__()
        global_state = torch.get_rng_state()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.proj = nn.Linear(8 * h * w, d_hidden)
        torch.set_rng_state(global_state)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, H, W) → add channel → (B, 1, H, W)
        h = torch.relu(self.conv(x.unsqueeze(1)))
        return cast(Tensor, self.proj(h.flatten(1)))
