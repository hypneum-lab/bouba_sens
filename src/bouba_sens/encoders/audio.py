"""AudioEncoder — Conv1D stack on 1-D spectrogram."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class AudioEncoder(nn.Module):
    """Maps (B, T_audio=128) spectrogram → (B, d_hidden=128)."""

    def __init__(self, t_audio: int = 128, d_hidden: int = 128) -> None:
        super().__init__()
        global_state = torch.get_rng_state()
        self.conv = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.proj = nn.Linear(16 * t_audio, d_hidden)
        torch.set_rng_state(global_state)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T_audio) → add channel → (B, 1, T_audio)
        h = torch.relu(self.conv(x.unsqueeze(1)))
        return cast(Tensor, self.proj(h.flatten(1)))
