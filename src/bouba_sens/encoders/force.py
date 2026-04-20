"""ForceEncoder — MLP on 6-wrench (3 F + 3 τ)."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class ForceEncoder(nn.Module):
    """Maps (B, 6) wrench → (B, d_hidden=128)."""

    def __init__(self, d_hidden: int = 128) -> None:
        super().__init__()
        global_state = torch.get_rng_state()
        self.fc1 = nn.Linear(6, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        torch.set_rng_state(global_state)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.fc2(torch.relu(self.fc1(x))))
