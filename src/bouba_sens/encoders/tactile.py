"""TactileEncoder — MLP on taxel vector."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class TactileEncoder(nn.Module):
    """Maps (B, N_taxels=32) → (B, d_hidden=128)."""

    def __init__(self, n_taxels: int = 32, d_hidden: int = 128) -> None:
        super().__init__()
        global_state = torch.get_rng_state()
        self.fc1 = nn.Linear(n_taxels, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        torch.set_rng_state(global_state)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.fc2(torch.relu(self.fc1(x))))
