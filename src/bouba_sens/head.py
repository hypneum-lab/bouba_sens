"""IntegrationHead — task-specific decoder on fused neuroletters.

Sprint 2 MVP per spec §3.5: a minimal linear classifier over the fused
`(B, K, d_hidden)` representation produced by `CrossModalNerve.fuse`.
Pooling is mean across the K symbol slots, so the head is invariant to
which gamma-slot carries which information. Sprint 3 may swap in richer
task heads (auto-encoding z, contrastive alignment — OQ2).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class IntegrationHead(nn.Module):
    """Pool fused representation across gamma-slots and classify."""

    def __init__(self, d_hidden: int = 128, n_classes: int = 10) -> None:
        super().__init__()
        global_state = torch.get_rng_state()
        self.classifier = nn.Linear(d_hidden, n_classes)
        torch.set_rng_state(global_state)

    def forward(self, fused: Tensor) -> Tensor:
        """fused: (B, K, d_hidden) -> logits: (B, n_classes)."""
        pooled = fused.mean(dim=1)  # (B, d_hidden)
        return cast(Tensor, self.classifier(pooled))
