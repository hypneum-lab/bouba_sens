"""CrossModalNerve and its plasticity mechanisms — spec §3.3.

Sprint 2 lands the plastic router in four layers:

- P1 `PlasticityGate`  : channel-wise attentional weights (Task 2.1).
- P2 `AdaptiveCodebook`: soft-assignment projection (Task 2.2).
- P3 `CrossModalTransducer`: per-pair MLP with 0/1 gating (Task 2.3).
- `CrossModalNerve`    : assembly + `fuse()` (Tasks 2.4-2.5).

All four classes live in this module so they share the Modality
vocabulary and keep the import surface tight.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from bouba_sens.sensory import MODALITIES, Modality


class PlasticityGate(nn.Module):
    """Channel-wise attentional weights over the 5 modalities (P1).

    A single learnable `alpha` vector of shape `(5,)` is passed through
    `softmax` to produce non-negative weights summing to 1. Init is all-
    zeros so the softmax is uniform (0.2 per channel) — any early
    imbalance must emerge from training or a `CrossModalNerve.on_lesion`
    update.

    Forward takes a `dict[Modality, Tensor]` (one carrier per modality,
    same leading dims) and returns the same dict with each tensor
    scaled by its modality's softmax weight.
    """

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(len(MODALITIES)))

    def weights(self) -> Tensor:
        """Return the current softmax gate weights as a `(5,)` tensor."""
        return self.alpha.softmax(dim=-1)

    def forward(self, letters: dict[Modality, Tensor]) -> dict[Modality, Tensor]:
        w = self.weights()
        return {m: letters[m] * w[i] for i, m in enumerate(MODALITIES)}
