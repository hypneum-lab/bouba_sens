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

from typing import cast

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


class AdaptiveCodebook(nn.Module):
    """Soft-assignment projection from code distributions to hidden vectors (P2).

    Wraps a learnable `(alphabet_size, d_hidden)` parameter. Forward takes
    a distribution `[B, K, A]` (already softmaxed — typically from
    `mux.demodulate(hard=False)`) and matmuls to `[B, K, d_hidden]`.

    The optional `tau` kwarg re-temperatures the input distribution
    before projection: `tau < 1.0` sharpens (entropy down), `tau > 1.0`
    smooths. `tau = 1.0` (default) is a no-op. `AdaptationLoop` anneals
    tau from 1.0 to 0.1 during training per spec §4.3.

    Init can optionally seed the codebook from an existing constellation
    (e.g. `mux.constellation`, shape `(A, 2)`) by projecting through a
    random `(2, d_hidden)` map — the angular structure of the PSK
    alphabet is preserved while leaving the codebook free to adapt.
    """

    def __init__(
        self,
        alphabet_size: int = 64,
        d_hidden: int = 128,
        *,
        init_from: Tensor | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        global_state = torch.get_rng_state()
        if seed is not None:
            torch.manual_seed(seed)
        if init_from is not None:
            proj = torch.randn(init_from.shape[-1], d_hidden) * 0.1
            init = (init_from @ proj).detach()
        else:
            init = torch.randn(alphabet_size, d_hidden) * 0.1
        torch.set_rng_state(global_state)
        self.codebook = nn.Parameter(init)

    def forward(self, soft_codes: Tensor, *, tau: float = 1.0) -> Tensor:
        if tau != 1.0:
            logits = soft_codes.clamp(min=1e-9).log() / tau
            soft_codes = logits.softmax(dim=-1)
        return soft_codes @ self.codebook


class CrossModalTransducer(nn.Module):
    """Directed-edge MLP that translates source-modality hidden states
    into target-modality hidden states (P3).

    A 2-hidden-layer MLP `(d_hidden -> d_hidden -> d_hidden)` with
    0/1 gating controlled by the caller at each forward call. When
    `active=False`, forward returns the input unchanged (identity fallback).

    Activation policy per spec §4.3: `active = (gate[source] < 0.1 AND
    gate[target] > 0.3)`. The activation decision is external — this
    class only executes the MLP when instructed.

    `(source, target)` is kept as instance metadata for logging and
    MigrationReport slicing.
    """

    def __init__(
        self,
        source: Modality,
        target: Modality,
        d_hidden: int = 128,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.source = source
        self.target = target
        global_state = torch.get_rng_state()
        if seed is not None:
            torch.manual_seed(seed)
        self.fc1 = nn.Linear(d_hidden, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        torch.set_rng_state(global_state)

    def forward(self, x: Tensor, *, active: bool = True) -> Tensor:
        if not active:
            return x
        return cast(Tensor, self.fc2(torch.relu(self.fc1(x))))
