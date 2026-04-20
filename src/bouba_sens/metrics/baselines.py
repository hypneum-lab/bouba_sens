"""Me8 (intact / robust-only / random-rewire baselines). Spec §5.2 and §4.4."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from bouba_sens.nerve import CrossModalNerve

BaselineRegime = Literal["intact", "robust", "random"]


def me8_baselines(intact: float, robust_only: float, random_rewire: float) -> dict[str, float]:
    """Pack the three baseline accuracy values into the §5.2 triplet."""
    return {
        "intact": intact,
        "robust_only": robust_only,
        "random_rewire": random_rewire,
    }


def freeze_nerve_plasticity(nerve: CrossModalNerve) -> None:
    """Freeze every parameter inside the CrossModalNerve (P1+P2+P3).

    Used for the `robust` baseline: lesion runs but no plasticity updates
    happen, exposing the "no adaptation" performance floor per spec §4.4.
    """
    for p in nerve.parameters():
        p.requires_grad_(False)


def random_rewire_nerve(nerve: CrossModalNerve, *, seed: int = 0) -> None:
    """Overwrite every nerve parameter with small random noise, then freeze.

    Used for the `random` baseline: chance-level control that rules out
    "any plasticity helps" as the explanation for a robust vs intact gap.
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in nerve.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)
    freeze_nerve_plasticity(nerve)
