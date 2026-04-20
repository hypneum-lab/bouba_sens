"""Task 1.5 — pairwise mutual information stays below 0.2 bit across modalities.

The orthogonal-factored projection scheme used by the 3 v0.1 worlds should
force each pair of modalities to share near-zero information once their
shared-z lineage is factored out. Estimating MI on the first feature of
each modality (scalar-by-scalar) is a pragmatic proxy: if the orthogonality
holds, pairs of projected scalars are asymptotically independent.

Spec §3.1 invariant: `test_pairwise_mi_below_0.2bit`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from sklearn.feature_selection import mutual_info_regression

from bouba_sens.world.gaussian import GaussianWorld
from bouba_sens.world.sinusoid import SinusoidWorld
from bouba_sens.world.xor import XORWorld

_MODALITIES = ("audio", "vision", "tactile", "gravity", "force")
_BATCH = 8192
_SEED = 42
_THRESHOLD_BITS = 0.2


def _first_feature(tensor: torch.Tensor) -> np.ndarray:
    """Return the first scalar feature across the batch as 1-D numpy."""
    if tensor.dim() == 3:  # vision: (B, H, W)
        flat = tensor.reshape(tensor.shape[0], -1)
        return flat[:, 0].detach().cpu().numpy()
    return tensor[:, 0].detach().cpu().numpy()


@pytest.mark.parametrize(
    "world_cls",
    [GaussianWorld, XORWorld, SinusoidWorld],
    ids=["gaussian", "xor", "sinusoid"],
)
def test_pairwise_mi_below_0_2_bit(
    world_cls: type[GaussianWorld | XORWorld | SinusoidWorld],
) -> None:
    """Every pair of modality first-features must have MI < 0.2 bit."""
    world = world_cls(seed=0)
    sample = world.sample(batch_size=_BATCH, seed=_SEED)

    features: dict[str, np.ndarray] = {m: _first_feature(getattr(sample, m)) for m in _MODALITIES}

    violations: list[tuple[str, str, float]] = []
    for i, m1 in enumerate(_MODALITIES):
        for m2 in _MODALITIES[i + 1 :]:
            # sklearn MI is in nats; /log(2) to convert to bits.
            mi_nats = mutual_info_regression(
                features[m1].reshape(-1, 1),
                features[m2],
                random_state=0,
            )[0]
            mi_bits = mi_nats / math.log(2)
            if mi_bits >= _THRESHOLD_BITS:
                violations.append((m1, m2, float(mi_bits)))

    assert not violations, (
        f"{world_cls.__name__}: pairs exceeding {_THRESHOLD_BITS} bit: {violations}"
    )
