"""Me3 (mutual-information migration post-lesion). Spec §5.2."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_regression

if TYPE_CHECKING:
    from numpy.typing import NDArray


def me3_mi(codes: torch.Tensor, labels: torch.Tensor) -> float:
    """Estimate MI(codes; labels) in bits via Kraskov kNN.

    `codes` may be a scalar feature per sample (shape `(B,)` / `(B, 1)`)
    OR the full multi-D fused representation (shape `(B, d)`, e.g.
    `(B, 896)`). `labels` is the target `(B,)` long tensor.

    1-D codes route through ``sklearn.mutual_info_regression`` (the
    legacy Sprint 5..17 path, preserved bit-for-bit). Multi-D codes
    route through the Kraskov-Stoegbauer-Grassberger (KSG) continuous
    estimator (``nerve_wml.methodology.mi_kraskov_ksg_continuous``),
    which respects the joint dimensionality instead of collapsing it
    to a single column — the bug fixed in ADR-0019 lockstep with the
    full sonde (paper §6.3.1).

    Returns bits = nats / ln(2). Robust below N=512 only at the
    structural level (positive vs zero) — numeric values are noisy for
    small batches (see R-sprint3-1 in the plan).
    """
    arr = codes.detach().cpu().numpy().astype(float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    y = labels.detach().cpu().numpy().astype(float)
    if arr.shape[1] == 1:
        # Legacy 1-D path — unchanged numerics.
        mi_nats = mutual_info_regression(arr, y, random_state=0)[0]
        return float(mi_nats / math.log(2))
    # Multi-D path — proper joint KSG estimator. Helper returns nats.
    from nerve_wml.methodology import mi_kraskov_ksg_continuous

    mi_nats = mi_kraskov_ksg_continuous(arr, y.reshape(-1, 1))
    return float(mi_nats / math.log(2))


def me3_delta(
    codes_pre: torch.Tensor,
    codes_post: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Me3 invariant B-2: `mi_post - mi_pre` for the surviving channel.

    Spec §1.2 B-2 threshold: a positive delta > 0.10 bit signals
    informational compensation on the surviving modality after lesion.
    """
    return me3_mi(codes_post, labels) - me3_mi(codes_pre, labels)


def _to_numpy_2d(
    a: torch.Tensor | NDArray[np.floating] | NDArray[np.integer],
) -> NDArray[np.floating]:
    """Convert tensor/ndarray to ``(N, d)`` float64 numpy array."""
    arr = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    arr = arr.astype(np.float64, copy=False)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def me3_mi_mine(
    codes: torch.Tensor | NDArray[np.floating],
    labels: torch.Tensor | NDArray[np.integer],
    *,
    n_epochs: int = 500,
    batch_size: int = 256,
    tail_average: int = 50,
) -> float:
    """Estimate MI(codes; labels) in bits via MINE (Donsker-Varadhan).

    Dispatches to ``nerve_wml.methodology.mi_mine``. Complements the
    Kraskov kNN estimator with a variational neural estimator that
    can catch signal invisible to density-based methods on 1-D
    mean-pooled codes (paper §6.3).

    Returns bits = nats / ln(2). Requires ``N >= batch_size`` (MINE is
    unstable with fewer samples).
    """
    # Local import: nerve-wml is an optional heavy dep; keep module
    # import-cheap for callers that only use me3_delta.
    from nerve_wml.methodology import mi_mine

    x = _to_numpy_2d(codes)
    y = _to_numpy_2d(labels)
    mi_nats = mi_mine(
        x,
        y,
        n_epochs=n_epochs,
        batch_size=batch_size,
        tail_average=tail_average,
    )
    return float(mi_nats / np.log(2))


def me3_delta_mine(
    codes_pre: torch.Tensor | NDArray[np.floating],
    codes_post: torch.Tensor | NDArray[np.floating],
    labels: torch.Tensor | NDArray[np.integer],
    *,
    n_epochs: int = 500,
    batch_size: int = 256,
    tail_average: int = 50,
) -> float:
    """Me3 delta computed with MINE rather than Kraskov kNN.

    Mirrors :func:`me3_delta` but routes through MINE. Sprint 17 probe
    of whether B-2 = 0 at LOCK_AFTER=100 reflects the network (no
    migration) or Kraskov blindness to 1-D mean-pooled codes.
    """
    mi_post = me3_mi_mine(
        codes_post,
        labels,
        n_epochs=n_epochs,
        batch_size=batch_size,
        tail_average=tail_average,
    )
    mi_pre = me3_mi_mine(
        codes_pre,
        labels,
        n_epochs=n_epochs,
        batch_size=batch_size,
        tail_average=tail_average,
    )
    return mi_post - mi_pre
