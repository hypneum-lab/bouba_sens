"""Me3 (mutual-information migration post-lesion). Spec §5.2."""

from __future__ import annotations

import math

import torch
from sklearn.feature_selection import mutual_info_regression  # type: ignore[import-not-found]


def me3_mi(codes: torch.Tensor, labels: torch.Tensor) -> float:
    """Estimate MI(codes; labels) in bits via Kraskov kNN regression.

    `codes` is a scalar feature per sample (shape `(B,)` or `(B, 1)`);
    callers typically pass a modality-specific code vector flattened to
    one dimension. `labels` is the target `(B,)` long tensor.

    Returns bits = nats / ln(2). Robust below N=512 only at the
    structural level (positive vs zero) — numeric values are noisy for
    small batches (see R-sprint3-1 in the plan).
    """
    x = codes.detach().cpu().numpy().reshape(-1, 1).astype(float)
    y = labels.detach().cpu().numpy().astype(float)
    mi_nats = mutual_info_regression(x, y, random_state=0)[0]
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
