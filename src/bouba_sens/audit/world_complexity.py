"""World-complexity audit (Sprint 7 / Task 7.5).

Six descriptive metrics computed on a `WorldSample` batch, used to
quantify how far apart the 3 synthetic worlds actually are in
complexity space before extrapolating B-3 PASS to biological
settings. All metrics are deterministic under a fixed seed; none of
them change the pre-registered B-1/B-2/B-3 thresholds.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from bouba_sens.sensory import MODALITIES, Modality

if TYPE_CHECKING:
    from bouba_sens.world.base import WorldSample


def _flatten(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().reshape(tensor.shape[0], -1)


def intrinsic_dim_pca(tensor: torch.Tensor, variance_cutoff: float = 0.95) -> int:
    """Effective rank = number of PCA components for `variance_cutoff`.

    Returns 1 when the tensor has a single feature dim so callers can
    treat e.g. `gravity` (B, 3) and `force` (B, 6) uniformly.
    """
    x = _flatten(tensor)
    if x.shape[1] < 2:
        return 1
    x_centered = x - x.mean(axis=0, keepdims=True)
    _, singular, _ = np.linalg.svd(x_centered, full_matrices=False)
    variances = (singular**2) / max(x.shape[0] - 1, 1)
    total = variances.sum()
    if total <= 0:
        return 1
    cumulative = np.cumsum(variances) / total
    return int(np.searchsorted(cumulative, variance_cutoff) + 1)


def mi_pairwise(sample: WorldSample, n_neighbours: int = 3) -> float:
    """Mean Kraskov MI(modality_i, modality_j) over the 10 ordered pairs.

    Each modality is reduced to its `flatten(1).mean(-1)` 1-D summary
    first so the sklearn estimator stays tractable on GrosMac.
    """
    summaries: dict[Modality, np.ndarray] = {}
    for modality in MODALITIES:
        tensor = getattr(sample, modality)
        summary = tensor.flatten(1).mean(dim=-1).detach().cpu().numpy()
        summaries[modality] = summary.astype(float)
    mi_values: list[float] = []
    for i, mod_i in enumerate(MODALITIES):
        for mod_j in MODALITIES[i + 1 :]:
            x = summaries[mod_i].reshape(-1, 1)
            y = summaries[mod_j]
            mi_nats = mutual_info_regression(x, y, n_neighbors=n_neighbours, random_state=0)
            mi_values.append(float(mi_nats[0] / math.log(2)))
    return float(np.mean(mi_values)) if mi_values else 0.0


def label_conditional_entropy(sample: WorldSample, n_bins: int = 16) -> float:
    """Mean H(label | modality) over the 5 modalities, in bits.

    Each modality summary is discretised into `n_bins` equal-width
    bins; the conditional entropy is computed by counting joint
    occurrences in a bins x classes contingency table.
    """
    labels = sample.label.detach().cpu().numpy().astype(int)
    n = labels.shape[0]
    if n == 0:
        return 0.0
    label_vals = np.unique(labels)
    entropies: list[float] = []
    for modality in MODALITIES:
        tensor = getattr(sample, modality)
        summary = tensor.flatten(1).mean(dim=-1).detach().cpu().numpy()
        bins = np.linspace(summary.min(), summary.max() + 1e-9, n_bins + 1)
        bin_ids = np.clip(np.digitize(summary, bins) - 1, 0, n_bins - 1)
        h_total = 0.0
        for b in range(n_bins):
            mask = bin_ids == b
            count = int(mask.sum())
            if count == 0:
                continue
            p_bin = count / n
            sub_labels = labels[mask]
            h_cond = 0.0
            for y in label_vals:
                p_yx = float(np.mean(sub_labels == y))
                if p_yx > 0:
                    h_cond -= p_yx * math.log2(p_yx)
            h_total += p_bin * h_cond
        entropies.append(h_total)
    return float(np.mean(entropies))


def linear_separability(sample: WorldSample, cv_folds: int = 5) -> float:
    """Cross-validated logistic-regression accuracy on all 5 modalities."""
    x_parts = [_flatten(getattr(sample, m)) for m in MODALITIES]
    x = np.concatenate(x_parts, axis=1)
    y = sample.label.detach().cpu().numpy().astype(int)
    if len(np.unique(y)) < 2:
        return float(np.max(np.bincount(y)) / y.shape[0])
    effective_folds = min(cv_folds, int(np.min(np.bincount(y))))
    if effective_folds < 2:
        return 0.0
    clf = LogisticRegression(max_iter=200)
    scores = cross_val_score(clf, x, y, cv=effective_folds, scoring="accuracy")
    return float(np.mean(scores))


def support_compactness(tensor: torch.Tensor, variance_cutoff: float = 0.95) -> float:
    """Fraction of the PCA mass captured before the 95% cutoff.

    1.0 means all variance is in the first few principal components
    (compact support); values closer to 0 indicate spread over many
    dimensions.
    """
    x = _flatten(tensor)
    if x.shape[1] < 2:
        return 1.0
    x_centered = x - x.mean(axis=0, keepdims=True)
    _, singular, _ = np.linalg.svd(x_centered, full_matrices=False)
    variances = (singular**2) / max(x.shape[0] - 1, 1)
    total = variances.sum()
    if total <= 0:
        return 1.0
    d_95 = intrinsic_dim_pca(tensor, variance_cutoff=variance_cutoff)
    return float(d_95 / x.shape[1])


def temporal_autocorr(sample: WorldSample) -> float:
    """Lag-1 Pearson autocorrelation on the label sequence.

    Zero for i.i.d. synthetic worlds; non-zero when the sampler has
    an implicit order (future Studyforrest-like datasets).
    """
    y = sample.label.detach().cpu().numpy().astype(float)
    if y.shape[0] < 2 or y.std() == 0:
        return 0.0
    y1 = y[:-1]
    y2 = y[1:]
    covariance = np.mean((y1 - y1.mean()) * (y2 - y2.mean()))
    normalisation = y1.std() * y2.std()
    if normalisation == 0:
        return 0.0
    return float(covariance / normalisation)


def compute_world_profile(sample: WorldSample) -> dict[str, float]:
    """Compute all 6 metrics in one call; returns a flat dict."""
    intrinsic_dims = {
        f"intrinsic_dim_pca_{m}": float(intrinsic_dim_pca(getattr(sample, m))) for m in MODALITIES
    }
    compactness = {
        f"support_compactness_{m}": support_compactness(getattr(sample, m)) for m in MODALITIES
    }
    return {
        **intrinsic_dims,
        **compactness,
        "mi_pairwise": mi_pairwise(sample),
        "label_conditional_entropy": label_conditional_entropy(sample),
        "linear_separability": linear_separability(sample),
        "temporal_autocorr": temporal_autocorr(sample),
    }
