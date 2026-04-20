"""Me3 (mutual-information migration post-lesion). Spec §5.2."""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

_ArrayLike = np.ndarray | torch.Tensor


def _to_numpy(arr: _ArrayLike) -> np.ndarray:
    """Convert Tensor or ndarray to a float64 numpy array."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy().astype(float)
    return np.asarray(arr, dtype=float)


def me3_mi(codes: _ArrayLike, labels: _ArrayLike) -> float:
    """Estimate MI(codes; labels) in bits via Kraskov kNN regression.

    `codes` is a scalar feature per sample (shape `(B,)` or `(B, 1)`),
    or a multi-dim array `(B, D)` with D > 1 (uses per-feature kNN MI sum).
    `labels` is the target `(B,)` array or tensor.

    Returns bits = nats / ln(2). Robust below N=512 only at the
    structural level (positive vs zero) — numeric values are noisy for
    small batches (see R-sprint3-1 in the plan).
    """
    codes_np = _to_numpy(codes)
    labels_np = _to_numpy(labels).astype(float)

    if codes_np.ndim == 1:
        codes_np = codes_np.reshape(-1, 1)
    else:
        codes_np = codes_np.reshape(len(codes_np), -1)

    if codes_np.shape[1] == 1:
        mi_nats = mutual_info_regression(codes_np, labels_np, random_state=0)[0]
    else:
        # Multi-dim: sum per-feature kNN MI estimates (discrete-label variant)
        labels_int = labels_np.round().astype(int)
        mi_nats = float(mutual_info_classif(codes_np, labels_int, random_state=0).sum())
    return float(mi_nats / math.log(2))


def me3_delta(
    codes_pre: _ArrayLike,
    codes_post: _ArrayLike,
    labels: _ArrayLike,
) -> float:
    """Me3 invariant B-2: `mi_post - mi_pre` for the surviving channel.

    Spec §1.2 B-2 threshold: a positive delta > 0.10 bit signals
    informational compensation on the surviving modality after lesion.
    Accepts torch.Tensor or np.ndarray inputs.
    """
    return me3_mi(codes_post, labels) - me3_mi(codes_pre, labels)


# --- Binning estimator ---------------------------------------------------


def _mi_binning(codes: np.ndarray, labels: np.ndarray, *, bins_per_dim: int) -> float:
    """Discrete plug-in MI estimate on binned codes vs integer labels.

    For 1-D input (B, 1), uses standard quantile-binning histogram.
    For multi-dim input (B, D) with D > 1, falls back to a parametric
    Gaussian class-conditional approximation which is equivalent to
    binning the class-posterior space; `bins_per_dim` governs the 1-D
    granularity when D == 1.
    """
    n_classes = int(labels.max()) + 1
    n = len(labels)
    codes_2d = codes.reshape(n, -1)
    d = codes_2d.shape[1]

    if d == 1:
        # 1-D exact quantile-binning MI
        col = codes_2d[:, 0]
        edges = np.quantile(col, np.linspace(0, 1, bins_per_dim + 1)[1:-1])
        quantiled = np.digitize(col, edges)
        joint, _, _ = np.histogram2d(quantiled, labels, bins=[bins_per_dim, n_classes])
        joint = joint / joint.sum()
        px = joint.sum(axis=1, keepdims=True)
        py = joint.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log(joint / (px * py))
        log_ratio[~np.isfinite(log_ratio)] = 0
        return float((joint * log_ratio).sum())

    # Multi-dim: parametric Gaussian class-conditional MI approximation
    # MI = H(Y) - H(Y|X) where H(Y|X) is estimated via naive-Bayes Gaussian
    # classifier (unit covariance, class-conditional means from data).
    prior = np.bincount(labels, minlength=n_classes) / n
    class_means = np.stack([codes_2d[labels == k].mean(axis=0) for k in range(n_classes)])
    log_likes = np.zeros((n, n_classes))
    for k in range(n_classes):
        diff = codes_2d - class_means[k]
        log_likes[:, k] = np.log(prior[k]) - 0.5 * (diff * diff).sum(axis=1)
    max_ll = log_likes.max(axis=1, keepdims=True)
    log_z = max_ll[:, 0] + np.log(np.exp(log_likes - max_ll).sum(axis=1))
    log_post = log_likes - log_z[:, None]
    label_log_posts = log_post[np.arange(n), labels]
    h_y_given_x = -label_log_posts.mean()
    h_y = float(-np.sum(prior[prior > 0] * np.log(prior[prior > 0])))
    return max(0.0, float(h_y - h_y_given_x))


def me3_delta_binning(
    pre_codes: np.ndarray,
    post_codes: np.ndarray,
    labels: np.ndarray,
    *,
    bins_per_dim: int = 16,
) -> float:
    return _mi_binning(post_codes, labels, bins_per_dim=bins_per_dim) - _mi_binning(
        pre_codes, labels, bins_per_dim=bins_per_dim
    )


# --- MINE neural estimator ----------------------------------------------


def _mi_mine(codes: np.ndarray, labels: np.ndarray, *, epochs: int, seed: int) -> float:
    from torch import nn

    torch.manual_seed(seed)
    n_classes = int(labels.max() + 1)
    labels_onehot = np.eye(n_classes)[labels].astype(np.float32)
    x = torch.from_numpy(codes.astype(np.float32))
    y = torch.from_numpy(labels_onehot)
    critic = nn.Sequential(
        nn.Linear(codes.shape[1] + n_classes, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(x.shape[0])
        y_shuf = y[perm]
        joint = critic(torch.cat([x, y], dim=1))
        margin = critic(torch.cat([x, y_shuf], dim=1))
        loss = -(joint.mean() - torch.log(torch.exp(margin).mean() + 1e-8))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        perm = torch.randperm(x.shape[0])
        y_shuf = y[perm]
        joint = critic(torch.cat([x, y], dim=1))
        margin = critic(torch.cat([x, y_shuf], dim=1))
        mi = float(joint.mean() - torch.log(torch.exp(margin).mean() + 1e-8))
    return mi


def me3_delta_mine(
    pre_codes: np.ndarray,
    post_codes: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 500,
    seed: int = 0,
) -> float:
    return _mi_mine(post_codes, labels, epochs=epochs, seed=seed) - _mi_mine(
        pre_codes, labels, epochs=epochs, seed=seed
    )
