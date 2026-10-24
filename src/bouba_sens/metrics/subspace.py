"""Subspace-shift metric (pre-registration amendment v0.8, section 3).

For a fixed probe, ``Z_pre`` and ``Z_post`` are ``(n, d)`` fused codes
before and after the lesion phase. Each is column-centred; its top-``k``
right singular vectors (principal directions in feature space) span a
``k``-dimensional subspace. The shift is the normalised squared chordal
distance on the Grassmannian::

    S = (1/k) * sum_i sin^2(theta_i) = 1 - ||V_pre^T V_post||_F^2 / k

with ``theta_i`` the principal angles. ``S`` is 0 for identical subspaces
and 1 for orthogonal ones; it is invariant to the basis chosen inside
each subspace (sign flips, rotations), to a global rescaling of the
codes and to any per-column offset (centring).

Rank guard (section 3.1): a matrix passes if its column-centred
numerical rank is at least ``k``. The numerical rank is 0 when the
largest singular value of the centred matrix is exactly 0 (zero or
constant codes); otherwise it counts singular values
``>= 1e-6 * largest``. Non-finite codes are a *technical* failure and
raise ``ValueError``; a rank failure raises ``SubspaceRankError`` (an
outcome-dependent exclusion, never replaced).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

K_DEFAULT = 12
RANK_RTOL = 1e-6


class SubspaceRankError(ValueError):
    """A code matrix fails the rank check of amendment v0.8 section 3.1."""


def _as_finite_f64(z: Tensor) -> Tensor:
    """Validate shape and finiteness (technical failure) before any rank logic."""
    if z.dim() != 2:
        raise ValueError(f"codes must be a 2-D (n, d) matrix; got shape {tuple(z.shape)}")
    z64 = z.detach().to(dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(z64).all()):
        raise ValueError("codes contain non-finite values (technical failure)")
    return z64


def _centred(z: Tensor) -> Tensor:
    z64 = _as_finite_f64(z)
    if z64.shape[0] == 0:
        return z64
    # Centre relative to a reference row first: identical rows then give an
    # exactly zero matrix (plain mean-subtraction leaves ~1e-16 residue that
    # passes the relative rank threshold), and accuracy is no worse.
    d = z64 - z64[:1]
    return d - d.mean(dim=0, keepdim=True)


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")


def _rank_from_svals(s: Tensor) -> int:
    if s.numel() == 0:
        return 0
    top = float(s[0])
    if top == 0.0:
        return 0
    return int((s >= RANK_RTOL * top).sum().item())


def numerical_rank(z: Tensor) -> int:
    """Column-centred numerical rank (0 for zero or constant codes)."""
    return _rank_from_svals(torch.linalg.svdvals(_centred(z)))


def passes_rank_check(z: Tensor, k: int = K_DEFAULT) -> bool:
    """True if the centred numerical rank of ``z`` is at least ``k``."""
    _check_k(k)
    return numerical_rank(z) >= k


def principal_directions(z: Tensor, k: int = K_DEFAULT) -> Tensor:
    """Top-``k`` right singular vectors of column-centred ``z``, shape ``(d, k)``.

    Raises ``SubspaceRankError`` if the rank check fails.
    """
    _check_k(k)
    zc = _centred(z)
    _, s, vh = torch.linalg.svd(zc, full_matrices=False)
    rank = _rank_from_svals(s)
    if rank < k:
        raise SubspaceRankError(f"centred numerical rank {rank} < k={k}")
    v: Tensor = vh[:k].T.contiguous()
    return v


def subspace_shift_from_bases(v_pre: Tensor, v_post: Tensor) -> float:
    """``S = 1 - ||V_pre^T V_post||_F^2 / k`` for orthonormal ``(d, k)`` bases."""
    if v_pre.shape != v_post.shape or v_pre.dim() != 2:
        raise ValueError(f"bases must share shape (d, k); got {v_pre.shape}, {v_post.shape}")
    k = v_pre.shape[1]
    _check_k(k)
    cos = torch.linalg.svdvals(v_pre.to(torch.float64).T @ v_post.to(torch.float64))
    sin2 = 1.0 - cos.clamp(max=1.0) ** 2
    return float(sin2.mean().clamp(min=0.0, max=1.0).item())


def subspace_shift(z_pre: Tensor, z_post: Tensor, k: int = K_DEFAULT) -> float:
    """Shift S in [0, 1] between the top-``k`` subspaces of two code matrices.

    Raises ``SubspaceRankError`` if either matrix fails the rank check and
    ``ValueError`` on non-finite codes or mismatched feature dimension.
    """
    if z_pre.dim() != 2 or z_post.dim() != 2 or z_pre.shape[1] != z_post.shape[1]:
        raise ValueError(
            f"codes must be 2-D with equal feature dim; got {tuple(z_pre.shape)}, "
            f"{tuple(z_post.shape)}"
        )
    # Non-finite codes (technical failure) take precedence over any rank
    # failure, whichever matrix carries them.
    _as_finite_f64(z_pre)
    _as_finite_f64(z_post)
    return subspace_shift_from_bases(
        principal_directions(z_pre, k), principal_directions(z_post, k)
    )


def split_half_noise_floor(z_pre: Tensor, k: int = K_DEFAULT, *, seed: int) -> float:
    """S_split: shift between top-``k`` subspaces of two disjoint random halves.

    Rows are permuted with a generator seeded by ``seed`` (the amendment
    uses the experiment seed) and split ``n//2 / n//2`` (256 / 256 for the
    512-sample probe). Both halves must pass the rank check.
    """
    n = z_pre.shape[0] if z_pre.dim() == 2 else 0
    if n < 2:
        raise ValueError("need a 2-D matrix with at least 2 rows to split")
    _as_finite_f64(z_pre)  # a non-finite entry must not hide behind a rank failure
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    half = n // 2
    return subspace_shift(z_pre[perm[:half]], z_pre[perm[half : 2 * half]], k)


def eigengap_ratio(z: Tensor, k: int = K_DEFAULT) -> float:
    """lambda_k / lambda_{k+1} of the centred covariance (stability report).

    ``inf`` when lambda_{k+1} is zero or absent, ``nan`` when lambda_k is zero. A ratio below 1.1 marks a
    top-``k`` subspace that is unstable to small perturbations.
    """
    _check_k(k)
    ev = torch.linalg.svdvals(_centred(z)) ** 2  # covariance eigenvalues, up to 1/(n-1)
    if ev.numel() < k or float(ev[k - 1]) == 0.0:
        return math.nan
    if ev.numel() == k or float(ev[k]) == 0.0:
        return math.inf
    return float(ev[k - 1] / ev[k])
