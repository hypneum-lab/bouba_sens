"""v0.8 rev. B outcome: lesion-vs-control representation distance (PI decision 2026-10-25, 19:47).

Per T2 cell c of block (s, l): d(c) = 1 - CKA_lin(Z_post(c), Z_control) on
the same 512 probe rows (both Phase-2 end states restored from the same
checkpoint, same data and noise seeds; only the lesion differs). Linear
CKA = ||Y_c^T X_c||_F^2 / (||X_c^T X_c||_F ||Y_c^T Y_c||_F) on
column-centred matrices; it is invariant to orthogonal transformations
and isotropic scaling of either matrix, not to arbitrary invertible maps.

* Non-finite codes: technical failure (the seed is replaced).
* Non-degeneracy (replaces the rank guard): a matrix is degenerate if its
  total centred variance is <= ``VAR_TOL`` (1e-12). A cell is valid if
  Z_post(c) and Z_control are non-degenerate; if Z_control is degenerate
  the whole block is invalid. Y(s, l) = median of d over valid cells,
  defined if at least 12 of the 15 are valid (as before).

Secondary, the principal-subspace shift at the eigengap k
(``k_gap_from_pilot``): for every pilot Z_pre, the k in
[ceil(PR / 2), floor(2 PR)] (within 1 ... 254) with the largest ratio
lambda_k / lambda_(k+1), eigenvalues of the centred covariance, among k
whose lambda_(k+1) >= 1e-6 lambda_1; k_gap = median over the pilot
matrices, rounded half up.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from bouba_sens.metrics.subspace import participation_ratio
from bouba_sens.metrics.subspace_aggregate import (
    MIN_VALID_CELLS,
    N_CELLS,
    BlockResult,
    TechnicalFailureError,
)
from bouba_sens.subspace_protocol import BlockCodes, CellKey

VAR_TOL = 1e-12
EIG_FLOOR = 1e-6
K_MAX = 254


def _centred(z: Tensor) -> Tensor:
    x = z.detach().double()
    return x - x.mean(0)


def total_variance(z: Tensor) -> float:
    x = _centred(z)
    return float((x**2).sum()) / max(1, x.shape[0] - 1)


def linear_cka(a: Tensor, b: Tensor) -> float:
    """Linear CKA of two row-paired matrices (centred inside)."""
    x, y = _centred(a), _centred(b)
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA needs the same rows in both matrices")
    num = float(torch.linalg.matrix_norm(y.T @ x) ** 2)
    den = float(torch.linalg.matrix_norm(x.T @ x) * torch.linalg.matrix_norm(y.T @ y))
    if den == 0.0:
        raise ValueError("CKA undefined for a degenerate matrix")
    return num / den


def _finite(z: Tensor) -> bool:
    return bool(torch.isfinite(z.detach()).all())


def evaluate_block_cka(block: BlockCodes, *, n_cells: int = N_CELLS) -> BlockResult:
    """Y(s, l) = median_c [1 - CKA(Z_post(c), Z_control)] over valid cells."""
    if len(block.z_post) != n_cells:
        raise ValueError(f"expected {n_cells} cells per block; got {len(block.z_post)}")
    for name, z in (
        ("Z_pre", block.z_pre),
        ("control", block.z_control),
        *((f"Z_post {k}", v) for k, v in block.z_post.items()),
    ):
        if not _finite(z):
            raise TechnicalFailureError(
                f"non-finite {name} at seed={block.seed}, lock={block.lock}"
            )
    keys = list(block.z_post)
    cell_valid: dict[CellKey, bool] = dict.fromkeys(keys, False)
    d: dict[CellKey, float] = {}
    if total_variance(block.z_control) > VAR_TOL:
        for key in keys:
            if total_variance(block.z_post[key]) > VAR_TOL:
                cell_valid[key] = True
                d[key] = 1.0 - linear_cka(block.z_post[key], block.z_control)
    n_valid = sum(cell_valid.values())
    y = float(np.median(list(d.values()))) if n_valid >= MIN_VALID_CELLS else None
    return BlockResult(
        seed=block.seed,
        lock=block.lock,
        n_cells=n_cells,
        n_valid=n_valid,
        y=y,
        s_split=None,
        cell_valid=cell_valid,
        delta_s=d,
    )


def eigenvalues(z: Tensor) -> np.ndarray:
    x = _centred(z)
    lam: np.ndarray = (torch.linalg.svdvals(x) ** 2 / max(1, x.shape[0] - 1)).numpy()
    return lam


def eigengap_k(z: Tensor) -> tuple[int, float]:
    """The eigengap k of one matrix and its ratio (module docstring)."""
    lam = eigenvalues(z)
    pr = participation_ratio(z)
    lo = max(1, math.ceil(pr / 2))
    hi = min(K_MAX, math.floor(2 * pr), len(lam) - 1)
    best_k, best_r = 0, -1.0
    for k in range(lo, hi + 1):
        if lam[k] < EIG_FLOOR * lam[0] or lam[k] <= 0:
            continue
        r = float(lam[k - 1] / lam[k])
        if r > best_r:
            best_k, best_r = k, r
    if best_k == 0:
        raise ValueError("no eligible eigengap in the window")
    return best_k, best_r


@dataclass(frozen=True)
class KGap:
    k: int
    per_matrix: tuple[tuple[int, float], ...]


def k_gap_from_pilot(z_pres: Sequence[Tensor]) -> KGap:
    """Median eigengap k over the non-degenerate pilot matrices; k = 0 if none qualifies.

    Degenerate matrices (total variance <= ``VAR_TOL``) and matrices with
    no eligible gap in their window are skipped (their count is
    len(z_pres) - len(per_matrix)).
    """
    per: list[tuple[int, float]] = []
    for z in z_pres:
        if total_variance(z) <= VAR_TOL:
            continue
        try:
            per.append(eigengap_k(z))
        except ValueError:
            continue
    if not per:
        return KGap(0, ())
    med = statistics.median(k for k, _ in per)
    return KGap(math.floor(med + 0.5), tuple(per))


def ratio_at(z: Tensor, k: int) -> float:
    lam = eigenvalues(z)
    return float(lam[k - 1] / lam[k]) if lam[k] > 0 else math.inf
