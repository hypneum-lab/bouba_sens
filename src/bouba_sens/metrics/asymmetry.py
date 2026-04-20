"""Me6 (perceptive / proprioceptive asymmetry index). Spec §5.2."""

from __future__ import annotations

import torch
from torch import Tensor


def me6_asymmetry(perf_matrix: Tensor) -> Tensor:
    """Signed antisymmetry `A[i, j] = perf[i, j] - perf[j, i]`.

    Input `perf_matrix` is a `(n, n)` table (v0.1 uses n=5 over modalities)
    where `perf[i, j]` is the accuracy when modality `i` is lesioned and
    query `j` is probed. The resulting antisymmetric matrix has zero
    diagonal by construction.

    Spec §1.2 B-3 invariant threshold: `max abs off-diagonal > 0.02`
    with reproducible sign structure across seeds.
    """
    if perf_matrix.dim() != 2 or perf_matrix.shape[0] != perf_matrix.shape[1]:
        raise ValueError(f"perf_matrix must be square (n, n); got {tuple(perf_matrix.shape)}")
    return perf_matrix - perf_matrix.T


def me6_max_abs_off_diag(asym_matrix: Tensor) -> float:
    """Maximum absolute off-diagonal value — the B-3 scalar summary."""
    mask = ~torch.eye(asym_matrix.shape[0], dtype=torch.bool)
    return float(asym_matrix[mask].abs().max().item())
