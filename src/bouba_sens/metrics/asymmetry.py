"""Me6 (perceptive / proprioceptive asymmetry index). Spec §5.2."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from bouba_sens.metrics.partitions import PERCEPTIVE_PROPRIOCEPTIVE


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


def me6_max_abs_off_diag_partitioned(
    perf_matrix: np.ndarray,
    *,
    modalities: tuple[str, ...],
    partition: tuple[frozenset[str], frozenset[str]] | None = None,
) -> float:
    """Max absolute off-diagonal of the partition-blocked perf matrix.

    This function measures the largest cross-block entry in a performance
    matrix (not the antisymmetry matrix). It is used for null-model studies
    (Task 7.1) where a random 3+2 partition replaces the pre-registered
    perceptive/proprioceptive split.

    When ``partition`` is ``None``, the pre-registered
    ``PERCEPTIVE_PROPRIOCEPTIVE`` partition is used, making this
    function backward-compatible with v0.3 behaviour.
    """
    if partition is None:
        partition = PERCEPTIVE_PROPRIOCEPTIVE
    big, small = partition
    idx_big = np.array([i for i, m in enumerate(modalities) if m in big])
    idx_small = np.array([i for i, m in enumerate(modalities) if m in small])
    # Cross-block asymmetry: big->small minus (small->big).T
    # For a symmetric matrix this is zero; for asymmetric it measures
    # the directional imbalance across the partition boundary.
    block_big_to_small = perf_matrix[np.ix_(idx_big, idx_small)]
    block_small_to_big = perf_matrix[np.ix_(idx_small, idx_big)]
    cross_asym = block_big_to_small - block_small_to_big.T
    return float(np.max(np.abs(cross_asym)))
