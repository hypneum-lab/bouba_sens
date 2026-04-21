"""Platonic-RH alignment metrics (Huh et al. 2024).

Vendored from `nerve_wml.scripts.platonic_rh_alignment` to avoid a run-time
dependency on the sibling research engine. The `mutual_knn` kernel is a
standard cross-model alignment score: for each sample in a batch of N,
find the k nearest neighbours in A's embedding space and in B's, count
the intersection size, and average over the batch. Score in [0, 1].

Interpretation (from Huh 2024, arXiv:2405.07987):
- 1.0 = identical neighbour structure (self-alignment sanity)
- k/N ~ 0.01 at k=10, N=1000 = chance level
- 0.1 to 0.5 = typical cross-model pairs in Huh's vision/language fleet
- Decision threshold in nerve-wml: ``alignment > max(3 * random, 0.05)``
  -> substrates converge under the Platonic RH.

In bouba_sens this kernel is used by the `--metric mutual_knn` option of
`bouba-sens eval` to compare T1 vs T2 probe codes without reducing to
a single-number accuracy. Vendoring keeps the reference implementation
bit-identical and documented per run_id.
"""

from __future__ import annotations

import torch
from torch.nn.functional import normalize

__all__ = ["mutual_knn"]


def mutual_knn(a: torch.Tensor, b: torch.Tensor, *, k: int = 10) -> float:
    """Mutual k-nearest-neighbour overlap (Huh et al. 2024).

    Both inputs are (N, d) embedding tensors with matching N. Returns the
    mean fraction of the k-nearest neighbours that agree between A and B
    under cosine similarity. Score in [0, 1]; higher = more aligned.
    """
    n = a.shape[0]
    assert b.shape[0] == n, f"batch size mismatch: {n} vs {b.shape[0]}"
    assert k < n, f"k={k} must be strictly less than N={n}"
    a_normed = normalize(a.float(), dim=-1)
    b_normed = normalize(b.float(), dim=-1)
    sim_a = a_normed @ a_normed.T
    sim_b = b_normed @ b_normed.T
    diag_mask = torch.eye(n, dtype=torch.bool, device=a.device)
    sim_a.masked_fill_(diag_mask, float("-inf"))
    sim_b.masked_fill_(diag_mask, float("-inf"))
    _, knn_a = sim_a.topk(k, dim=-1)
    _, knn_b = sim_b.topk(k, dim=-1)
    has_a = torch.zeros(n, n, dtype=torch.bool, device=a.device)
    has_b = torch.zeros(n, n, dtype=torch.bool, device=a.device)
    has_a.scatter_(1, knn_a, True)
    has_b.scatter_(1, knn_b, True)
    overlap = (has_a & has_b).sum(dim=-1).float()
    return (overlap / k).mean().item()
