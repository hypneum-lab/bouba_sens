"""Ross (2014) mixed discrete-continuous k-NN mutual information.

Primary B-2 estimator of the v0.7 amendment (§4.2). Reference: Ross, B. C.
(2014), "Mutual information between discrete and continuous data sets",
PLoS ONE 9(2): e87357, doi:10.1371/journal.pone.0087357.

Definition (frozen, no standardisation / jitter / clipping):

* Euclidean distance on the raw coordinates.
* d_i = distance from x_i to its k-th nearest neighbour among the *other*
  points with the same label.
* m_i = number of points j != i (any label) with ||x_j - x_i|| <= d_i. This
  includes the k-th neighbour itself (Ross's convention) and counts every
  tie at d_i, so m_i >= k.
* I_i = psi(N) - psi(N_y(i)) + psi(k) - psi(m_i), N_y(i) counting point i;
  MI-hat = mean_i I_i (nats), / ln 2 for bits. May be negative.

Exact brute force with chunking (no neighbour-search library): distances are
squared Euclidean sums of coordinate differences, so ties and duplicated
rows are compared exactly, with no Gram-matrix cancellation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import digamma  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

_CHUNK_ELEMENTS = 8_000_000  # max float64 elements of the (rows, N, D) difference block


def _validate(
    x: ArrayLike, y: ArrayLike, k: int
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim == 1:
        xa = xa.reshape(-1, 1)
    if xa.ndim != 2:
        raise ValueError(f"x must have shape (N, D), got {xa.shape}")
    ya = np.asarray(y)
    if ya.ndim != 1 or ya.shape[0] != xa.shape[0]:
        raise ValueError(f"x and y must have the same length, got {xa.shape[0]} and {ya.shape}")
    if not np.all(np.isfinite(xa)):
        raise ValueError("x must be finite")
    _, inverse, counts = np.unique(ya, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1).astype(np.int64)
    if counts.min() < k + 1:
        raise ValueError(
            f"every class needs at least k + 1 = {k + 1} points, smallest has {int(counts.min())}"
        )
    return xa, inverse, counts[inverse].astype(np.int64)


def _squared_distance_rows(xa: NDArray[np.float64], lo: int, hi: int) -> NDArray[np.float64]:
    diff = xa[lo:hi, None, :] - xa[None, :, :]
    out: NDArray[np.float64] = np.einsum("ijk,ijk->ij", diff, diff)
    return out


def _ross_terms(
    xa: NDArray[np.float64], lab: NDArray[np.int64], n_y: NDArray[np.int64], k: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return (d_i squared, m_i) for every point."""
    n, d = xa.shape
    rows = max(1, _CHUNK_ELEMENTS // max(1, n * d))
    d2k = np.empty(n)
    m = np.empty(n, dtype=np.int64)
    for lo in range(0, n, rows):
        hi = min(n, lo + rows)
        d2 = _squared_distance_rows(xa, lo, hi)
        idx = np.arange(lo, hi)
        d2[idx - lo, idx] = np.inf  # exclude self everywhere
        same = lab[lo:hi, None] == lab[None, :]
        d2_same = np.where(same, d2, np.inf)
        kth = np.partition(d2_same, k - 1, axis=1)[:, k - 1]
        d2k[lo:hi] = kth
        m[lo:hi] = np.sum(d2 <= kth[:, None], axis=1)
    return d2k, m


def mi_ross_discrete_continuous_nats(x: ArrayLike, y: ArrayLike, k: int = 3) -> float:
    """MI-hat(x; y) in nats, Ross (2014) estimator. ``x`` is ``(N, D)``, ``y`` is ``(N,)``.

    Raises ``ValueError`` if any class has fewer than ``k + 1`` points
    (a technical failure under the pre-registered protocol).
    """
    xa, lab, n_y = _validate(x, y, k)
    n = xa.shape[0]
    _, m = _ross_terms(xa, lab, n_y, k)
    terms = digamma(n) - digamma(n_y) + digamma(k) - digamma(m)
    return float(np.mean(terms))


def mi_ross_discrete_continuous(x: ArrayLike, y: ArrayLike, k: int = 3) -> float:
    """MI-hat(x; y) in bits (nats / ln 2). No clipping: may be negative."""
    return mi_ross_discrete_continuous_nats(x, y, k) / math.log(2.0)


def ross_n_zero_distance(x: ArrayLike) -> int:
    """Number of rows of ``x`` that have an exact duplicate elsewhere in ``x``.

    Reported next to the estimate, per the amendment (duplicated codes).
    """
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim == 1:
        xa = xa.reshape(-1, 1)
    _, counts = np.unique(xa, axis=0, return_counts=True)
    return int(counts[counts > 1].sum())
