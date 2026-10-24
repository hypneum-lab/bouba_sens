"""Frozen Page trend test (pre-registration amendment v0.8, section 5).

``L = sum_l j(l) * R_l`` where ``R_l`` is the sum over complete blocks
(seeds) of the within-block rank of the level-``l`` value, ``j(l) = 1..m``
in increasing order of the level, ties receiving mid-ranks.

The p-value is a within-block permutation test: the ``m`` ranks of every
block are permuted independently, ``N_PERM = 100_000`` times, with a
generator seeded by ``PERM_SEED = 20261023``::

    p = (1 + #{L_perm >= L_obs}) / (1 + n_perm)

(one-sided, increasing). The null is that, within each block, the ``m``
values are exchangeable across levels; the test is then exact up to Monte
Carlo error and valid with ties. ``alternative="decreasing"`` mirrors the
level order (exploratory contrary-trend report).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

N_PERM = 100_000
PERM_SEED = 20_261_023
_TOL = 1e-9  # mid-rank sums are half-integers; tolerance guards float round-off


def midranks(data: np.ndarray) -> np.ndarray:
    """Row-wise ranks (1-based) with mid-ranks for ties."""
    x = np.asarray(data, dtype=np.float64)
    out = np.empty_like(x)
    for i, row in enumerate(x):
        _, inv, counts = np.unique(row, return_inverse=True, return_counts=True)
        upper = np.cumsum(counts)  # last rank of each tie group
        lower = upper - counts + 1  # first rank of each tie group
        out[i] = ((lower + upper) / 2.0)[inv]
    return out


def _validate(data: np.ndarray) -> np.ndarray:
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 2:
        raise ValueError(f"need a (blocks >= 1, levels >= 2) matrix; got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("data contain non-finite values")
    return x


def page_l(data: np.ndarray) -> float:
    """Observed Page L statistic (increasing alternative), mid-ranks for ties."""
    x = _validate(data)
    ranks = midranks(x)
    return float((ranks * np.arange(1, x.shape[1] + 1)).sum())


@dataclass(frozen=True)
class PageResult:
    l_obs: float
    p_value: float
    n_blocks: int
    n_levels: int
    n_perm: int
    seed: int
    alternative: str


def page_permutation_test(
    data: np.ndarray,
    *,
    n_perm: int = N_PERM,
    seed: int = PERM_SEED,
    alternative: Literal["increasing", "decreasing"] = "increasing",
    chunk: int = 5_000,
) -> PageResult:
    """One-sided within-block permutation Page test on a ``(blocks, levels)`` matrix.

    ``chunk`` only bounds memory; the result does not depend on it.
    """
    x = _validate(data)
    if alternative == "decreasing":
        x = x[:, ::-1]
    elif alternative != "increasing":
        raise ValueError(f"alternative must be 'increasing' or 'decreasing'; got {alternative!r}")
    if n_perm < 1 or chunk < 1:
        raise ValueError("n_perm and chunk must be >= 1")
    ranks = midranks(x)
    n, m = ranks.shape
    weights = np.arange(1, m + 1)
    l_obs = float((ranks * weights).sum())
    rng = np.random.default_rng(seed)
    base = ranks[None, :, :]
    hits = 0
    done = 0
    while done < n_perm:
        size = min(chunk, n_perm - done)
        order = rng.random((size, n, m)).argsort(axis=-1)
        permuted = np.take_along_axis(np.broadcast_to(base, (size, n, m)), order, axis=-1)
        l_perm = (permuted * weights).sum(axis=(-2, -1))
        hits += int(np.count_nonzero(l_perm >= l_obs - _TOL))
        done += size
    return PageResult(
        l_obs=l_obs,
        p_value=(1 + hits) / (1 + n_perm),
        n_blocks=n,
        n_levels=m,
        n_perm=n_perm,
        seed=seed,
        alternative=alternative,
    )
