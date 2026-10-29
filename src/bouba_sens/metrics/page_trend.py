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

import itertools
from dataclasses import dataclass
from functools import cache
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


# --- selection safeguard: worst-case completion (v0.8 section 3.1) ---------
#
# Rank exclusions are outcome-dependent, so an incomplete block cannot simply
# be dropped (Codex rounds 2-3). For every block, the missing values are
# completed in the way that MINIMISES the block's Page contribution
# sum_j j * R_j over every strict ranking of the m levels consistent with the
# observed order. The sum of these minima is at most the L of the unobserved
# complete data, whatever the exclusion mechanism, so comparing it with the
# registered complete-data null (within-block exchangeability of the m
# values, strict ranks) gives a conservative test.

_MAX_LEVELS = 7
_CODE_BASE = 8


@cache
def _worst_case_table(m: int) -> np.ndarray:
    """min_L[code] over all strict rank orders, code = observed relative ranks."""
    if not 2 <= m <= _MAX_LEVELS:
        raise ValueError(f"worst-case completion supports 2..{_MAX_LEVELS} levels; got {m}")
    perms = np.array(list(itertools.permutations(range(1, m + 1))), dtype=np.int64)
    l_full = perms @ np.arange(1, m + 1)
    pw = _CODE_BASE ** np.arange(m)
    table = np.full(_CODE_BASE**m, np.iinfo(np.int64).max, dtype=np.int64)
    for mask in range(2**m):
        obs = [j for j in range(m) if not (mask >> j) & 1]
        if obs:
            rel = perms[:, obs].argsort(axis=1).argsort(axis=1) + 1
            code = rel @ pw[obs]
        else:
            code = np.zeros(len(perms), dtype=np.int64)
        np.minimum.at(table, code, l_full)
    return table


def _block_codes(x: np.ndarray) -> np.ndarray:
    """Relative ranks of the observed values (0 where missing), base-8 code.

    Ties among observed values are broken against the trend (the higher
    level gets the lower rank), which can only lower L.
    """
    n, m = x.shape
    miss = np.isnan(x)
    key = np.where(miss, np.inf, x)[:, ::-1]  # reversed: stable sort puts high levels first
    order = np.argsort(key, axis=1, kind="stable")
    ranks_rev = np.empty((n, m), dtype=np.int64)
    np.put_along_axis(ranks_rev, order, np.broadcast_to(np.arange(1, m + 1), (n, m)), axis=1)
    ranks = np.where(miss, 0, ranks_rev[:, ::-1])
    codes: np.ndarray = ranks @ (_CODE_BASE ** np.arange(m))
    return codes


def worst_case_block_l(data: np.ndarray) -> np.ndarray:
    """Per-block minimum of sum_j j * R_j over completions of the NaN entries.

    Complete blocks give their own (strict-rank) contribution; a fully
    missing block gives the fully reversed value sum_j j * (m + 1 - j).
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"need a (blocks, levels) matrix; got shape {x.shape}")
    if np.isinf(x).any():
        raise ValueError("data contain infinite values (NaN marks a missing value)")
    out: np.ndarray = _worst_case_table(x.shape[1])[_block_codes(x)]
    return out


def page_worst_case_test(
    data: np.ndarray,
    *,
    n_perm: int = N_PERM,
    seed: int = PERM_SEED,
    chunk: int = 5_000,
) -> PageResult:
    """Conservative Page test on all blocks, NaN = excluded value (increasing).

    ``l_obs`` is the sum of the per-block worst-case completions; the null
    distribution is that of the complete-data L under within-block
    exchangeability with strict ranks (each block a uniform permutation of
    1..m), estimated with the same generator scheme as
    ``page_permutation_test``.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 2:
        raise ValueError(f"need a (blocks >= 1, levels >= 2) matrix; got shape {x.shape}")
    if n_perm < 1 or chunk < 1:
        raise ValueError("n_perm and chunk must be >= 1")
    n, m = x.shape
    l_obs = float(worst_case_block_l(x).sum())
    weights = np.arange(1, m + 1)
    base = np.broadcast_to(np.arange(1, m + 1, dtype=np.float64), (n, m))[None, :, :]
    rng = np.random.default_rng(seed)
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
        alternative="increasing",
    )
