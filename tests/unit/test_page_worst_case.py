"""Worst-case completion of incomplete blocks for the v0.8 Page test (Codex round 3)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from bouba_sens.metrics.page_trend import (
    page_l,
    page_permutation_test,
    page_worst_case_test,
    worst_case_block_l,
)

M = 7
W = np.arange(1, M + 1)
ORDERS = np.array(list(itertools.permutations(range(M))), dtype=float)  # 5040 value orders


def _true_l(row: np.ndarray) -> int:
    return int(((np.argsort(np.argsort(row)) + 1) * W).sum())


def _brute_min(row: np.ndarray) -> int:
    """Independent check: try every strict rank completion of the missing levels."""
    miss = np.isnan(row)
    obs = np.where(~miss)[0]
    best = math.inf
    for ranks in itertools.permutations(range(1, M + 1)):
        r = np.array(ranks)
        if all((r[a] < r[b]) == (row[a] < row[b]) for a in obs for b in obs if a != b):
            best = min(best, int((r * W).sum()))
    return int(best)


@pytest.mark.parametrize("k", [1, 2])
def test_completion_never_exceeds_the_true_l_over_all_orders(k: int) -> None:
    for missing in itertools.combinations(range(M), k):
        x = ORDERS.copy()
        x[:, list(missing)] = np.nan
        worst = worst_case_block_l(x)
        true = np.array([_true_l(r) for r in ORDERS])
        assert np.all(worst <= true)
        assert np.all(worst >= 84)  # smallest possible block L: fully reversed ranks


def test_completion_equals_independent_brute_force() -> None:
    rng = np.random.default_rng(0)
    for _ in range(60):
        row = rng.permutation(M).astype(float)
        k = int(rng.integers(1, 4))
        row[rng.choice(M, k, replace=False)] = np.nan
        assert worst_case_block_l(row[None, :])[0] == _brute_min(row)


def test_codex_example_interior_missing_cannot_raise_l() -> None:
    row = np.array([4.0, 5, 6, 3, 0, 1, 2])
    assert _true_l(row) == 92  # 1-based ranks (Codex quotes 88 -> 94, the same +6)
    fixed = row.copy()
    fixed[3] = 1e6  # earlier fixed-direction rule: lock 100 imputed above the maximum
    assert _true_l(fixed) == 98  # +6: the old imputation could raise L
    row[3] = np.nan
    assert worst_case_block_l(row[None, :])[0] <= 92


def test_complete_and_fully_missing_blocks() -> None:
    rows = ORDERS[:50]
    assert np.array_equal(worst_case_block_l(rows), [_true_l(r) for r in rows])
    assert worst_case_block_l(np.full((1, M), np.nan))[0] == int((W * (M + 1 - W)).sum())  # 84


def test_ties_are_broken_against_the_trend() -> None:
    row = np.array([[0.0, 0.0, 1, 2, 3, 4, 5]])
    assert worst_case_block_l(row)[0] == min(
        _true_l(np.array([0.0, 0.1, 1, 2, 3, 4, 5])), _true_l(np.array([0.1, 0.0, 1, 2, 3, 4, 5]))
    )


def test_worst_case_test_on_complete_data_matches_registered_test() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, M)) + np.linspace(0, 0.4, M)
    a = page_worst_case_test(x, n_perm=20_000)
    b = page_permutation_test(x, n_perm=20_000)
    assert a.l_obs == page_l(x) == b.l_obs
    assert abs(a.p_value - b.p_value) < 0.01


def test_worst_case_test_handles_missing_and_is_conservative() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(20, M))
    full = page_worst_case_test(x, n_perm=5_000)
    x[:3, 3] = np.nan
    part = page_worst_case_test(x, n_perm=5_000)
    assert part.l_obs <= full.l_obs and part.p_value >= full.p_value
    with pytest.raises(ValueError):
        page_worst_case_test(np.full((3, M), np.inf), n_perm=10)
