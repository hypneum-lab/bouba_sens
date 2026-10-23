"""Frozen Page trend test (amendment v0.8, section 5)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import stats

from bouba_sens.metrics.page_trend import (
    N_PERM,
    PERM_SEED,
    midranks,
    page_l,
    page_permutation_test,
)


def _exact_p(data: np.ndarray) -> float:
    """Exhaustive within-block permutation p-value of Page's L (handles ties)."""
    ranks = midranks(data)
    n, m = ranks.shape
    w = np.arange(1, m + 1)
    l_obs = float((ranks * w).sum())
    total = 0
    hits = 0
    perms = [list(itertools.permutations(r)) for r in ranks]
    for combo in itertools.product(*perms):
        total += 1
        if sum(float((np.array(p) * w).sum()) for p in combo) >= l_obs - 1e-9:
            hits += 1
    return hits / total


def test_frozen_constants() -> None:
    assert N_PERM == 100_000
    assert PERM_SEED == 20_261_023


def test_midranks_ties() -> None:
    r = midranks(np.array([[3.0, 1.0, 3.0, 2.0], [5.0, 5.0, 5.0, 5.0]]))
    assert r.tolist() == [[3.5, 1.0, 3.5, 2.0], [2.5, 2.5, 2.5, 2.5]]


def test_page_l_matches_scipy_statistic() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(6, 7))
    assert page_l(data) == pytest.approx(stats.page_trend_test(data, ranked=False).statistic)


@pytest.mark.parametrize("shape", [(3, 4), (4, 3), (2, 5)])
def test_p_value_matches_scipy_exact(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(shape[0] * 10 + shape[1])
    data = rng.normal(size=shape) + 0.8 * np.arange(shape[1])
    exact = stats.page_trend_test(data, ranked=False, method="exact").pvalue
    assert exact == pytest.approx(_exact_p(data))  # the exhaustive helper is itself correct
    got = page_permutation_test(data).p_value
    se = math.sqrt(exact * (1 - exact) / N_PERM)
    assert abs(got - exact) < 5 * se + 2e-5


def test_p_value_with_ties_matches_exhaustive_enumeration() -> None:
    data = np.array([[1.0, 2.0, 2.0, 5.0], [0.0, 0.0, 3.0, 4.0], [2.0, 1.0, 3.0, 3.0]])
    exact = _exact_p(data)
    got = page_permutation_test(data).p_value
    se = math.sqrt(exact * (1 - exact) / N_PERM)
    assert abs(got - exact) < 5 * se + 2e-5


def test_fully_tied_data_gives_p_one() -> None:
    assert page_permutation_test(np.ones((5, 4))).p_value == 1.0


def test_perfect_trend_hits_the_floor() -> None:
    data = np.tile(np.arange(7.0), (20, 1))
    res = page_permutation_test(data, n_perm=1000)
    assert res.p_value == pytest.approx(1 / 1001)


def test_seeded_and_chunk_invariant() -> None:
    data = np.random.default_rng(3).normal(size=(8, 7))
    a = page_permutation_test(data, n_perm=5000)
    b = page_permutation_test(data, n_perm=5000)
    c = page_permutation_test(data, n_perm=5000, chunk=777)
    d = page_permutation_test(data, n_perm=5000, seed=1)
    assert a.p_value == b.p_value == c.p_value
    assert a.p_value != d.p_value


def test_decreasing_alternative_is_increasing_on_reversed_levels() -> None:
    data = np.random.default_rng(4).normal(size=(10, 7)) - 0.4 * np.arange(7)
    dec = page_permutation_test(data, n_perm=4000, alternative="decreasing")
    inc_rev = page_permutation_test(data[:, ::-1], n_perm=4000)
    assert dec.p_value == inc_rev.p_value
    assert dec.p_value < page_permutation_test(data, n_perm=4000).p_value


def test_result_fields() -> None:
    res = page_permutation_test(np.arange(12.0).reshape(3, 4), n_perm=200)
    assert (res.n_blocks, res.n_levels, res.n_perm, res.seed) == (3, 4, 200, PERM_SEED)
    assert res.l_obs == pytest.approx(3 * (1 + 4 + 9 + 16))


@pytest.mark.parametrize(
    "bad",
    [np.array([1.0, 2.0, 3.0]), np.ones((3, 1)), np.array([[1.0, np.nan, 2.0]]), np.ones((0, 4))],
)
def test_rejects_malformed_input(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        page_permutation_test(bad, n_perm=10)
