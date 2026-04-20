"""Task 3.6 acceptance tests for Me9 (bootstrap IC)."""

from __future__ import annotations

import numpy as np

from bouba_sens.metrics.replication import me9_bootstrap


def test_me9_mean_matches_numpy_mean() -> None:
    values = [0.1, 0.3, 0.5, 0.7, 0.9]
    mean, _, _ = me9_bootstrap(values, n_resamples=200)
    assert abs(mean - np.mean(values)) < 1e-9


def test_me9_ci_contains_true_mean() -> None:
    """For Gaussian samples around 0.5, the 95 % CI must bracket 0.5."""
    rng = np.random.default_rng(0)
    values = rng.normal(0.5, 0.1, size=50).tolist()
    _, lo, hi = me9_bootstrap(values, n_resamples=1000, seed=0)
    assert lo < 0.5 < hi


def test_me9_reproducible_with_seed() -> None:
    values = [0.2, 0.4, 0.6, 0.8, 1.0] * 4
    a = me9_bootstrap(values, n_resamples=500, seed=42)
    b = me9_bootstrap(values, n_resamples=500, seed=42)
    assert a == b


def test_me9_singleton_returns_degenerate_ci() -> None:
    mean, lo, hi = me9_bootstrap([0.42])
    assert mean == lo == hi == 0.42


def test_me9_empty_returns_zeros() -> None:
    assert me9_bootstrap([]) == (0.0, 0.0, 0.0)


def test_me9_ci_widens_with_variance() -> None:
    """More-variable data -> wider CI."""
    low_var = [0.5] * 20 + [0.51] * 20
    high_var = [0.0, 1.0] * 20
    _, lo_l, hi_l = me9_bootstrap(low_var, n_resamples=500, seed=1)
    _, lo_h, hi_h = me9_bootstrap(high_var, n_resamples=500, seed=1)
    assert (hi_h - lo_h) > (hi_l - lo_l)
