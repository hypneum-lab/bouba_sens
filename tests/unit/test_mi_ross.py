"""Pre-filing implementation check for the Ross (2014) mixed MI estimator.

Mirrors the pass/fail list of the v0.7 amendment (§4.3): brute-force
agreement to 1e-10, invariances to 1e-10, and a 1-D known-MI test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import digamma

from bouba_sens.metrics.mi_ross import (
    mi_ross_discrete_continuous,
    mi_ross_discrete_continuous_nats,
    ross_n_zero_distance,
)

LN2 = math.log(2.0)


def brute_force_nats(x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    """Independent O(N^2) pure-Python transcription of the §4.2 definition."""
    n = len(y)
    total = 0.0
    for i in range(n):
        same = [j for j in range(n) if j != i and y[j] == y[i]]
        d2_same = sorted(float(np.sum((x[j] - x[i]) ** 2)) for j in same)
        d2_k = d2_same[k - 1]
        n_y = len(same) + 1
        m = sum(1 for j in range(n) if j != i and float(np.sum((x[j] - x[i]) ** 2)) <= d2_k)
        total += digamma(n) - digamma(n_y) + digamma(k) - digamma(m)
    return total / n


def _random_fixture(seed: int, n: int, d: int, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n)
    y[: n_classes * 5] = np.repeat(np.arange(n_classes), 5)  # every class has >= 5 points
    x = rng.normal(size=(n, d)) + 0.7 * y[:, None] * rng.normal(size=(1, d))
    return x, y


@pytest.mark.parametrize(
    ("seed", "n", "d", "n_classes"),
    [(0, 40, 1, 2), (1, 60, 3, 4), (2, 80, 17, 3), (3, 50, 64, 4), (4, 120, 5, 6)],
)
def test_agrees_with_brute_force_random(seed: int, n: int, d: int, n_classes: int) -> None:
    x, y = _random_fixture(seed, n, d, n_classes)
    fast = mi_ross_discrete_continuous_nats(x, y, k=3)
    assert abs(fast - brute_force_nats(x, y, 3)) < 1e-10


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_agrees_with_brute_force_other_k(k: int) -> None:
    x, y = _random_fixture(7, 70, 4, 3)
    assert abs(mi_ross_discrete_continuous_nats(x, y, k=k) - brute_force_nats(x, y, k)) < 1e-10


def test_agrees_with_brute_force_chunk_boundaries() -> None:
    """N not a multiple of the internal chunk size."""
    x, y = _random_fixture(11, 197, 6, 4)
    assert abs(mi_ross_discrete_continuous_nats(x, y) - brute_force_nats(x, y)) < 1e-10


def test_duplicated_rows_and_integer_ties() -> None:
    """Integer lattice -> many exact ties at d_i; plus exactly duplicated rows."""
    rng = np.random.default_rng(5)
    x = rng.integers(0, 3, size=(60, 2)).astype(float)
    y = rng.integers(0, 3, size=60)
    y[:12] = np.repeat(np.arange(3), 4)
    x[30:40] = x[0:10]  # duplicated rows (labels may differ -> cross-label zeros)
    fast = mi_ross_discrete_continuous_nats(x, y)
    assert abs(fast - brute_force_nats(x, y)) < 1e-10


def test_all_identical_points_matches_brute_force() -> None:
    """d_i = 0 everywhere: m_i = N - 1 and the estimate is still defined."""
    x = np.zeros((20, 3))
    y = np.repeat([0, 1], 10)
    fast = mi_ross_discrete_continuous_nats(x, y)
    assert abs(fast - brute_force_nats(x, y)) < 1e-10
    assert ross_n_zero_distance(x) == 20


def test_zero_distance_count_reports_duplicates() -> None:
    x = np.arange(10, dtype=float).reshape(-1, 1)
    x[5] = x[4]
    assert ross_n_zero_distance(x) == 2


def test_label_permutation_invariance() -> None:
    x, y = _random_fixture(21, 100, 8, 4)
    perm = np.array([2, 0, 3, 1])
    relabelled = np.array([10, 20, 30, 40])[perm[y]]
    a = mi_ross_discrete_continuous_nats(x, y)
    assert abs(a - mi_ross_discrete_continuous_nats(x, relabelled)) < 1e-10


def test_row_permutation_invariance() -> None:
    x, y = _random_fixture(22, 100, 8, 4)
    p = np.random.default_rng(0).permutation(len(y))
    a = mi_ross_discrete_continuous_nats(x, y)
    assert abs(a - mi_ross_discrete_continuous_nats(x[p], y[p])) < 1e-10


def test_translation_and_rotation_invariance() -> None:
    x, y = _random_fixture(23, 100, 8, 4)
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    x2 = x @ q + rng.normal(size=(1, 8)) * 5.0
    a = mi_ross_discrete_continuous_nats(x, y)
    assert abs(a - mi_ross_discrete_continuous_nats(x2, y)) < 1e-10


def test_uniform_scaling_invariance() -> None:
    x, y = _random_fixture(24, 100, 8, 4)
    a = mi_ross_discrete_continuous_nats(x, y)
    for s in (0.25, 8.0):  # powers of two: scaling is exact in floating point
        assert abs(a - mi_ross_discrete_continuous_nats(x * s, y)) < 1e-10


def test_bits_are_nats_over_ln2() -> None:
    x, y = _random_fixture(25, 80, 4, 3)
    nats = mi_ross_discrete_continuous_nats(x, y)
    assert mi_ross_discrete_continuous(x, y) == pytest.approx(nats / LN2, abs=1e-12)


def test_no_clipping_estimate_can_be_negative() -> None:
    """Independent labels: at least one small-N draw gives a negative estimate."""
    vals = []
    for s in range(30):
        rng = np.random.default_rng(100 + s)
        x = rng.normal(size=(30, 2))
        y = np.repeat([0, 1, 2], 10)
        vals.append(mi_ross_discrete_continuous(x, y))
    assert min(vals) < 0.0


def test_ceiling_near_label_entropy() -> None:
    """Perfectly separated balanced classes: I <= psi(N) - psi(N_y) ~ log2(4) bits."""
    rng = np.random.default_rng(3)
    y = np.repeat(np.arange(4), 64)
    x = rng.normal(size=(256, 2)) * 0.01 + 100.0 * y[:, None]
    mi = mi_ross_discrete_continuous(x, y)
    cap = (digamma(256) - digamma(64)) / LN2
    assert mi <= cap + 1e-9
    assert mi > 1.8


def test_class_with_k_or_fewer_members_raises() -> None:
    x = np.random.default_rng(0).normal(size=(13, 2))
    y = np.array([0] * 10 + [1] * 3)  # class 1 has 3 = k members, needs k + 1
    with pytest.raises(ValueError, match="at least k \\+ 1"):
        mi_ross_discrete_continuous(x, y, k=3)
    # k = 2 is fine for the same data
    assert math.isfinite(mi_ross_discrete_continuous(x, y, k=2))


def test_input_validation() -> None:
    x = np.zeros((10, 2))
    y = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="same length"):
        mi_ross_discrete_continuous(x, y[:5])
    with pytest.raises(ValueError, match="k must be"):
        mi_ross_discrete_continuous(x, y, k=0)
    bad = x.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        mi_ross_discrete_continuous(bad, y)


def test_1d_input_is_accepted() -> None:
    x, y = _random_fixture(30, 60, 1, 3)
    a = mi_ross_discrete_continuous_nats(x, y)
    assert a == mi_ross_discrete_continuous_nats(x[:, 0], y)


# ---------------------------------------------------------------- known MI


def _true_mi_bits_gaussian_mixture(a: float, n_classes: int = 4) -> float:
    """I(Y; X) for equiprobable classes N(j*a, 1), by trapezoid integration."""
    xs = np.linspace(-12.0, (n_classes - 1) * a + 12.0, 200001)
    mus = a * np.arange(n_classes)
    log_c = -0.5 * (xs[None, :] - mus[:, None]) ** 2 - 0.5 * math.log(2 * math.pi)
    top = log_c.max(axis=0)
    log_p = top + np.log(np.exp(log_c - top).mean(axis=0))  # stable log-mean-exp
    integrand = np.exp(log_c) * (log_c - log_p)  # p(x|y) log(p(x|y)/p(x))
    return float(np.trapezoid(integrand.mean(axis=0), xs) / LN2)


def test_true_mi_integrator_sanity() -> None:
    assert abs(_true_mi_bits_gaussian_mixture(0.0)) < 1e-9
    assert abs(_true_mi_bits_gaussian_mixture(60.0) - 2.0) < 1e-6  # well separated


@pytest.mark.parametrize("a", [0.0, 0.5, 1.0])
def test_known_mi_1d_four_gaussian_classes(a: float) -> None:
    """N = 512, 200 replicates: mean estimate within +-0.02 bit of the truth."""
    n, reps = 512, 200
    truth = _true_mi_bits_gaussian_mixture(a)
    y = np.repeat(np.arange(4), n // 4)
    est = []
    for r in range(reps):
        rng = np.random.default_rng(10_000 * (1 + int(2 * a)) + r)
        x = rng.normal(size=(n, 1)) + a * y[:, None]
        est.append(mi_ross_discrete_continuous(x, y, k=3))
    mean = float(np.mean(est))
    print(f"a={a}: true={truth:.5f} mean={mean:.5f} sd={np.std(est):.5f} bit")
    assert abs(mean - truth) <= 0.02, f"a={a}: mean {mean:.4f} vs true {truth:.4f} bit"
