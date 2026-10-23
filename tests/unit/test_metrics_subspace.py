"""Tests for the v0.8 subspace-shift metric (amendment v0.8, section 3)."""

from __future__ import annotations

import math

import pytest
import torch

from bouba_sens.metrics.subspace import (
    SubspaceRankError,
    eigengap_ratio,
    numerical_rank,
    passes_rank_check,
    principal_directions,
    split_half_noise_floor,
    subspace_shift,
    subspace_shift_from_bases,
)

D = 40
K = 4


def _codes(n: int = 128, d: int = D, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float64)


def _planted(basis: torch.Tensor, n: int = 200, seed: int = 0) -> torch.Tensor:
    """Codes whose top-k right singular subspace is span(basis) (d, k)."""
    g = torch.Generator().manual_seed(seed)
    coef = torch.randn(n, basis.shape[1], generator=g, dtype=torch.float64) * 10.0
    noise = torch.randn(n, basis.shape[0], generator=g, dtype=torch.float64) * 0.01
    return coef @ basis.T + noise


def _std_basis(cols: list[int], d: int = D) -> torch.Tensor:
    return torch.eye(d, dtype=torch.float64)[:, cols]


# --- shift values ---------------------------------------------------------


def test_identical_subspaces_give_zero() -> None:
    z = _codes()
    assert subspace_shift(z, z, k=K) == pytest.approx(0.0, abs=1e-12)


def test_orthogonal_subspaces_give_one() -> None:
    z_pre = _planted(_std_basis([0, 1, 2, 3]))
    z_post = _planted(_std_basis([4, 5, 6, 7]), seed=1)
    assert subspace_shift(z_pre, z_post, k=K) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("theta", [0.1, 0.5, math.pi / 4, 1.2])
def test_known_plane_rotation_closed_form(theta: float) -> None:
    # k=2 subspaces: span(e0,e1) vs span(cos*e0+sin*e2, e1 rotated by same angle e3).
    c, s = math.cos(theta), math.sin(theta)
    b_pre = _std_basis([0, 1])
    b_post = torch.zeros(D, 2, dtype=torch.float64)
    b_post[0, 0], b_post[2, 0] = c, s
    b_post[1, 1], b_post[3, 1] = c, s
    expected = math.sin(theta) ** 2  # both principal angles equal theta
    assert subspace_shift_from_bases(b_pre, b_post) == pytest.approx(expected, abs=1e-12)
    z_pre, z_post = _planted(b_pre), _planted(b_post, seed=1)
    assert subspace_shift(z_pre, z_post, k=2) == pytest.approx(expected, abs=1e-4)


def test_mixed_angles_average() -> None:
    # One direction identical (angle 0), one rotated by theta: S = sin^2(theta)/2.
    theta = 0.7
    b_pre = _std_basis([0, 1])
    b_post = torch.zeros(D, 2, dtype=torch.float64)
    b_post[0, 0] = 1.0
    b_post[1, 1], b_post[2, 1] = math.cos(theta), math.sin(theta)
    assert subspace_shift_from_bases(b_pre, b_post) == pytest.approx(
        math.sin(theta) ** 2 / 2, abs=1e-12
    )


def test_shift_is_in_unit_interval_and_symmetric() -> None:
    a, b = _codes(seed=1), _codes(seed=2)
    s_ab = subspace_shift(a, b, k=K)
    assert 0.0 <= s_ab <= 1.0
    assert s_ab == pytest.approx(subspace_shift(b, a, k=K), abs=1e-12)


# --- invariances ----------------------------------------------------------


def test_invariant_to_sign_flips_of_directions() -> None:
    z = _codes(seed=3)
    v = principal_directions(z, K)
    flipped = v * torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64)
    assert subspace_shift_from_bases(v, flipped) == pytest.approx(0.0, abs=1e-12)


def test_invariant_to_rotation_within_subspace() -> None:
    z = _codes(seed=4)
    v = principal_directions(z, K)
    q, _ = torch.linalg.qr(_codes(K, K, seed=9))
    assert subspace_shift_from_bases(v, v @ q) == pytest.approx(0.0, abs=1e-12)


def test_invariant_to_column_centring_offset() -> None:
    z_pre, z_post = _codes(seed=5), _codes(seed=6)
    offset = torch.randn(1, D, dtype=torch.float64) * 100.0
    assert subspace_shift(z_pre + offset, z_post - offset, k=K) == pytest.approx(
        subspace_shift(z_pre, z_post, k=K), abs=1e-9
    )


def test_invariant_to_global_scaling() -> None:
    z_pre, z_post = _codes(seed=5), _codes(seed=6)
    assert subspace_shift(7.5 * z_pre, 0.2 * z_post, k=K) == pytest.approx(
        subspace_shift(z_pre, z_post, k=K), abs=1e-9
    )


def test_invariant_to_row_permutation() -> None:
    z_pre, z_post = _codes(seed=5), _codes(seed=6)
    perm = torch.randperm(z_pre.shape[0], generator=torch.Generator().manual_seed(0))
    assert subspace_shift(z_pre[perm], z_post, k=K) == pytest.approx(
        subspace_shift(z_pre, z_post, k=K), abs=1e-9
    )


def test_float32_input_accepted() -> None:
    z = _codes(seed=7).float()
    assert subspace_shift(z, z, k=K) == pytest.approx(0.0, abs=1e-9)


# --- rank checks ----------------------------------------------------------


def test_numerical_rank_zero_matrix_is_zero() -> None:
    assert numerical_rank(torch.zeros(64, D)) == 0
    assert not passes_rank_check(torch.zeros(64, D), K)


def test_numerical_rank_constant_matrix_is_zero() -> None:
    const = torch.full((64, D), 3.7)
    assert numerical_rank(const) == 0
    assert not passes_rank_check(const, K)


def test_constant_rows_repeated_vector_is_zero() -> None:
    row = torch.randn(1, D)
    assert numerical_rank(row.expand(64, D).clone()) == 0


def test_rank_deficient_fails_below_k() -> None:
    low = _codes(n=64, d=D) @ torch.diag(
        torch.tensor([1.0] * (K - 1) + [0.0] * (D - K + 1), dtype=torch.float64)
    )
    assert numerical_rank(low) == K - 1
    assert not passes_rank_check(low, K)


def test_rank_exactly_k_passes() -> None:
    exact = _codes(n=64, d=D) @ torch.diag(
        torch.tensor([1.0] * K + [0.0] * (D - K), dtype=torch.float64)
    )
    assert numerical_rank(exact) == K
    assert passes_rank_check(exact, K)


def test_rank_threshold_is_relative_1e6() -> None:
    s = torch.tensor([1.0, 1.0, 1.0, 5e-7], dtype=torch.float64)  # last below 1e-6 * max
    z = torch.diag(s)
    z = torch.cat([z, -z.sum(0, keepdim=True)], dim=0)  # centred rows sum to zero
    assert numerical_rank(z) == 3
    s2 = torch.tensor([1.0, 1.0, 1.0, 2e-6], dtype=torch.float64)
    z2 = torch.diag(s2)
    z2 = torch.cat([z2, -z2.sum(0, keepdim=True)], dim=0)
    assert numerical_rank(z2) == 4


def test_too_few_rows_fails() -> None:
    assert not passes_rank_check(_codes(n=K, d=D), K)  # centred rank <= K-1


@pytest.mark.parametrize("which", ["pre", "post"])
def test_subspace_shift_raises_on_rank_failure(which: str) -> None:
    good, bad = _codes(seed=1), torch.zeros(128, D, dtype=torch.float64)
    args = (bad, good) if which == "pre" else (good, bad)
    with pytest.raises(SubspaceRankError):
        subspace_shift(*args, k=K)


def test_non_finite_raises_value_error() -> None:
    z = _codes()
    z[3, 5] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        numerical_rank(z)
    z[3, 5] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        subspace_shift(z, _codes(seed=1), k=K)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        subspace_shift(_codes(), _codes(d=D + 1), k=K)
    with pytest.raises(ValueError):
        subspace_shift(torch.zeros(10), torch.zeros(10), k=K)
    with pytest.raises(ValueError):
        subspace_shift(_codes(), _codes(), k=0)


# --- split-half noise floor ----------------------------------------------


def test_split_half_floor_small_for_strong_low_rank_signal() -> None:
    z = _planted(_std_basis([0, 1, 2, 3]), n=512)
    assert split_half_noise_floor(z, k=K, seed=0) < 1e-3


def test_split_half_floor_large_for_pure_noise() -> None:
    z = _codes(n=64, d=D, seed=11)  # few samples, high dim: halves disagree
    assert split_half_noise_floor(z, k=K, seed=0) > 0.5


def test_split_half_floor_deterministic_and_seed_dependent() -> None:
    z = _codes(n=256, seed=12)
    a = split_half_noise_floor(z, k=K, seed=3)
    assert a == split_half_noise_floor(z, k=K, seed=3)
    assert a != split_half_noise_floor(z, k=K, seed=4)


def test_split_half_floor_rank_failure_on_a_half() -> None:
    z = _planted(_std_basis([0, 1, 2, 3]), n=256)
    z = torch.cat([z[:128], z[:128]])  # any split still has rank fine; make rank low
    z[:, K:] = 0.0
    z = z @ torch.diag(torch.tensor([1.0] * (K - 1) + [0.0] * (D - K + 1), dtype=torch.float64))
    with pytest.raises(SubspaceRankError):
        split_half_noise_floor(z, k=K, seed=0)


def test_split_half_requires_even_enough_rows() -> None:
    with pytest.raises(ValueError):
        split_half_noise_floor(_codes(n=1), k=K, seed=0)


# --- eigengap stability ---------------------------------------------------


def test_eigengap_ratio_known_spectrum() -> None:
    # Centred covariance eigenvalues proportional to s^2; k=2 -> lambda_2/lambda_3.
    s = torch.tensor([4.0, 3.0, 1.0], dtype=torch.float64)
    base = torch.diag(s)
    z = torch.cat([base, -base.sum(0, keepdim=True) * 0], dim=0)
    z = z - z.mean(0, keepdim=True)
    # Recompute expected from the centred matrix directly.
    ev = torch.linalg.svdvals(z) ** 2
    assert eigengap_ratio(z, k=2) == pytest.approx(float(ev[1] / ev[2]), rel=1e-9)


def test_eigengap_ratio_inf_when_next_eigenvalue_zero() -> None:
    z = _codes(n=64, d=D) @ torch.diag(
        torch.tensor([1.0] * K + [0.0] * (D - K), dtype=torch.float64)
    )
    assert math.isinf(eigengap_ratio(z, k=K))


def test_eigengap_ratio_flat_spectrum_near_one() -> None:
    z = torch.linalg.qr(_codes(n=64, d=D, seed=2))[0] @ torch.eye(D, dtype=torch.float64)
    assert eigengap_ratio(z, k=K) < 1.1
