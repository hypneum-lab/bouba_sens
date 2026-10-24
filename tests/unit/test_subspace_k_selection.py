"""Pilot rule for k (amendment v0.8 section 3 item 3, PI decision 2026-10-24).

k = round-half-up of the median participation ratio of the pilot Z_pre
matrices, clamped to [1, 255]. Deterministic, order-free, pilot-only.
"""

from __future__ import annotations

import pytest
import torch

from bouba_sens.metrics.subspace import (
    K_SELECTION_MAX,
    k_from_median_pr,
    participation_ratio,
    select_k_from_pilot,
)


def _equal_variance(n: int, d: int, r: int) -> torch.Tensor:
    """n x d codes whose centred covariance has exactly r equal non-zero eigenvalues."""
    gen = torch.Generator().manual_seed(r)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    cols = torch.randn(n, r, generator=gen, dtype=torch.float64)
    cols = cols - cols.mean(0)
    u, _, _ = torch.linalg.svd(cols, full_matrices=False)  # orthonormal, centred columns
    return (u * n**0.5) @ q[:, :r].T


def _pr_ref(z: torch.Tensor) -> float:
    zc = z.double() - z.double().mean(0)
    lam = torch.linalg.eigvalsh(zc.T @ zc / (z.shape[0] - 1))
    lam = lam[lam > 0]
    return float(lam.sum() ** 2 / (lam**2).sum())


def test_participation_ratio_matches_definition() -> None:
    z = torch.randn(512, 64, generator=torch.Generator().manual_seed(1)) * torch.linspace(
        0.1, 3.0, 64
    )
    assert participation_ratio(z) == pytest.approx(_pr_ref(z), rel=1e-10)


def test_participation_ratio_equal_spectrum_is_rank() -> None:
    for r in (1, 5, 12):
        z = _equal_variance(64, 30, r)
        assert participation_ratio(z) == pytest.approx(r, rel=1e-9)


def test_participation_ratio_invariances_and_zero() -> None:
    gen = torch.Generator().manual_seed(2)
    z = torch.randn(200, 20, generator=gen, dtype=torch.float64) * torch.arange(1, 21)
    base = participation_ratio(z)
    assert participation_ratio(3.0 * z + 7.0) == pytest.approx(base, rel=1e-10)
    assert participation_ratio(torch.zeros(10, 4)) == 0.0
    assert participation_ratio(torch.ones(10, 4) * 5) == 0.0
    with pytest.raises(ValueError, match="non-finite"):
        participation_ratio(torch.tensor([[1.0, float("nan")], [0.0, 1.0]]))


def _with_pr(target_rank: int) -> torch.Tensor:
    return _equal_variance(64, 40, target_rank)


def test_rounding_is_half_up_and_clamped() -> None:
    # exact binary values: no float noise at the .5 boundary
    assert k_from_median_pr(11.5) == 12
    assert k_from_median_pr(10.5) == 11  # not banker's rounding to 10
    assert k_from_median_pr(11.49) == 11
    assert k_from_median_pr(0.0) == 1
    assert k_from_median_pr(0.4) == 1
    assert k_from_median_pr(1e6) == K_SELECTION_MAX


def test_k_is_rounded_median_of_pilot_prs() -> None:
    sel = select_k_from_pilot([_with_pr(r) for r in (11, 12, 13)])
    assert sel.k == 12
    assert sel.median_pr == pytest.approx(12.0, rel=1e-9)
    # even count: median = mean of the two middle values (11 + 13) / 2
    sel = select_k_from_pilot([_with_pr(r) for r in (9, 11, 13, 20)])
    assert sel.median_pr == pytest.approx(12.0, rel=1e-9)
    assert sel.k == 12
    sel = select_k_from_pilot([_with_pr(r) for r in (6, 9)])
    assert sel.median_pr == pytest.approx(7.5, rel=1e-9)
    assert sel.k == k_from_median_pr(sel.median_pr)


def test_k_selection_is_order_free_and_reports_inputs() -> None:
    zs = [_with_pr(r) for r in (3, 17, 8, 12, 5)]
    a = select_k_from_pilot(zs)
    b = select_k_from_pilot(list(reversed(zs)))
    assert a.k == b.k == 8
    assert a.n_matrices == 5
    assert sorted(a.prs) == pytest.approx(sorted(b.prs))


def test_k_selection_clamps_and_refuses_bad_input() -> None:
    assert select_k_from_pilot([torch.zeros(8, 3)]).k == 1
    assert K_SELECTION_MAX == 255
    wide = _equal_variance(600, 400, 300)  # participation ratio 300
    assert select_k_from_pilot([wide]).k == K_SELECTION_MAX
    with pytest.raises(ValueError, match="at least one"):
        select_k_from_pilot([])
    with pytest.raises(ValueError, match="non-finite"):
        select_k_from_pilot([torch.full((4, 2), float("inf"))])
