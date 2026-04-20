"""Task 3.3 acceptance tests for Me6 (asymmetry index)."""

from __future__ import annotations

import pytest
import torch

from bouba_sens.metrics.asymmetry import me6_asymmetry, me6_max_abs_off_diag


def test_me6_zero_for_symmetric_matrix() -> None:
    base = torch.randn(5, 5)
    symmetric = (base + base.T) / 2
    asym = me6_asymmetry(symmetric)
    assert torch.allclose(asym, torch.zeros_like(asym), atol=1e-6)


def test_me6_diagonal_is_zero() -> None:
    perf = torch.rand(5, 5)
    asym = me6_asymmetry(perf)
    assert torch.equal(asym.diag(), torch.zeros(5))


def test_me6_max_abs_exceeds_threshold_for_asym_matrix() -> None:
    """A deliberately asymmetric matrix must exceed the B-3 threshold."""
    perf = torch.zeros(5, 5)
    perf[0, 1] = 0.9
    perf[1, 0] = 0.1
    asym = me6_asymmetry(perf)
    assert me6_max_abs_off_diag(asym) > 0.02


def test_me6_antisymmetric_by_construction() -> None:
    perf = torch.rand(5, 5)
    asym = me6_asymmetry(perf)
    assert torch.allclose(asym, -asym.T, atol=1e-6)


def test_me6_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="square"):
        me6_asymmetry(torch.zeros(3, 4))
