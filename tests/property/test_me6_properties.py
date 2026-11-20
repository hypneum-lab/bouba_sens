"""Hypothesis property tests for Me6 asymmetry functions.

Covers algebraic invariants of `me6_asymmetry` and `me6_max_abs_off_diag`
that hold by construction (not pre-registered thresholds).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from bouba_sens.metrics.asymmetry import me6_asymmetry, me6_max_abs_off_diag

# Bounded finite floats avoid NaN/Inf propagation through subtraction.
_FLOAT_STRATEGY = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


def _square_tensor(n: int) -> st.SearchStrategy[torch.Tensor]:
    """Hypothesis strategy: (n, n) finite float tensor."""
    return arrays(dtype=np.float32, shape=(n, n), elements=_FLOAT_STRATEGY).map(torch.from_numpy)


@given(
    n=st.integers(min_value=2, max_value=8),
    data=st.data(),
)
@settings(deadline=None)
def test_me6_antisymmetry(n: int, data: st.DataObject) -> None:
    """a = me6_asymmetry(p)  satisfies  a == -a.T  for any square p."""
    p = data.draw(_square_tensor(n))
    a = me6_asymmetry(p)
    assert torch.equal(a, -a.T), f"antisymmetry violated for n={n}"


@given(
    n=st.integers(min_value=2, max_value=8),
    data=st.data(),
)
@settings(deadline=None)
def test_me6_diagonal_zero(n: int, data: st.DataObject) -> None:
    """Diagonal of a = me6_asymmetry(p) is identically zero."""
    p = data.draw(_square_tensor(n))
    a = me6_asymmetry(p)
    assert torch.equal(a.diag(), torch.zeros(n, dtype=a.dtype))


@given(
    n=st.integers(min_value=2, max_value=8),
    data=st.data(),
)
@settings(deadline=None)
def test_me6_max_abs_off_diag_nonneg(n: int, data: st.DataObject) -> None:
    """me6_max_abs_off_diag returns a non-negative value."""
    p = data.draw(_square_tensor(n))
    a = me6_asymmetry(p)
    assert me6_max_abs_off_diag(a) >= 0.0


@given(
    n=st.integers(min_value=2, max_value=8),
    data=st.data(),
)
@settings(deadline=None)
def test_me6_transpose_negates(n: int, data: st.DataObject) -> None:
    """me6_asymmetry(p.T) == -me6_asymmetry(p)."""
    p = data.draw(_square_tensor(n))
    a = me6_asymmetry(p)
    at = me6_asymmetry(p.T)
    assert torch.equal(at, -a), "me6_asymmetry(p.T) != -me6_asymmetry(p)"


def test_me6_non_square_raises() -> None:
    """A non-square input must raise ValueError."""
    with pytest.raises(ValueError, match="square"):
        me6_asymmetry(torch.randn(3, 4))


def test_me6_1d_raises() -> None:
    """A 1-D input must raise ValueError."""
    with pytest.raises(ValueError, match="square"):
        me6_asymmetry(torch.randn(5))
