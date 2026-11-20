"""Hypothesis property tests for m2_snr_schedule.

Covers invariants of the linear ramp + floor schedule that hold by
construction (not pre-registered thresholds).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from bouba_sens.lesion import m2_snr_schedule

# Tolerances named per tests/CLAUDE.md convention.
# Floating-point rounding in linear interpolation can introduce tiny error.
_MONOTONE_ATOL: float = 1e-5


@given(
    snr_floor=st.floats(min_value=-100.0, max_value=0.0, allow_nan=False, allow_infinity=False),
    snr_init=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    k=st.integers(min_value=1, max_value=10_000),
    step=st.integers(min_value=0, max_value=20_000),
)
@settings(deadline=None)
def test_schedule_in_range(
    snr_floor: float, snr_init: float, step: int, k: int
) -> None:
    """SNR value is always in [snr_floor, snr_init]."""
    val = m2_snr_schedule(step, snr_floor=snr_floor, snr_init=snr_init, k=k)
    assert val >= snr_floor - _MONOTONE_ATOL, (
        f"value {val} < snr_floor {snr_floor} at step={step}"
    )
    assert val <= snr_init + _MONOTONE_ATOL, (
        f"value {val} > snr_init {snr_init} at step={step}"
    )


@given(
    snr_floor=st.floats(min_value=-100.0, max_value=0.0, allow_nan=False, allow_infinity=False),
    snr_init=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    k=st.integers(min_value=1, max_value=10_000),
    step=st.integers(min_value=0, max_value=19_999),
)
@settings(deadline=None)
def test_schedule_non_increasing(
    snr_floor: float, snr_init: float, step: int, k: int
) -> None:
    """SNR at step+1 is never higher than at step."""
    v0 = m2_snr_schedule(step, snr_floor=snr_floor, snr_init=snr_init, k=k)
    v1 = m2_snr_schedule(step + 1, snr_floor=snr_floor, snr_init=snr_init, k=k)
    assert v1 <= v0 + _MONOTONE_ATOL, (
        f"schedule increased from step={step} ({v0}) to step+1 ({v1})"
    )


@given(
    snr_floor=st.floats(min_value=-100.0, max_value=0.0, allow_nan=False, allow_infinity=False),
    snr_init=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    k=st.integers(min_value=1, max_value=10_000),
)
@settings(deadline=None)
def test_schedule_starts_at_init(
    snr_floor: float, snr_init: float, k: int
) -> None:
    """At step=0 the schedule returns snr_init."""
    val = m2_snr_schedule(0, snr_floor=snr_floor, snr_init=snr_init, k=k)
    assert abs(val - snr_init) <= _MONOTONE_ATOL, (
        f"expected snr_init={snr_init} at step=0, got {val}"
    )


@given(
    snr_floor=st.floats(min_value=-100.0, max_value=0.0, allow_nan=False, allow_infinity=False),
    snr_init=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    k=st.integers(min_value=1, max_value=10_000),
    extra=st.integers(min_value=0, max_value=10_000),
)
@settings(deadline=None)
def test_schedule_clamps_at_floor(
    snr_floor: float, snr_init: float, k: int, extra: int
) -> None:
    """For step >= k the schedule returns snr_floor."""
    val = m2_snr_schedule(k + extra, snr_floor=snr_floor, snr_init=snr_init, k=k)
    assert abs(val - snr_floor) <= _MONOTONE_ATOL, (
        f"expected snr_floor={snr_floor} at step={k + extra}, got {val}"
    )
