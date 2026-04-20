"""Task 3.4 acceptance tests for Me7 (congenital gap)."""

from __future__ import annotations

from bouba_sens.metrics.congenital import me7_congenital_gap


def test_me7_positive_when_t1_beats_t2() -> None:
    assert abs(me7_congenital_gap(0.70, 0.60) - 0.10) < 1e-9


def test_me7_zero_for_equal_perf() -> None:
    assert me7_congenital_gap(0.55, 0.55) == 0.0


def test_me7_negative_when_t2_beats_t1() -> None:
    """Negative gap signals an invariant B-1 failure."""
    assert me7_congenital_gap(0.50, 0.60) < 0
