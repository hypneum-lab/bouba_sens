"""Task 3.1 acceptance tests for Me1 (accuracy) + Me2 (recovery AUC)."""

from __future__ import annotations

from bouba_sens.loop import AdaptationReport
from bouba_sens.metrics.performance import me1_accuracy, me2_recovery_auc


def _report(curve: list[float]) -> AdaptationReport:
    return AdaptationReport(accuracy_curve=curve)


def test_me1_accuracy_on_constant_curve() -> None:
    assert abs(me1_accuracy(_report([0.75] * 100)) - 0.75) < 1e-9


def test_me1_reflects_tail_not_head() -> None:
    """Me1 must use the LAST 10 %, not the full curve mean."""
    curve = [0.0] * 90 + [1.0] * 10  # first 90 steps useless, last 10 perfect
    assert abs(me1_accuracy(_report(curve)) - 1.0) < 1e-9


def test_me1_empty_curve_is_zero() -> None:
    assert me1_accuracy(_report([])) == 0.0


def test_me2_auc_on_flat_curve() -> None:
    """Flat 0.75 curve -> AUC == 0.75 by normalisation."""
    assert abs(me2_recovery_auc(_report([0.75] * 100)) - 0.75) < 1e-9


def test_me2_auc_increasing_curve() -> None:
    """Monotonic 0 -> 1 ramp -> AUC == 0.5."""
    curve = [i / 99 for i in range(100)]
    auc = me2_recovery_auc(_report(curve))
    assert abs(auc - 0.5) < 1e-6


def test_me2_auc_empty_curve_is_zero() -> None:
    assert me2_recovery_auc(_report([])) == 0.0


def test_me2_auc_single_value() -> None:
    assert abs(me2_recovery_auc(_report([0.42])) - 0.42) < 1e-9
