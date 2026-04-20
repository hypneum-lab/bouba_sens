"""Task 3.7 acceptance tests for the metrics registry."""

from __future__ import annotations

from bouba_sens.metrics import (
    EvalReport,
    Metric,
    me1_accuracy,
    me3_delta,
    me3_mi,
    me6_asymmetry,
    me6_max_abs_off_diag,
    me7_congenital_gap,
    me8_baselines,
    me9_bootstrap,
)


def test_metric_protocol_is_runtime_checkable() -> None:
    """A duck-typed class with name + update + compute satisfies Metric."""

    class _DummyMetric:
        name = "dummy"

        def update(self, report: EvalReport) -> None:
            pass

        def compute(self) -> dict[str, float]:
            return {}

    assert isinstance(_DummyMetric(), Metric)


def test_eval_report_defaults() -> None:
    report = EvalReport()
    assert report.me1 is None
    assert report.me8 == {}
    assert report.me9 == {}


def test_registry_reexports_all_seven_metrics() -> None:
    """Sanity: every v0.1 metric function must be callable from bouba_sens.metrics."""
    assert callable(me1_accuracy)
    assert callable(me3_mi)
    assert callable(me3_delta)
    assert callable(me6_asymmetry)
    assert callable(me6_max_abs_off_diag)
    assert callable(me7_congenital_gap)
    assert callable(me8_baselines)
    assert callable(me9_bootstrap)
