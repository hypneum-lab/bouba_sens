"""Metric implementations — each corresponds to an entry in spec §5.2.

Public surface:
- `Metric`        : runtime-checkable Protocol per spec §3.7.
- `EvalReport`    : container passed through the Protocol's `update`.
- `me1_accuracy`, `me2_recovery_auc`, `me3_mi`, `me3_delta`,
  `me6_asymmetry`, `me6_max_abs_off_diag`, `me7_congenital_gap`,
  `me8_baselines`, `freeze_nerve_plasticity`, `random_rewire_nerve`,
  `me9_bootstrap`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from bouba_sens.metrics.asymmetry import me6_asymmetry, me6_max_abs_off_diag
from bouba_sens.metrics.baselines import (
    freeze_nerve_plasticity,
    me8_baselines,
    random_rewire_nerve,
)
from bouba_sens.metrics.congenital import me7_congenital_gap
from bouba_sens.metrics.mi_migration import me3_delta, me3_mi
from bouba_sens.metrics.performance import me1_accuracy, me2_recovery_auc
from bouba_sens.metrics.replication import me9_bootstrap


@dataclass
class EvalReport:
    """Aggregated metric values from one run; serialised to `eval_report.json`."""

    me1: float | None = None
    me2: float | None = None
    me3_delta: float | None = None
    me6_max_abs: float | None = None
    me7: float | None = None
    me8: dict[str, float] = field(default_factory=dict)
    me9: dict[str, tuple[float, float, float]] = field(default_factory=dict)


@runtime_checkable
class Metric(Protocol):
    """Metric Protocol per spec §3.7.

    Concrete metrics are free to subclass OR simply duck-type this shape.
    The Sprint 3 metric functions above are pure callables — they do NOT
    implement this Protocol directly; the Protocol is reserved for
    stateful accumulating metrics that Sprint 4 may introduce for
    streaming-evaluation scenarios.
    """

    name: str

    def update(self, report: EvalReport) -> None: ...

    def compute(self) -> dict[str, float]: ...


__all__ = [
    "EvalReport",
    "Metric",
    "freeze_nerve_plasticity",
    "me1_accuracy",
    "me2_recovery_auc",
    "me3_delta",
    "me3_mi",
    "me6_asymmetry",
    "me6_max_abs_off_diag",
    "me7_congenital_gap",
    "me8_baselines",
    "me9_bootstrap",
    "random_rewire_nerve",
]
