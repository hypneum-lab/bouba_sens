"""Me1 (accuracy post-adaptation) + Me2 (recovery curve AUC). Spec §5.2."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bouba_sens.loop import AdaptationReport


def me1_accuracy(report: AdaptationReport) -> float:
    """Final-phase accuracy: mean of the last 10 % of the curve.

    Represents "accuracy post-adaptation" per spec §5.2. On a flat curve
    this collapses to the constant value; on a rising recovery curve it
    captures the plateau.
    """
    curve = report.accuracy_curve
    if not curve:
        return 0.0
    tail = max(1, len(curve) // 10)
    return float(np.mean(curve[-tail:]))


def me2_recovery_auc(report: AdaptationReport) -> float:
    """Normalised AUC of the accuracy recovery curve over Phase 2.

    Trapezoidal integration of `accuracy_curve` divided by the number of
    segments (N - 1), giving a value in `[0, 1]` that matches the mean
    of a flat curve and equals 0.5 for a linear 0 -> 1 ramp.
    """
    values = report.accuracy_curve
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    curve = np.asarray(values, dtype=np.float64)
    # numpy 2.x dropped `np.trapz` in favour of `np.trapezoid`; support both.
    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # type: ignore[attr-defined]
    return float(trapz_fn(curve) / (curve.size - 1))
