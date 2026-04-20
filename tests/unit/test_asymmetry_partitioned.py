"""Tests for the partition-aware Me6 scalar (Task 7.1 Step 6)."""

from __future__ import annotations

import numpy as np
import pytest

from bouba_sens.metrics.asymmetry import me6_max_abs_off_diag_partitioned
from bouba_sens.metrics.partitions import PERCEPTIVE_PROPRIOCEPTIVE

MODALITIES = ("audio", "vision", "tactile", "gravity", "force")


def _symmetric_matrix() -> np.ndarray:
    rng = np.random.default_rng(0)
    m = rng.uniform(0.2, 0.3, (5, 5))
    return (m + m.T) / 2


def _asymmetric_matrix() -> np.ndarray:
    # Strongly asymmetric only on the pre-reg partition boundary.
    # audio(0)->gravity(3) and audio(0)->force(4) are much larger than reverse.
    # The alt partition {audio,gravity,force} vs {vision,tactile} puts
    # these entries WITHIN the big group, so cross-block score differs.
    m = np.full((5, 5), 0.25)
    m[0, 3] += 0.3  # audio helped by gravity (cross pre-reg boundary)
    m[0, 4] += 0.3  # audio helped by force (cross pre-reg boundary)
    return m


def test_prereg_partition_is_default() -> None:
    perf = _asymmetric_matrix()
    default_score = me6_max_abs_off_diag_partitioned(perf, modalities=MODALITIES)
    explicit_score = me6_max_abs_off_diag_partitioned(
        perf, modalities=MODALITIES, partition=PERCEPTIVE_PROPRIOCEPTIVE
    )
    assert default_score == explicit_score


def test_partition_swap_changes_score() -> None:
    perf = _asymmetric_matrix()
    alt = (frozenset({"audio", "gravity", "force"}), frozenset({"vision", "tactile"}))
    prereg_score = me6_max_abs_off_diag_partitioned(perf, modalities=MODALITIES)
    alt_score = me6_max_abs_off_diag_partitioned(perf, modalities=MODALITIES, partition=alt)
    assert prereg_score != pytest.approx(alt_score)


def test_symmetric_matrix_scores_near_zero_under_any_partition() -> None:
    sym = _symmetric_matrix()
    alt = (frozenset({"audio", "tactile", "force"}), frozenset({"vision", "gravity"}))
    for partition in (None, alt):
        score = me6_max_abs_off_diag_partitioned(sym, modalities=MODALITIES, partition=partition)
        assert abs(score) < 0.05
