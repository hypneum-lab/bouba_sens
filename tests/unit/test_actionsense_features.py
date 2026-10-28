"""Unit tests for the pure ActionSense reduction helpers."""

from __future__ import annotations

import numpy as np
import pytest

from bouba_sens.world.actionsense_features import (
    activity_segments,
    check_aligned,
    emg_sectors,
    frame_grid,
    label_family,
    pool_tactile,
    window_mean,
    zscore,
)


def test_label_family_maps_and_rejects() -> None:
    assert label_family("Slice a cucumber") == 0
    assert label_family("Open/close a jar of almond butter") == 1
    assert label_family("Clean a plate with a towel") == 2
    assert label_family("Pour water from a pitcher into a glass") == 3
    assert label_family("Get items from refrigerator") is None


def test_activity_segments_counts_unclosed_and_bad() -> None:
    rows = [
        ("Slice a potato", "Start", "Good"),
        ("Slice a potato", "Stop", "Good"),
        ("Peel a potato", "Start", "Good"),  # never stopped
        ("Pour water", "Start", "Bad"),
        ("Pour water", "Stop", "Bad"),
        ("Get items", "Start", "Good"),
        ("Get items", "Stop", "Good"),
    ]
    segs, counts = activity_segments(rows, np.arange(7.0))
    assert segs == [(0.0, 1.0, 0)]
    assert counts["unclosed"] == 1 and counts["bad"] == 1
    assert counts["unmapped:Get items"] == 1


def test_frame_grid_10hz() -> None:
    starts, labels = frame_grid([(0.0, 0.35, 2)])
    np.testing.assert_allclose(starts, [0.0, 0.1, 0.2])  # last full window ends at 0.3
    assert labels.tolist() == [2, 2, 2]


def test_window_mean_interpolates_short_gap_and_drops_long() -> None:
    t = np.array([0.0, 0.05, 0.25, 0.9])
    x = np.array([[0.0], [1.0], [3.0], [9.0]])
    m, valid = window_mean(t, x, np.array([0.0, 0.1, 0.5]))
    assert valid.tolist() == [True, True, False]
    assert m[0, 0] == 0.5 and 1.0 < m[1, 0] < 3.0


def test_pool_tactile_and_emg_sectors_shapes() -> None:
    assert pool_tactile(np.ones((5, 32, 32))).shape == (5, 16)
    out = emg_sectors(np.arange(8.0)[None].repeat(2, 0))
    np.testing.assert_allclose(out[0], [1.0, 4.0, 6.5])


def test_zscore_constant_channel_is_zero() -> None:
    z, _mu, _sd = zscore(np.array([[1.0, 5.0], [3.0, 5.0]]))
    assert np.isfinite(z).all() and (z[:, 1] == 0).all()


def test_check_aligned_names_mismatch() -> None:
    with pytest.raises(ValueError, match=r"vision=120"):
        check_aligned({"audio": np.zeros((1800, 128)), "vision": np.zeros((120, 16, 16))})
