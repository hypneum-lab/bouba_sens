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
    monotonic_violations,
    pool_tactile,
    window_mean,
    window_rms,
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


# The 20 activity strings of the ActionNet recording code, with the family each
# must map to (spec section 6). Source: delpreto/ActionNet,
# recording_data/sensor_streamers/ExperimentControlStreamer.py, ``self._activities``.
UPSTREAM_LABELS: tuple[tuple[str, int | None], ...] = (
    ("Get/replace items from refrigerator/cabinets/drawers", None),
    ("Clear cutting board", None),
    ("Peel a cucumber", 0),
    ("Slice a cucumber", 0),
    ("Peel a potato", 0),
    ("Slice a potato", 0),
    ("Slice bread", 0),
    ("Spread almond butter on a bread slice", 1),
    ("Spread jelly on a bread slice", 1),
    ("Open/close a jar of almond butter", 1),
    ("Pour water from a pitcher into a glass", 3),
    ("Clean a plate with a sponge", 2),
    ("Clean a plate with a towel", 2),
    ("Clean a pan with a sponge", 2),
    ("Clean a pan with a towel", 2),
    (
        "Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, "
        "sets of utensils",
        None,
    ),
    ("Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils", 3),
    ("Stack on table: 3 each large/small plates, bowls", None),
    ("Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils", 3),
    ("Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils", 3),
)


def test_label_family_all_upstream_labels() -> None:
    assert len(UPSTREAM_LABELS) == 20
    for label, expected in UPSTREAM_LABELS:
        assert label_family(label) == expected, label


def test_window_mean_masks_nonfinite_samples() -> None:
    t = np.arange(0.0, 1.0, 0.02)
    x = np.ones((len(t), 1))
    x[3, 0] = np.nan
    x[30, 0] = np.inf
    counts: dict[str, int] = {}
    m, valid = window_mean(t, x, np.array([0.0, 0.5, 0.8]), counts=counts)
    assert counts["nonfinite_samples"] == 2
    assert np.isfinite(m).all() and valid.all()
    np.testing.assert_allclose(m, 1.0)


def test_window_rms_masks_nonfinite_samples() -> None:
    t = np.arange(0.0, 1.0, 0.005)
    x = np.full((len(t), 2), 2.0)
    x[10] = np.nan
    counts: dict[str, int] = {}
    r, valid = window_rms(t, x, np.array([0.0, 0.5]), counts=counts)
    assert counts["nonfinite_samples"] == 1 and valid.all()
    np.testing.assert_allclose(r, 2.0)


def test_monotonic_violations() -> None:
    assert monotonic_violations(np.array([0.0, 1.0, 1.0, 2.0])) == 0
    assert monotonic_violations(np.array([0.0, 2.0, 1.0, 3.0, 2.5])) == 2
