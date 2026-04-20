"""Task 2.6 acceptance tests for `LesionSpec` + `m2_snr_schedule`."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from bouba_sens.lesion import LesionSpec, m2_snr_schedule


def test_lesionspec_is_frozen() -> None:
    spec = LesionSpec(
        modality="audio",
        mode="M2",
        timing="T2",
        schedule=m2_snr_schedule,
    )
    with pytest.raises(FrozenInstanceError):
        spec.modality = "vision"  # type: ignore[misc]


def test_m2_starts_at_snr_init() -> None:
    """At step=0 the schedule must sit at SNR_init."""
    assert abs(m2_snr_schedule(0) - 20.0) < 1e-9
    assert abs(m2_snr_schedule(0, snr_init=15.0) - 15.0) < 1e-9


def test_m2_converges_to_floor() -> None:
    """By step=K the schedule must clamp at SNR_floor."""
    assert abs(m2_snr_schedule(5000) - (-20.0)) < 1e-9
    # And past K it stays at SNR_floor.
    assert abs(m2_snr_schedule(50000) - (-20.0)) < 1e-9


def test_m2_linear_interpolation_midpoint() -> None:
    """At step=K/2 the schedule sits at the midpoint of init+floor."""
    mid = m2_snr_schedule(2500)
    assert abs(mid - 0.0) < 1e-9  # (20 + -20) / 2


def test_m2_honours_custom_params() -> None:
    """snr_init / snr_floor / k kwargs must be respected."""
    assert abs(m2_snr_schedule(0, snr_init=10.0, snr_floor=-10.0, k=1000) - 10.0) < 1e-9
    assert abs(m2_snr_schedule(1000, snr_init=10.0, snr_floor=-10.0, k=1000) - (-10.0)) < 1e-9
    assert abs(m2_snr_schedule(500, snr_init=10.0, snr_floor=-10.0, k=1000) - 0.0) < 1e-9


def test_lesionspec_stores_schedule_callable() -> None:
    spec = LesionSpec(
        modality="audio",
        mode="M2",
        timing="T2",
        schedule=lambda t: 10.0,
    )
    assert spec.schedule(0) == 10.0
