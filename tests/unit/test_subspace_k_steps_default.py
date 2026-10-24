"""v0.8 protocol default: the SNR ramp of ``capture_block`` is k_steps = 50.

Amendment v0.8 section 4 freezes ``k_steps`` = 50 (ADR-0020; PI decision
2026-10-23) so that the nominal floors -20 / -10 / +10 dB are reached at
lesion step 50. ``capture_block`` is new v0.8 code with no historical
runs, so its default is the frozen protocol value (historical CLI and
``m2_snr_schedule`` defaults are unchanged).
"""

from __future__ import annotations

import inspect

from bouba_sens.lesion import m2_snr_schedule
from bouba_sens.subspace_protocol import K_STEPS, capture_block


def test_capture_block_default_k_steps_is_frozen_value() -> None:
    assert K_STEPS == 50
    assert inspect.signature(capture_block).parameters["k_steps"].default == K_STEPS


def test_frozen_ramp_reaches_each_floor_at_step_50() -> None:
    for floor in (-20.0, -10.0, 10.0):
        assert m2_snr_schedule(49, snr_init=20.0, snr_floor=floor, k=K_STEPS) > floor
        for step in (50, 99):
            assert m2_snr_schedule(step, snr_init=20.0, snr_floor=floor, k=K_STEPS) == floor
