"""Sprint 14 — transducer_gating_schedule flip test."""

from __future__ import annotations

import pytest
from track_p.multiplexer import GammaThetaMultiplexer

from bouba_sens.nerve import CrossModalNerve


def test_schedule_flips_after_threshold() -> None:
    mux = GammaThetaMultiplexer(seed=0)
    nerve = CrossModalNerve(
        mux,
        seed=0,
        transducer_gating="hard",
        transducer_gating_schedule=3,
        transducer_gating_target="gumbel",
    )
    assert nerve._active_gating() == "hard"
    nerve.step()
    nerve.step()
    assert nerve._active_gating() == "hard"
    nerve.step()  # codebook_step crosses 3
    assert nerve._active_gating() == "gumbel"


def test_schedule_resumes_after_state_restore() -> None:
    mux = GammaThetaMultiplexer(seed=0)
    nerve = CrossModalNerve(
        mux,
        seed=0,
        transducer_gating="hard",
        transducer_gating_schedule=2,
        transducer_gating_target="gumbel",
    )
    for _ in range(5):
        nerve.step()
    sd = {k: v.detach().clone() for k, v in nerve.state_dict().items()}

    fresh = CrossModalNerve(
        mux,
        seed=0,
        transducer_gating="hard",
        transducer_gating_schedule=2,
        transducer_gating_target="gumbel",
    )
    fresh.load_state_dict(sd)
    assert fresh._active_gating() == "gumbel"


def test_missing_target_raises() -> None:
    mux = GammaThetaMultiplexer(seed=0)
    with pytest.raises(ValueError, match="transducer_gating_target"):
        CrossModalNerve(mux, transducer_gating_schedule=5)


def test_no_schedule_preserves_legacy_mode() -> None:
    mux = GammaThetaMultiplexer(seed=0)
    nerve = CrossModalNerve(mux, transducer_gating="gumbel")
    for _ in range(10):
        nerve.step()
    assert nerve._active_gating() == "gumbel"
