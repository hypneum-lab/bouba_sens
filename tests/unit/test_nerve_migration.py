"""Task 2.5 acceptance tests for on_lesion + migration_stats."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from track_p.multiplexer import GammaThetaMultiplexer  # type: ignore[import-not-found]

from bouba_sens.nerve import CrossModalNerve, MigrationReport
from bouba_sens.sensory import MODALITIES


def test_migration_report_is_frozen_dataclass() -> None:
    report = MigrationReport(
        gate_values={m: 0.2 for m in MODALITIES},
        codebook_entropy=4.1,
        transducer_active_count=0,
        lesion_events=(),
    )
    with pytest.raises(FrozenInstanceError):
        report.transducer_active_count = 1  # type: ignore[misc]


def test_migration_stats_shape_at_init() -> None:
    """At init, all gates uniform, no lesion events, no active transducers."""
    mux = GammaThetaMultiplexer(seed=0)
    nerve = CrossModalNerve(mux, seed=0)
    report = nerve.migration_stats()
    assert set(report.gate_values.keys()) == set(MODALITIES)
    for m in MODALITIES:
        assert abs(report.gate_values[m] - 0.2) < 1e-6
    assert report.transducer_active_count == 0
    assert report.lesion_events == ()
    # Entropy of near-uniform softmax over 64 elements ~ log(64) ~ 4.16 nats.
    assert 3.5 < report.codebook_entropy < 4.5


def test_on_lesion_appends_to_log_and_biases_gate() -> None:
    mux = GammaThetaMultiplexer(seed=0)
    nerve = CrossModalNerve(mux, seed=0)
    alpha_audio_before = nerve.gates.alpha[0].item()
    nerve.on_lesion("audio", -20.0)
    alpha_audio_after = nerve.gates.alpha[0].item()
    # Alpha for audio must have decreased (default bias = -1.0).
    assert alpha_audio_after < alpha_audio_before - 0.5
    # Lesion event logged.
    report = nerve.migration_stats()
    assert report.lesion_events == (("audio", -20.0),)


def test_on_lesion_bias_zero_leaves_alpha_unchanged() -> None:
    mux = GammaThetaMultiplexer(seed=0)
    nerve = CrossModalNerve(mux, seed=0)
    alpha_before = nerve.gates.alpha.clone()
    nerve.on_lesion("vision", -15.0, bias=0.0)
    assert torch.equal(nerve.gates.alpha, alpha_before)


def test_migration_stats_reflects_post_lesion_shift() -> None:
    mux = GammaThetaMultiplexer(seed=0)
    nerve = CrossModalNerve(mux, seed=0)
    report_before = nerve.migration_stats()
    nerve.on_lesion("audio", -20.0)
    report_after = nerve.migration_stats()
    # Audio gate dropped; the other 4 gates shared the lost mass.
    assert report_after.gate_values["audio"] < report_before.gate_values["audio"]
    for m in ("vision", "tactile", "gravity", "force"):
        assert report_after.gate_values[m] > report_before.gate_values[m]
