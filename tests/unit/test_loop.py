"""Task 2.9-2.10 acceptance tests for `AdaptationLoop`."""

from __future__ import annotations

import torch
from track_p.multiplexer import GammaThetaMultiplexer

from bouba_sens.encoders import (
    AudioEncoder,
    ForceEncoder,
    GravityEncoder,
    TactileEncoder,
    VisionEncoder,
)
from bouba_sens.head import IntegrationHead
from bouba_sens.lesion import LesionSpec, m2_snr_schedule
from bouba_sens.loop import AdaptationLoop
from bouba_sens.nerve import CrossModalNerve
from bouba_sens.sensory import SensoryWML
from bouba_sens.world.gaussian import GaussianWorld


def _build_loop(*, n_classes: int = 4) -> AdaptationLoop:
    world = GaussianWorld(seed=0)
    mux = GammaThetaMultiplexer(seed=0)
    sensories = {
        "audio": SensoryWML(id=0, modality="audio", input_proj=AudioEncoder(), mux=mux, seed=1),
        "vision": SensoryWML(id=1, modality="vision", input_proj=VisionEncoder(), mux=mux, seed=2),
        "tactile": SensoryWML(
            id=2, modality="tactile", input_proj=TactileEncoder(), mux=mux, seed=3
        ),
        "gravity": SensoryWML(
            id=3, modality="gravity", input_proj=GravityEncoder(), mux=mux, seed=4
        ),
        "force": SensoryWML(id=4, modality="force", input_proj=ForceEncoder(), mux=mux, seed=5),
    }
    nerve = CrossModalNerve(mux, seed=0)
    head = IntegrationHead(n_classes=n_classes)
    return AdaptationLoop(world, mux, sensories, nerve, head, lr=1e-3)


def test_pretrain_10_steps_no_nan() -> None:
    loop = _build_loop()
    losses: list[float] = []
    for step in range(10):
        sample = loop.world.sample(batch_size=8, seed=step)
        loss, _ = loop._forward(sample)
        losses.append(loss.item())
        loop.opt.zero_grad()
        loss.backward()
        loop.opt.step()
    assert all(loss_val == loss_val for loss_val in losses)  # no NaN


def test_pretrain_returns_checkpoint_with_all_states() -> None:
    loop = _build_loop()
    ckpt = loop.pretrain(steps=5, batch_size=4, seed=0)
    assert ckpt.mux_state
    assert ckpt.nerve_state
    assert ckpt.head_state
    assert set(ckpt.sensory_states.keys()) == {
        "audio",
        "vision",
        "tactile",
        "gravity",
        "force",
    }


def test_pretrain_checkpoint_roundtrip() -> None:
    """save -> mutate -> restore must reset parameters exactly."""
    loop = _build_loop()
    ckpt = loop.snapshot()
    # Mutate one param.
    with torch.no_grad():
        loop.head.classifier.weight.add_(1.0)
    weight_mutated = loop.head.classifier.weight.clone()
    loop.restore(ckpt)
    assert not torch.equal(loop.head.classifier.weight, weight_mutated)
    assert torch.allclose(
        loop.head.classifier.weight,
        ckpt.head_state["classifier.weight"],
    )


def test_lesion_phase_10_steps_no_nan() -> None:
    loop = _build_loop()
    spec = LesionSpec(
        modality="audio",
        mode="M2",
        timing="T2",
        schedule=m2_snr_schedule,
    )
    report = loop.lesion_phase(spec, steps=10, batch_size=4, stats_every=5)
    assert len(report.loss_curve) == 10
    assert all(loss == loss for loss in report.loss_curve)  # no NaN


def test_lesion_phase_populates_migration_report() -> None:
    loop = _build_loop()
    spec = LesionSpec(
        modality="vision",
        mode="M2",
        timing="T2",
        schedule=m2_snr_schedule,
    )
    report = loop.lesion_phase(spec, steps=20, batch_size=4, stats_every=5)
    assert len(report.gate_trajectory) >= 2
    assert len(report.codebook_entropy_trajectory) >= 2
    assert len(report.lesion_events) == 1
    # on_lesion was called once at t=0.
    assert report.lesion_events[0][0] == "vision"


def test_lesion_phase_probe_batch_size_default_is_128() -> None:
    """ADR-0006 Sprint 8 fix — probe batch decouples from training batch."""
    loop = _build_loop()
    spec = LesionSpec(modality="audio", mode="M2", timing="T2", schedule=m2_snr_schedule)
    # batch_size=4 (small training), probes should still be 128 by default
    report = loop.lesion_phase(spec, steps=5, batch_size=4, stats_every=5)
    assert report.pre_lesion_codes is not None
    assert report.post_lesion_codes is not None
    assert report.pre_lesion_codes.shape[0] == 128
    assert report.post_lesion_codes.shape[0] == 128
    assert report.probe_labels is not None
    assert report.probe_labels.shape[0] == 128


def test_lesion_phase_probe_batch_size_override() -> None:
    """probe_batch_size parameter overrides the default 128."""
    loop = _build_loop()
    spec = LesionSpec(modality="tactile", mode="M2", timing="T2", schedule=m2_snr_schedule)
    report = loop.lesion_phase(spec, steps=5, batch_size=4, probe_batch_size=64, stats_every=5)
    assert report.pre_lesion_codes is not None
    assert report.pre_lesion_codes.shape[0] == 64
