"""Probe decoupling + no-lesion control arm (amendment v0.8, section 3)."""

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


def _build_loop() -> AdaptationLoop:
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
    return AdaptationLoop(world, mux, sensories, nerve, IntegrationHead(n_classes=4), lr=1e-3)


def _restored_loop(ckpt) -> AdaptationLoop:  # type: ignore[no-untyped-def]
    """Fresh loop restored from a Phase-1 checkpoint (encoders init from global RNG)."""
    loop = _build_loop()
    loop.restore(ckpt)
    return loop


def _spec(modality: str = "audio") -> LesionSpec:
    return LesionSpec(modality=modality, mode="M2", timing="T2", schedule=m2_snr_schedule)  # type: ignore[arg-type]


def test_probe_size_default_is_batch_size() -> None:
    report = _build_loop().lesion_phase(_spec(), steps=2, batch_size=4)
    assert report.pre_lesion_codes is not None
    assert report.pre_lesion_codes.shape[0] == 4


def test_probe_size_decoupled_from_batch_size() -> None:
    report = _build_loop().lesion_phase(_spec(), steps=3, batch_size=4, probe_size=64)
    assert report.pre_lesion_codes is not None
    assert report.post_lesion_codes is not None
    assert report.probe_labels is not None
    assert report.pre_lesion_codes.shape[0] == 64
    assert report.post_lesion_codes.shape == report.pre_lesion_codes.shape
    assert report.probe_labels.shape == (64,)
    assert len(report.loss_curve) == 3  # training batch unchanged


def test_probe_is_world_sample_at_seed_minus_99999() -> None:
    loop = _build_loop()
    report = loop.lesion_phase(_spec(), steps=1, batch_size=4, seed=777, probe_size=32)
    expected = loop.world.sample(batch_size=32, seed=777 - 99_999)
    assert report.probe_labels is not None
    assert torch.equal(report.probe_labels, expected.label)


def test_probe_seed_override() -> None:
    loop = _build_loop()
    report = loop.lesion_phase(
        _spec(), steps=1, batch_size=4, seed=777, probe_size=32, probe_seed=123
    )
    assert report.probe_labels is not None
    assert torch.equal(report.probe_labels, loop.world.sample(batch_size=32, seed=123).label)


def test_control_arm_has_no_lesion_event() -> None:
    report = _build_loop().lesion_phase(None, steps=3, batch_size=4, probe_size=16)
    assert report.lesion_events == []
    assert len(report.loss_curve) == 3
    assert report.pre_lesion_codes is not None
    assert report.post_lesion_codes is not None
    assert report.post_lesion_codes.shape == report.pre_lesion_codes.shape


def test_control_shares_probe_and_z_pre_with_lesion_arm() -> None:
    ckpt = _build_loop().pretrain(steps=3, batch_size=4, seed=0)
    ctrl = _restored_loop(ckpt).lesion_phase(
        None, steps=3, batch_size=4, seed=50, probe_size=16, probe_demod="hard"
    )
    les = _restored_loop(ckpt).lesion_phase(
        _spec(), steps=3, batch_size=4, seed=50, probe_size=16, probe_demod="hard"
    )
    assert ctrl.probe_labels is not None and les.probe_labels is not None
    assert ctrl.pre_lesion_codes is not None and les.pre_lesion_codes is not None
    assert torch.equal(ctrl.probe_labels, les.probe_labels)
    assert torch.equal(ctrl.pre_lesion_codes, les.pre_lesion_codes)


def test_control_post_differs_from_lesion_post() -> None:
    ckpt = _build_loop().pretrain(steps=3, batch_size=4, seed=0)
    ctrl = _restored_loop(ckpt).lesion_phase(
        None, steps=5, batch_size=4, seed=50, probe_size=16, probe_demod="hard"
    )
    les = _restored_loop(ckpt).lesion_phase(
        _spec(), steps=5, batch_size=4, seed=50, probe_size=16, probe_demod="hard"
    )
    assert ctrl.post_lesion_codes is not None and les.post_lesion_codes is not None
    assert not torch.equal(ctrl.post_lesion_codes, les.post_lesion_codes)


def test_hard_probe_is_bit_identical_without_any_seeding() -> None:
    """Hard (argmax) demodulation: same checkpoint -> same codes, no RNG involved."""
    ckpt = _build_loop().pretrain(steps=3, batch_size=4, seed=0)
    kw = {"batch_size": 4, "seed": 50, "probe_size": 16, "probe_demod": "hard"}
    torch.manual_seed(1)
    a = _restored_loop(ckpt).lesion_phase(None, steps=0, **kw)
    torch.manual_seed(98765)  # unrelated global RNG state must not matter
    b = _restored_loop(ckpt).lesion_phase(None, steps=0, **kw)
    assert a.pre_lesion_codes is not None and b.pre_lesion_codes is not None
    assert torch.equal(a.pre_lesion_codes, b.pre_lesion_codes)
    # Untrained control (0 steps): same net -> post == pre exactly.
    assert torch.equal(a.pre_lesion_codes, a.post_lesion_codes)  # type: ignore[arg-type]


def test_hard_probe_does_not_consume_global_rng() -> None:
    ckpt = _build_loop().pretrain(steps=3, batch_size=4, seed=0)
    loop = _restored_loop(ckpt)
    torch.manual_seed(7)
    state = torch.get_rng_state()
    loop.lesion_phase(None, steps=0, batch_size=4, probe_size=8, probe_demod="hard")
    assert torch.equal(torch.get_rng_state(), state)


def test_soft_probe_remains_the_stochastic_legacy_default() -> None:
    ckpt = _build_loop().pretrain(steps=3, batch_size=4, seed=0)
    a = _restored_loop(ckpt).lesion_phase(None, steps=0, batch_size=4, probe_size=8)
    b = _restored_loop(ckpt).lesion_phase(None, steps=0, batch_size=4, probe_size=8)
    assert a.pre_lesion_codes is not None and b.pre_lesion_codes is not None
    assert not torch.equal(a.pre_lesion_codes, b.pre_lesion_codes)


def test_unknown_probe_demod_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="probe_demod"):
        _build_loop().lesion_phase(None, steps=0, batch_size=4, probe_demod="nope")
