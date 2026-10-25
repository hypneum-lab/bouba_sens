"""Common Phase-2 schedule + probe sharing per (seed, lock) (amendment v0.8)."""

from __future__ import annotations

import pytest
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
from bouba_sens.loop import AdaptationLoop
from bouba_sens.nerve import CrossModalNerve
from bouba_sens.sensory import SensoryWML
from bouba_sens.subspace_protocol import (
    PROBE_SIZE,
    SNR_FLOORS,
    TRAIN_BATCH_SIZE,
    BlockCodes,
    capture_block,
)
from bouba_sens.world.gaussian import GaussianWorld

# Bit-identical comparisons need deterministic CPU kernels (tests/conftest.py).
pytestmark = pytest.mark.usefixtures("deterministic_torch")


def _make_loop() -> AdaptationLoop:
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


def _block(ckpt=None, **over) -> BlockCodes:  # type: ignore[no-untyped-def]
    if ckpt is None:
        ckpt = _make_loop().pretrain(steps=3, batch_size=4, seed=0)
    kw = {
        "seed": 2001,
        "lock": 25,
        "modalities": ("audio", "vision"),
        "snr_floors": (-20.0, 10.0),
        "steps": 3,
        "batch_size": 4,
        "probe_size": 16,
    }
    kw.update(over)
    return capture_block(_make_loop, ckpt, **kw)  # type: ignore[arg-type]


def test_frozen_protocol_constants() -> None:
    assert PROBE_SIZE == 512
    assert TRAIN_BATCH_SIZE == 16
    assert SNR_FLOORS == (-20.0, -10.0, 10.0)


def test_block_has_control_and_all_cells() -> None:
    b = _block()
    assert (b.seed, b.lock) == (2001, 25)
    assert set(b.z_post) == {
        ("audio", -20.0),
        ("audio", 10.0),
        ("vision", -20.0),
        ("vision", 10.0),
    }
    for z in (b.z_pre, b.z_control, *b.z_post.values()):
        assert z.shape[0] == 16
        assert z.shape[1] == b.z_pre.shape[1]


def test_all_arms_share_probe_labels_and_z_pre() -> None:
    b = _block()
    assert b.probe_labels.shape == (16,)
    # The probe is the documented draw world.sample(n, seed=phase2_seed - 99_999).
    expected = GaussianWorld(seed=0).sample(batch_size=16, seed=2001 - 99_999).label
    assert torch.equal(b.probe_labels, expected)


def test_block_is_reproducible(deterministic_torch: None) -> None:
    ckpt = _make_loop().pretrain(steps=3, batch_size=4, seed=0)
    a = _block(ckpt)
    b = _block(ckpt)
    assert torch.equal(a.z_pre, b.z_pre)
    assert torch.equal(a.z_control, b.z_control)
    for key in a.z_post:
        assert torch.equal(a.z_post[key], b.z_post[key])


def test_lesion_cells_differ_from_control() -> None:
    b = _block()
    assert not torch.equal(b.z_post[("audio", -20.0)], b.z_control)


def test_phase2_seed_defaults_to_experiment_seed_and_is_overridable() -> None:
    a = _block(phase2_seed=555)
    expected = GaussianWorld(seed=0).sample(batch_size=16, seed=555 - 99_999).label
    assert torch.equal(a.probe_labels, expected)
    assert a.phase2_seed == 555


# --- PI decision 2026-10-25: Gumbel probe readout -------------------------


def test_probe_readout_is_gumbel_and_block_reproducible() -> None:
    from bouba_sens.subspace_protocol import PROBE_DEMOD

    assert PROBE_DEMOD == "soft"
    ckpt = _make_loop().pretrain(steps=3, batch_size=4, seed=0)
    a, b = _block(ckpt), _block(ckpt)
    assert torch.equal(a.z_pre, b.z_pre) and torch.equal(a.z_control, b.z_control)
    for key in a.z_post:
        assert torch.equal(a.z_post[key], b.z_post[key])
    hard = _block(ckpt, probe_demod="hard")
    assert not torch.equal(hard.z_pre, a.z_pre)
