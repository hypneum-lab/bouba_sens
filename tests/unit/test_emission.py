"""ADR-0024 follow-up — SensoryWML emission option (straight-through, opt-in)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

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
from bouba_sens.sensory import DEFAULT_EMISSION, Modality, SensoryWML
from bouba_sens.world.gaussian import GaussianWorld


def _loop(emission: str, seed: int = 0) -> AdaptationLoop:
    torch.manual_seed(seed)
    mux = GammaThetaMultiplexer(seed=seed)
    sens: dict[Modality, SensoryWML] = {
        "audio": SensoryWML(0, "audio", AudioEncoder(), mux, seed=seed + 1, emission=emission),
        "vision": SensoryWML(1, "vision", VisionEncoder(), mux, seed=seed + 2, emission=emission),
        "tactile": SensoryWML(
            2, "tactile", TactileEncoder(), mux, seed=seed + 3, emission=emission
        ),
        "gravity": SensoryWML(
            3, "gravity", GravityEncoder(), mux, seed=seed + 4, emission=emission
        ),
        "force": SensoryWML(4, "force", ForceEncoder(), mux, seed=seed + 5, emission=emission),
    }
    return AdaptationLoop(
        GaussianWorld(seed=seed),
        mux,
        sens,
        CrossModalNerve(mux, seed=seed),
        IntegrationHead(n_classes=4),
    )


def _legacy_step(s: SensoryWML, x: torch.Tensor) -> torch.Tensor:
    """The pre-option body of SensoryWML.step, kept for non-regression."""
    h = s.core(s.input_proj(x))
    pi_logits = s.emit_head_pi(h)
    base_code = pi_logits.argmax(-1)
    k = s.mux.cfg.symbols_per_theta
    carrier = s.mux.forward(base_code.unsqueeze(-1).expand(-1, k))
    soft_iq = pi_logits.softmax(-1) @ s.mux.constellation
    return carrier + soft_iq.mean(-1, keepdim=True).expand_as(carrier) * 1e-3


def test_default_emission_is_the_historical_rule() -> None:
    assert DEFAULT_EMISSION == "argmax-bridge"
    loop = _loop(DEFAULT_EMISSION)
    sample = loop.world.sample(batch_size=32, seed=3)
    for m, s in loop.sensories.items():
        assert s.emission == "argmax-bridge"
        assert torch.equal(s.step(getattr(sample, m)), _legacy_step(s, getattr(sample, m)))


def test_default_construction_unchanged_by_the_option() -> None:
    torch.manual_seed(0)
    mux_a = GammaThetaMultiplexer(seed=0)
    a = SensoryWML(0, "audio", AudioEncoder(), mux_a, seed=1)
    torch.manual_seed(0)
    mux_b = GammaThetaMultiplexer(seed=0)
    b = SensoryWML(0, "audio", AudioEncoder(), mux_b, seed=1, emission="straight-through")
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert ka == kb and torch.equal(va, vb)


def test_invalid_emission_raises() -> None:
    with pytest.raises(ValueError, match="emission"):
        SensoryWML(0, "audio", AudioEncoder(), GammaThetaMultiplexer(seed=0), seed=1, emission="x")


def test_straight_through_forward_is_the_argmax_carrier() -> None:
    loop = _loop("straight-through")
    sample = loop.world.sample(batch_size=16, seed=5)
    for m, s in loop.sensories.items():
        x = getattr(sample, m)
        with torch.no_grad():
            codes = s.emit_head_pi(s.core(s.input_proj(x))).argmax(-1)
            carrier = s.step(x)
            ref = s.mux.forward(codes.unsqueeze(-1).expand(-1, s.mux.cfg.symbols_per_theta))
        assert torch.allclose(carrier, ref, atol=1e-6)
        assert torch.equal(s.mux.demodulate(carrier, hard=True)[:, 0], codes)


def test_straight_through_carries_the_task_gradient() -> None:
    """The emission head gets a gradient orders of magnitude larger than the bridge."""
    norms = {}
    for emission in ("argmax-bridge", "straight-through"):
        loop = _loop(emission)
        sample = loop.world.sample(batch_size=64, seed=9)
        torch.manual_seed(1)
        loss, _ = loop._forward(sample)
        loop.opt.zero_grad()
        loss.backward()
        norms[emission] = float(loop.sensories["vision"].emit_head_pi.weight.grad.norm())
    assert norms["straight-through"] > 100 * norms["argmax-bridge"]


def test_straight_through_learns_on_gaussian_where_default_does_not() -> None:
    losses = {}
    for emission in ("argmax-bridge", "straight-through"):
        loop = _loop(emission, seed=0)
        curve: list[float] = []
        loop.pretrain(steps=120, batch_size=16, seed=0, loss_log=curve)
        assert len(curve) == 120
        losses[emission] = sum(curve[-20:]) / 20
    assert losses["argmax-bridge"] > 1.35
    assert losses["straight-through"] < 1.30


def test_cli_records_emission_and_losses(tmp_path: Path) -> None:
    def run(args: list[str]) -> None:
        subprocess.run(["uv", "run", "bouba-sens", *args], check=True, capture_output=True)

    train_dir, cell = tmp_path / "p1", tmp_path / "cell"
    run(["train", "--steps", "3", "--batch-size", "4", "--seed", "0", "--no-probe-v2",
         "--emission", "straight-through", "--out", str(train_dir)])  # fmt: skip
    run(["lesion", "--ckpt", str(train_dir), "--timing", "T2", "--steps", "2", "--seed", "1",
         "--no-probe-v2", "--emission", "straight-through", "--out", str(cell)])  # fmt: skip
    run(["eval", "--run", str(cell), "--out", str(cell / "eval_report.json")])
    assert len(json.loads((train_dir / "pretrain_losses.json").read_text())) == 3
    assert "emission: straight-through" in (train_dir / "config.yaml").read_text()
    assert "emission: straight-through" in (cell / "config.yaml").read_text()
    assert json.loads((cell / "eval_report.json").read_text())["me1_probe"] is not None
    # Default runs keep their historical config.yaml (no emission key).
    run(["train", "--steps", "2", "--batch-size", "4", "--seed", "0", "--no-probe-v2",
         "--out", str(tmp_path / "d")])  # fmt: skip
    assert "emission" not in (tmp_path / "d" / "config.yaml").read_text()
