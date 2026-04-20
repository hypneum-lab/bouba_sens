"""Task 2.4 acceptance tests for `bouba_sens.nerve.CrossModalNerve.fuse`."""

from __future__ import annotations

import torch
from track_p.multiplexer import GammaThetaMultiplexer  # type: ignore[import-not-found]

from bouba_sens.nerve import CrossModalNerve
from bouba_sens.sensory import MODALITIES


def _make_carriers(mux: GammaThetaMultiplexer, batch: int = 2) -> dict[str, torch.Tensor]:
    """Produce realistic (B, T) carriers from random codes via mux.forward."""
    out: dict[str, torch.Tensor] = {}
    for i, m in enumerate(MODALITIES):
        codes = torch.randint(0, mux.cfg.alphabet_size, (batch, mux.cfg.symbols_per_theta))
        # Offset by modality index so carriers differ.
        out[m] = mux.forward(codes) + 0.01 * (i + 1)
    return out


def test_fuse_shape() -> None:
    """5 carriers (B, T) -> fused (B, K, d_hidden)."""
    mux = GammaThetaMultiplexer(seed=0)
    nerve = CrossModalNerve(mux, seed=0)
    carriers = _make_carriers(mux, batch=2)
    fused = nerve.fuse(carriers)
    assert fused.shape == (2, mux.cfg.symbols_per_theta, 128)


def test_fuse_respects_gate_weights() -> None:
    """Biasing the gate toward one modality changes the fused output."""
    mux = GammaThetaMultiplexer(seed=0)
    nerve = CrossModalNerve(mux, seed=0)
    carriers = _make_carriers(mux, batch=2)

    # Baseline (uniform gates).
    fused_uniform = nerve.fuse(carriers).detach().clone()

    # Bias alpha strongly toward "audio".
    with torch.no_grad():
        nerve.gates.alpha.copy_(torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0]))
    fused_audio = nerve.fuse(carriers).detach().clone()

    assert not torch.allclose(fused_uniform, fused_audio, atol=1e-3)


def test_fuse_end_to_end_differentiable() -> None:
    """Gradient reaches mux constellation, codebook, gates, AND at least
    one active transducer when the gate activation policy fires."""
    mux = GammaThetaMultiplexer(seed=0)
    nerve = CrossModalNerve(mux, seed=0)
    carriers = _make_carriers(mux, batch=2)

    # Bias gates so that at least one transducer is active:
    # make audio's gate very low (< 0.1) AND vision's gate very high (> 0.3).
    with torch.no_grad():
        # audio idx=0, vision idx=1, set audio low, vision high, rest middle.
        nerve.gates.alpha.copy_(torch.tensor([-5.0, 5.0, 0.0, 0.0, 0.0]))

    # Confirm the "audio->vision" transducer is active under this gate.
    w = nerve.gates.weights()
    assert w[0].item() < 0.1
    assert w[1].item() > 0.3

    fused = nerve.fuse(carriers)
    fused.pow(2).mean().backward()

    # 1. mux constellation gets gradient (via demodulate soft path).
    assert mux.constellation.grad is not None
    assert mux.constellation.grad.abs().sum().item() > 0.0

    # 2. Codebook param gets gradient.
    assert nerve.codebook.codebook.grad is not None
    assert nerve.codebook.codebook.grad.abs().sum().item() > 0.0

    # 3. Gate alpha gets gradient.
    assert nerve.gates.alpha.grad is not None
    assert nerve.gates.alpha.grad.abs().sum().item() > 0.0

    # 4. Active transducer "audio->vision" gets gradient on its fc1.weight.
    active_t = nerve.transducers["audio->vision"]
    assert active_t.fc1.weight.grad is not None
    assert active_t.fc1.weight.grad.abs().sum().item() > 0.0


def test_fuse_has_20_transducers() -> None:
    mux = GammaThetaMultiplexer(seed=0)
    nerve = CrossModalNerve(mux, seed=0)
    # 5 modalities x 5 modalities = 25, minus 5 self-loops = 20.
    assert len(nerve.transducers) == 20


def test_on_lesion_appends_to_log() -> None:
    mux = GammaThetaMultiplexer(seed=0)
    nerve = CrossModalNerve(mux, seed=0)
    assert nerve._lesion_log == []
    nerve.on_lesion("audio", -20.0)
    assert nerve._lesion_log == [("audio", -20.0)]
