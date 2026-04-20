"""Task 1.9 — Sprint 1 integration smoke test.

Five SensoryWMLs (one per modality) sharing a single
`GammaThetaMultiplexer` must each emit a valid `(B, T)` carrier
that round-trips to `(B, K)` long indices in `[0, alphabet_size)`.

This is the acceptance gate for Sprint 1: the left half of the
spec §2.1 block diagram (WorldSimulator -> SensoryWMLs -> carriers)
is wired end-to-end, with no mocking of the shared multiplexer.
"""

from __future__ import annotations

import torch
from track_p.multiplexer import GammaThetaMultiplexer  # type: ignore[import-not-found]

from bouba_sens.encoders import (
    AudioEncoder,
    ForceEncoder,
    GravityEncoder,
    TactileEncoder,
    VisionEncoder,
)
from bouba_sens.sensory import SensoryWML
from bouba_sens.world.gaussian import GaussianWorld


def test_five_modality_carriers_roundtrip() -> None:
    """5 SensoryWMLs on a shared mux produce valid carriers + codes."""
    batch = 8
    mux = GammaThetaMultiplexer(seed=0)

    world = GaussianWorld(seed=0)
    sample = world.sample(batch_size=batch, seed=42)

    wmls = {
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

    t_expected = mux._t_grid.numel()
    k_expected = mux.cfg.symbols_per_theta
    alphabet = mux.cfg.alphabet_size

    for modality, wml in wmls.items():
        x = getattr(sample, modality)
        carrier = wml.step(x)

        assert carrier.shape == (batch, t_expected), (
            f"{modality}: carrier shape {tuple(carrier.shape)} != ({batch}, {t_expected})"
        )

        recovered = mux.demodulate(carrier, hard=True)
        assert recovered.shape == (batch, k_expected)
        assert recovered.dtype == torch.long
        assert (recovered >= 0).all().item()
        assert (recovered < alphabet).all().item(), (
            f"{modality}: codes out of range, max={recovered.max().item()}"
        )


def test_shared_mux_identity_across_wmls() -> None:
    """All 5 SensoryWMLs must reference the same `mux` object (ADR-0001)."""
    mux = GammaThetaMultiplexer(seed=0)
    encoders = [
        AudioEncoder(),
        VisionEncoder(),
        TactileEncoder(),
        GravityEncoder(),
        ForceEncoder(),
    ]
    modalities = ("audio", "vision", "tactile", "gravity", "force")
    wmls = [
        SensoryWML(id=i, modality=modalities[i], input_proj=enc, mux=mux, seed=i)
        for i, enc in enumerate(encoders)
    ]
    # All SensoryWMLs should share the same mux identity.
    assert all(w.mux is mux for w in wmls)
    assert all(id(w.mux.constellation) == id(mux.constellation) for w in wmls)
