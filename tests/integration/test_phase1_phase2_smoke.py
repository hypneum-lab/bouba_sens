"""Task 2.11 — Sprint 2 acceptance gate.

End-to-end smoke: 100-step Phase 1 (intact pretrain) + 100-step Phase 2
(audio lesion) on GaussianWorld must:
- run without NaN
- finish above chance accuracy (4-class, chance = 0.25)
- show the audio gate dropping at least 10 % across Phase 2
  (evidence that on_lesion's bias + loss gradient actually
  push the gate away from the lesioned channel)
"""

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


def test_phase1_phase2_end_to_end_smoke() -> None:
    torch.manual_seed(0)
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
    head = IntegrationHead(n_classes=4)
    loop = AdaptationLoop(world, mux, sensories, nerve, head, lr=1e-3)

    # Phase 1 — intact pretrain.
    ckpt = loop.pretrain(steps=100, batch_size=16, seed=0)
    assert ckpt.mux_state

    # Phase 2 — audio lesion.
    spec = LesionSpec(
        modality="audio",
        mode="M2",
        timing="T2",
        schedule=m2_snr_schedule,
    )
    report = loop.lesion_phase(spec, steps=100, batch_size=16, stats_every=10)

    # No NaN anywhere in the loss curve.
    for loss_val in report.loss_curve:
        assert loss_val == loss_val, "NaN encountered in Phase 2 loss"

    # Final accuracy must beat chance (0.25 for 4-class).
    final_acc = sum(report.accuracy_curve[-10:]) / 10
    assert final_acc > 0.25, f"final acc {final_acc:.3f} <= chance 0.25"

    # Audio gate must DECREASE by at least 10 % across Phase 2.
    audio_start = report.gate_trajectory[0]["audio"]
    audio_end = report.gate_trajectory[-1]["audio"]
    assert audio_end < audio_start * 0.9, (
        f"audio gate did not decrease by ≥10 % (start={audio_start:.3f}, end={audio_end:.3f})"
    )

    # Migration trajectories should be non-empty.
    assert len(report.gate_trajectory) >= 2
    assert len(report.codebook_entropy_trajectory) >= 2
    assert len(report.transducer_activation_trajectory) >= 2
