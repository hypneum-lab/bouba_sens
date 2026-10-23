"""Deterministic trace of the DEFAULT AdaptationLoop path (regression guard).

Used by test_loop_default_regression.py and, to generate the golden file,
against a checkout of origin/main (before the v0.8 probe/control options).
"""

from __future__ import annotations

import hashlib
import json
import platform

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


class RecordingWorld(GaussianWorld):
    def __init__(self) -> None:
        super().__init__(seed=0)
        self.calls: list[tuple[int, int]] = []

    def sample(self, batch_size: int, seed: int):  # type: ignore[no-untyped-def]
        self.calls.append((batch_size, seed))
        return super().sample(batch_size=batch_size, seed=seed)


def _sha(tensors: list[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def trace() -> dict[str, object]:
    torch.manual_seed(0)
    world = RecordingWorld()
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

    torch.manual_seed(1)
    loop.pretrain(steps=5, batch_size=4, seed=0)
    n_pretrain = len(world.calls)
    spec = LesionSpec(modality="audio", mode="M2", timing="T2", schedule=m2_snr_schedule)
    report = loop.lesion_phase(spec, steps=6, batch_size=4, seed=50, stats_every=2)

    params: list[torch.Tensor] = []
    for module in (mux, nerve, head, *sensories.values()):
        params += [p for _, p in sorted(module.state_dict().items())]
    assert report.pre_lesion_codes is not None and report.post_lesion_codes is not None
    return {
        "torch": torch.__version__,
        "machine": platform.machine(),
        "pretrain_calls": world.calls[:n_pretrain],
        "lesion_calls": world.calls[n_pretrain:],
        "loss_curve": report.loss_curve,
        "params_sha256": _sha(params),
        "pre_codes_sha256": _sha([report.pre_lesion_codes]),
        "post_codes_sha256": _sha([report.post_lesion_codes]),
    }


if __name__ == "__main__":
    print(json.dumps(trace(), indent=1))
