"""Model construction shared by the amendment drivers (v0.8, v0.9).

``build_loop`` builds the same network as the B-2 runner
(``scripts/spikes/e1_b2_dimensional_mine.py``, ``build_loop``) and the
CLI: five ``SensoryWML`` (module seeds s + 1 ... s + 5), one
``GammaThetaMultiplexer`` (seed s, optional constellation lock), a
``CrossModalNerve`` with HARD transducer gating and a 4-class head, under
the default ``AdaptationLoop`` optimiser (Adam, constant lr 1e-3). The
world is passed in, so one loaded cache serves every arm of a seed.
"""

from __future__ import annotations

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
from bouba_sens.sensory import Modality, SensoryWML
from bouba_sens.world.base import WorldSimulator

N_CLASSES = 4
TRANSDUCER_GATING = "hard"


def build_loop(
    world: WorldSimulator,
    seed: int,
    *,
    lock_after: int | None,
    seed_namespace: int | None = None,
) -> AdaptationLoop:
    """Fresh loop for training seed ``seed``; ``lock_after=None`` = never locked."""
    mux = GammaThetaMultiplexer(seed=seed, constellation_lock_after=lock_after)
    sensories: dict[Modality, SensoryWML] = {
        "audio": SensoryWML(0, "audio", AudioEncoder(), mux, seed=seed + 1),
        "vision": SensoryWML(1, "vision", VisionEncoder(), mux, seed=seed + 2),
        "tactile": SensoryWML(2, "tactile", TactileEncoder(), mux, seed=seed + 3),
        "gravity": SensoryWML(3, "gravity", GravityEncoder(), mux, seed=seed + 4),
        "force": SensoryWML(4, "force", ForceEncoder(), mux, seed=seed + 5),
    }
    nerve = CrossModalNerve(mux, seed=seed, transducer_gating=TRANSDUCER_GATING)
    head = IntegrationHead(n_classes=N_CLASSES)
    return AdaptationLoop(world, mux, sensories, nerve, head, seed_namespace=seed_namespace)
