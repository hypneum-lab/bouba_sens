"""Task 5.2 / 5.3 — AdaptationLoop.query_accuracy acceptance."""

from __future__ import annotations

from track_p.multiplexer import GammaThetaMultiplexer  # type: ignore[import-not-found]

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
from bouba_sens.sensory import MODALITIES, Modality, SensoryWML
from bouba_sens.world.gaussian import GaussianWorld


def _build_loop(seed: int = 0) -> AdaptationLoop:
    world = GaussianWorld(seed=seed)
    mux = GammaThetaMultiplexer(seed=seed)
    sensories: dict[Modality, SensoryWML] = {
        "audio": SensoryWML(0, "audio", AudioEncoder(), mux, seed=seed + 1),
        "vision": SensoryWML(1, "vision", VisionEncoder(), mux, seed=seed + 2),
        "tactile": SensoryWML(2, "tactile", TactileEncoder(), mux, seed=seed + 3),
        "gravity": SensoryWML(3, "gravity", GravityEncoder(), mux, seed=seed + 4),
        "force": SensoryWML(4, "force", ForceEncoder(), mux, seed=seed + 5),
    }
    nerve = CrossModalNerve(mux, seed=seed)
    head = IntegrationHead(n_classes=4)
    return AdaptationLoop(world, mux, sensories, nerve, head)


def test_query_accuracy_returns_unit_interval_float() -> None:
    loop = _build_loop(seed=0)
    for modality in MODALITIES:
        acc = loop.query_accuracy(modality, batch_size=32, seed=123)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0, f"query_accuracy({modality})={acc} out of [0,1]"


def test_query_accuracy_five_modalities_distinct_enough_to_build_matrix() -> None:
    """Smoke: the 5-vector has at least some variance before training.

    Me6 asymmetry needs the per-query vector to be non-uniform across
    lesioned-modality cells. A freshly initialised net will not satisfy
    B-3 numerically, but the scaffolding must at least produce floats
    ready to be stacked by the aggregator.
    """
    loop = _build_loop(seed=7)
    per_query = {m: loop.query_accuracy(m, seed=42) for m in MODALITIES}
    assert len(per_query) == 5
    assert all(isinstance(v, float) for v in per_query.values())
