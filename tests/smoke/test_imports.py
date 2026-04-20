"""Smoke test — verify all Sprint 0 skeleton modules import cleanly.

These tests prove the package structure is valid; they do NOT exercise any
behaviour (all real entry points currently raise NotImplementedError).
"""

from __future__ import annotations

import importlib

BOUBA_SENS_MODULES = [
    "bouba_sens",
    "bouba_sens._version",
    "bouba_sens.sensory",
    "bouba_sens.nerve",
    "bouba_sens.lesion",
    "bouba_sens.head",
    "bouba_sens.loop",
    "bouba_sens.report",
    "bouba_sens.cli",
    "bouba_sens.world",
    "bouba_sens.world.base",
    "bouba_sens.world.gaussian",
    "bouba_sens.world.xor",
    "bouba_sens.world.sinusoid",
    "bouba_sens.metrics",
    "bouba_sens.metrics.performance",
    "bouba_sens.metrics.mi_migration",
    "bouba_sens.metrics.asymmetry",
    "bouba_sens.metrics.congenital",
    "bouba_sens.metrics.baselines",
    "bouba_sens.metrics.replication",
]


def test_version_exposed() -> None:
    import bouba_sens

    assert bouba_sens.__version__ == "0.0.3"


def test_all_modules_import() -> None:
    for module_name in BOUBA_SENS_MODULES:
        importlib.import_module(module_name)


def test_modality_type_constants_are_five() -> None:
    from bouba_sens.sensory import MODALITIES

    assert MODALITIES == ("audio", "vision", "tactile", "gravity", "force")
    assert len(MODALITIES) == 5


def test_world_sample_dataclass_importable() -> None:
    from bouba_sens.world import WorldSample, WorldSimulator

    assert WorldSample is not None
    assert WorldSimulator is not None
