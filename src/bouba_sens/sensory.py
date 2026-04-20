"""SensoryWML — modality-specific subclass of track_w.mlp_wml.MlpWML.

Placeholder for Sprint 0. Implementation lands in Sprint 1.
"""

from __future__ import annotations

from typing import Literal

Modality = Literal["audio", "vision", "tactile", "gravity", "force"]
MODALITIES: tuple[Modality, ...] = ("audio", "vision", "tactile", "gravity", "force")


class SensoryWML:
    """Per-modality wrapper around track_w.mlp_wml.MlpWML. See spec §3.2."""

    def __init__(self, modality: Modality) -> None:
        raise NotImplementedError("Sprint 1 — see docs/superpowers/plans/sprint1")
