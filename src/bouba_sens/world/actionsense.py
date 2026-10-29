"""ActionSenseWorld — 5-modality real-data bridge (ActionSense kitchen).

Reads the tensors written by `scripts/extract_actionsense.py`
(`audio, vision, tactile, gravity, force, labels`; `subject` is optional
and not part of the contract) and serves temporally contiguous batches.
Serving behaviour is identical to `StudyforrestRealWorld`; this class
adds a strict row-alignment check and its own `mode`.
"""

from __future__ import annotations

from pathlib import Path

import torch

from bouba_sens.world.actionsense_features import check_aligned
from bouba_sens.world.studyforrest import StudyforrestRealWorld


class ActionSenseWorld(StudyforrestRealWorld):
    """Real egocentric multimodal cache, one row per aligned window."""

    def __init__(self, *, seed: int = 0, data_dir: Path) -> None:
        super().__init__(seed=seed, data_dir=data_dir)
        tensors: dict[str, torch.Tensor] = {
            "audio": self._audio,
            "vision": self._vision,
            "tactile": self._tactile,
            "gravity": self._gravity,
            "force": self._force,
            "labels": self._labels,
        }
        check_aligned({k: t.numpy() for k, t in tensors.items()})
        self._mode = "actionsense"
