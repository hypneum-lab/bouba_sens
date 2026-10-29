"""Issue #45 — extract_phase2_adapted.py refuses misaligned modalities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from extract_phase2_adapted import (  # type: ignore[import-not-found]  # noqa: E402
    check_aligned_lengths,
)


def _six(n: int, n_vision: int) -> dict[str, torch.Tensor]:
    return {
        "audio.pt": torch.zeros(n, 128),
        "vision.pt": torch.zeros(n_vision, 16, 16),
        "tactile.pt": torch.zeros(n, 32),
        "gravity.pt": torch.zeros(n, 3),
        "force.pt": torch.zeros(n, 6),
        "labels.pt": torch.zeros(n, dtype=torch.long),
    }


def test_aligned_set_passes() -> None:
    check_aligned_lengths(_six(1800, 1800))


def test_short_vision_is_rejected_with_every_length() -> None:
    """The public 12 s video gives 120 vision rows against 1800."""
    with pytest.raises(ValueError) as excinfo:
        check_aligned_lengths(_six(1800, 120))
    msg = str(excinfo.value)
    assert "vision.pt=120" in msg
    for name in ("audio.pt", "tactile.pt", "gravity.pt", "force.pt", "labels.pt"):
        assert f"{name}=1800" in msg
