"""ADR-0022 — AdaptationLoop.query_successes and the query_accuracy refactor."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

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
from bouba_sens.loop import AdaptationLoop, successes_to_accuracy
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


def _pre_refactor_query_accuracy(
    loop: AdaptationLoop, query_modality: Modality, *, batch_size: int, seed: int
) -> float:
    """The exact pre-ADR-0022 body of `query_accuracy`, kept for non-regression."""
    sample = loop.world.sample(batch_size=batch_size, seed=seed)
    masked_tensors = {}
    for m in MODALITIES:
        tensor = getattr(sample, m)
        masked_tensors[m] = tensor if m == query_modality else torch.zeros_like(tensor)
    masked = replace(
        sample,
        audio=masked_tensors["audio"],
        vision=masked_tensors["vision"],
        tactile=masked_tensors["tactile"],
        gravity=masked_tensors["gravity"],
        force=masked_tensors["force"],
    )
    with torch.no_grad():
        _, logits = loop._forward(masked)
        preds = logits.argmax(-1)
        return float((preds == sample.label).float().mean().item())


def test_query_successes_has_batch_size_binary_entries() -> None:
    loop = _build_loop(seed=0)
    for modality in MODALITIES:
        successes = loop.query_successes(modality, batch_size=37, seed=123)
        assert isinstance(successes, list)
        assert len(successes) == 37
        assert all(isinstance(s, int) and s in (0, 1) for s in successes)


def test_query_successes_is_deterministic_for_a_seed(deterministic_torch: None) -> None:
    """Same probe seed and same global torch RNG state -> same outcomes.

    The forward pass draws from the global torch RNG (modules are in
    training mode), so the RNG is re-seeded before each compared call.
    """
    loop = _build_loop(seed=3)
    torch.manual_seed(0)
    a = loop.query_successes("vision", batch_size=64, seed=99)
    torch.manual_seed(0)
    b = loop.query_successes("vision", batch_size=64, seed=99)
    assert a == b


def test_query_accuracy_equals_mean_of_successes(deterministic_torch: None) -> None:
    """Non power-of-two batch so the float32 mean actually rounds."""
    loop = _build_loop(seed=1)
    for modality in MODALITIES:
        torch.manual_seed(1)
        successes = loop.query_successes(modality, batch_size=48, seed=321)
        torch.manual_seed(1)
        acc = loop.query_accuracy(modality, batch_size=48, seed=321)
        assert acc == successes_to_accuracy(successes)
        assert acc == float(torch.tensor(successes, dtype=torch.float32).mean().item())
        assert abs(acc - sum(successes) / len(successes)) < 1e-6


def test_query_accuracy_bit_identical_to_pre_refactor_path(deterministic_torch: None) -> None:
    """Bit-for-bit non-regression against the pre-ADR-0022 implementation."""
    loop = _build_loop(seed=5)
    for batch_size in (64, 48, 7):
        for modality in MODALITIES:
            torch.manual_seed(2)
            old = _pre_refactor_query_accuracy(loop, modality, batch_size=batch_size, seed=2024)
            torch.manual_seed(2)
            new = loop.query_accuracy(modality, batch_size=batch_size, seed=2024)
            assert new == old, (modality, batch_size, new, old)


def test_successes_to_accuracy_matches_float32_mean() -> None:
    successes = [1, 0, 1, 1, 0, 1, 0]
    expected = float(torch.tensor(successes, dtype=torch.float32).mean().item())
    assert successes_to_accuracy(successes) == expected
    assert successes_to_accuracy([1, 1, 1, 1]) == 1.0
    assert successes_to_accuracy([0, 0]) == 0.0


def test_cli_lesion_writes_per_query_successes(tmp_path: Path) -> None:
    """`bouba-sens lesion` emits per_query_successes.json next to per_query_me1.json."""
    lesion_dir = tmp_path / "phase2"
    subprocess.run(
        [
            "uv",
            "run",
            "bouba-sens",
            "lesion",
            "--modality",
            "audio",
            "--timing",
            "T1",
            "--steps",
            "3",
            "--seed",
            "1",
            "--out",
            str(lesion_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    successes = json.loads((lesion_dir / "per_query_successes.json").read_text())
    me1 = json.loads((lesion_dir / "per_query_me1.json").read_text())
    assert set(successes) == set(MODALITIES) == set(me1)
    for modality in MODALITIES:
        outcomes = successes[modality]
        assert len(outcomes) == 64
        assert all(isinstance(s, int) and s in (0, 1) for s in outcomes)
        assert me1[modality] == successes_to_accuracy(outcomes)
