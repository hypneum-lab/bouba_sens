"""ADR-0026 — opt-in length of the Studyforrest mock stream."""

from __future__ import annotations

import pytest
import torch
import typer

from bouba_sens.cli import MOCK_SAMPLES_ENV, _build_world, _mock_record
from bouba_sens.world.studyforrest import StudyforrestWorld, _build_mock_cache, stream_sha256


def test_default_mock_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MOCK_SAMPLES_ENV, raising=False)
    monkeypatch.delenv("BOUBA_SENS_STUDYFORREST_DATA", raising=False)
    world = _build_world("studyforrest", seed=0)
    assert isinstance(world, StudyforrestWorld)
    assert world.n_rows == 4096
    ref = _build_mock_cache(4096, 0)
    assert world.cache_sha256() == stream_sha256(*ref)
    assert _mock_record("studyforrest", world) == {}
    a = world.sample(64, seed=3)
    b = StudyforrestWorld(seed=0).sample(64, seed=3)
    assert torch.equal(a.audio, b.audio) and torch.equal(a.label, b.label)


def test_override_builds_a_longer_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOUBA_SENS_STUDYFORREST_DATA", raising=False)
    monkeypatch.setenv(MOCK_SAMPLES_ENV, "8192")
    world = _build_world("studyforrest", seed=0)
    assert isinstance(world, StudyforrestWorld) and world.n_rows == 8192
    record = _mock_record("studyforrest", world)
    assert record == {"mock_samples": 8192, "mock_cache_sha256": world.cache_sha256()}
    assert record["mock_cache_sha256"] == stream_sha256(*_build_mock_cache(8192, 0))
    assert record["mock_cache_sha256"] != StudyforrestWorld(seed=0).cache_sha256()
    # Deterministic: same seed, same stream.
    assert _build_world("studyforrest", seed=0).cache_sha256() == world.cache_sha256()
    # The override does not touch other worlds.
    assert _mock_record("gaussian", _build_world("gaussian", seed=0)) == {}


@pytest.mark.parametrize("bad", ["0", "1", "-5", "lots"])
def test_override_rejects_bad_values(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.delenv("BOUBA_SENS_STUDYFORREST_DATA", raising=False)
    monkeypatch.setenv(MOCK_SAMPLES_ENV, bad)
    with pytest.raises(typer.BadParameter):
        _build_world("studyforrest", seed=0)


def test_stream_sha256_is_sensitive_to_every_tensor() -> None:
    audio, vision, labels = _build_mock_cache(64, 0)
    base = stream_sha256(audio, vision, labels)
    assert base == stream_sha256(audio.clone(), vision.clone(), labels.clone())
    for i in range(3):
        parts = [audio.clone(), vision.clone(), labels.clone()]
        parts[i].view(-1)[0] += 1
        assert stream_sha256(*parts) != base
