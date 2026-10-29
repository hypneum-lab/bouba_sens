"""Held-out probe split of ``StudyforrestRealWorld`` (amendment v0.9, section 3.4).

With ``probe_fraction`` > 0 the last frames are reserved for the
fusion-gain probe: no training draw (plain batches, decorrelation
anchors and donors, manipulation-check anchors) may touch them, and
windows never straddle the boundary. With the default (0.0) the world
behaves exactly as before.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from bouba_sens.decorrelation import (
    PROBE_ROWS,
    V09_PROBE_FRACTION,
    DecorrelationSchedule,
    DecorrelationWindow,
    DonorIndex,
    check_anchors,
    fusion_gain,
)
from bouba_sens.sensory import MODALITIES
from bouba_sens.world.base import WorldSample
from bouba_sens.world.studyforrest import StudyforrestRealWorld

N_ROWS = 1200


@pytest.fixture
def real_dir(tmp_path: Path) -> Path:
    gen = torch.Generator().manual_seed(0)
    n = N_ROWS
    torch.save(torch.randn(n, 128, generator=gen), tmp_path / "audio.pt")
    torch.save(torch.randn(n, 16, 16, generator=gen), tmp_path / "vision.pt")
    torch.save(torch.randn(n, 32, generator=gen), tmp_path / "tactile.pt")
    torch.save(torch.randn(n, 3, generator=gen), tmp_path / "gravity.pt")
    torch.save(torch.randn(n, 6, generator=gen), tmp_path / "force.pt")
    torch.save(torch.randint(0, 4, (n,), generator=gen), tmp_path / "labels.pt")
    return tmp_path


def _legacy_indices(n: int, batch_size: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    start = int(torch.randint(0, max(1, n - batch_size), (1,), generator=gen).item())
    return torch.arange(start, start + batch_size) % n


def _split(real_dir: Path) -> StudyforrestRealWorld:
    return StudyforrestRealWorld(seed=0, data_dir=real_dir, probe_fraction=V09_PROBE_FRACTION)


def test_v09_probe_fraction_is_ten_percent() -> None:
    assert V09_PROBE_FRACTION == 0.10


def test_default_world_is_unchanged(real_dir: Path) -> None:
    world = StudyforrestRealWorld(seed=0, data_dir=real_dir)
    assert world.n_train == N_ROWS
    assert world.n_probe == 0
    assert world.dataset_labels.shape == (N_ROWS,)
    for b, s in [(16, 0), (16, 12345), (512, 7), (1199, 3)]:
        _, idx = world.sample_with_indices(b, s)
        assert torch.equal(idx, _legacy_indices(N_ROWS, b, s))
    with pytest.raises(ValueError, match="held-out probe"):
        world.probe_sample_with_indices(16, 0)


def test_split_sizes_take_the_last_frames(real_dir: Path) -> None:
    world = _split(real_dir)
    assert world.n_probe == math.ceil(0.10 * N_ROWS) == 120
    assert world.n_train == N_ROWS - 120
    assert world.dataset_labels.shape == (world.n_train,)
    labels = torch.load(real_dir / "labels.pt")
    assert torch.equal(world.dataset_labels, labels[: world.n_train])


@pytest.mark.parametrize("batch_size", [1, 16, 64, 512, 1079])
def test_training_windows_never_reach_probe_frames(real_dir: Path, batch_size: int) -> None:
    world = _split(real_dir)
    for seed in range(300):
        sample, idx = world.sample_with_indices(batch_size, seed)
        assert int(idx.max()) < world.n_train
        assert torch.equal(idx, torch.arange(int(idx[0]), int(idx[0]) + batch_size))
        assert torch.equal(sample.label, world.dataset_labels[idx])
        assert torch.equal(world.sample(batch_size, seed).audio, sample.audio)


def test_training_batch_larger_than_train_split_is_refused(real_dir: Path) -> None:
    world = _split(real_dir)
    with pytest.raises(ValueError, match="training split"):
        world.sample_with_indices(world.n_train + 1, 0)


def test_probe_windows_stay_inside_probe_frames(real_dir: Path) -> None:
    world = _split(real_dir)
    for seed in range(300):
        sample, idx = world.probe_sample_with_indices(64, seed)
        assert int(idx.min()) >= world.n_train
        assert int(idx.max()) < N_ROWS
        assert torch.equal(idx, torch.arange(int(idx[0]), int(idx[0]) + 64))
        assert torch.equal(sample.audio, world.gather(idx).audio)
    _, whole = world.probe_sample_with_indices(world.n_probe, 0)
    assert torch.equal(whole, torch.arange(world.n_train, N_ROWS))
    with pytest.raises(ValueError, match="probe split"):
        world.probe_sample_with_indices(world.n_probe + 1, 0)


def test_donors_and_check_anchors_come_from_training_frames(real_dir: Path) -> None:
    world = _split(real_dir)
    index = DonorIndex(world.dataset_labels)
    _, anchors = check_anchors(world, train_seed=9001)  # outside every seed pool
    assert int(anchors.max()) < world.n_train
    for draw in range(50):
        donors, _ = index.draw(anchors, 10**9 + draw)
        assert int(donors.max()) < world.n_train


@pytest.mark.parametrize("arm", ["plain", "joint", "decor"])
def test_phase1_training_never_sees_a_probe_frame(real_dir: Path, arm: str) -> None:
    from tests.unit.test_loop import _build_loop

    world = _split(real_dir)
    seen_rows: list[torch.Tensor] = []
    gather = world.gather

    def spy_gather(idx: torch.Tensor) -> WorldSample:
        seen_rows.append(idx.clone())
        return gather(idx)

    world.gather = spy_gather  # type: ignore[method-assign]
    loop = _build_loop(n_classes=4)
    loop.world = world
    sch = DecorrelationSchedule(DecorrelationWindow(1, 3), arm, seed=5, world=world)
    loop.pretrain(steps=5, batch_size=32, seed=0, decorrelation=sch)
    assert seen_rows
    assert int(torch.cat(seen_rows).max()) < world.n_train


def test_fusion_gain_uses_held_out_probe_rows(real_dir: Path) -> None:
    world = StudyforrestRealWorld(seed=0, data_dir=real_dir, probe_fraction=0.5)
    assert world.n_probe >= PROBE_ROWS
    forwarded: list[WorldSample] = []

    class StubLoop:
        def __init__(self) -> None:
            self.world = world

        def _forward(self, sample: WorldSample) -> tuple[torch.Tensor, torch.Tensor]:
            forwarded.append(sample)
            logits = torch.nn.functional.one_hot(sample.label, 4).float()
            if float(sample.audio.abs().sum()) == 0.0:
                logits = torch.roll(logits, 1, dims=-1)  # audio masked -> always wrong
            return torch.tensor(0.0), logits

        def query_accuracy(self, *a: object, **kw: object) -> float:
            raise AssertionError("held-out path must not resample the training stream")

    res = fusion_gain(StubLoop(), train_seed=11)  # type: ignore[arg-type]
    expected, idx = world.probe_sample_with_indices(PROBE_ROWS, 11 + 2 * 10**6)
    assert int(idx.min()) >= world.n_train
    assert len(forwarded) == 1 + len(MODALITIES)
    assert torch.equal(forwarded[0].audio, expected.audio)
    for s in forwarded:
        assert torch.equal(s.label, expected.label)
    assert res.acc_all == 1.0
    assert res.acc_by_modality["audio"] == 1.0
    assert all(res.acc_by_modality[m] == 0.0 for m in MODALITIES if m != "audio")
    assert res.fusion_gain == 0.0


@pytest.mark.parametrize("frac", [-0.1, 1.0, 1.5])
def test_probe_fraction_out_of_range_is_refused(real_dir: Path, frac: float) -> None:
    with pytest.raises(ValueError, match="probe_fraction"):
        StudyforrestRealWorld(seed=0, data_dir=real_dir, probe_fraction=frac)
