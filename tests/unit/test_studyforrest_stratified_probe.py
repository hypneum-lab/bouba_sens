"""Class-stratified held-out probe for v0.9 (section 3.4, PI decision 2026-10-25)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bouba_sens.decorrelation import DonorIndex, check_anchors, fusion_gain
from bouba_sens.probe import (
    V09_HELDOUT_SEED,
    label_run_blocks,
    proportional_block_counts,
    stratified_heldout_indices,
)
from bouba_sens.world.studyforrest import StudyforrestRealWorld

# Run structure of the Studyforrest cache (10 runs, 9,000 frames).
RUNS = [(0, 170), (2, 2550), (1, 1620), (2, 140), (3, 680),
        (1, 1860), (2, 1450), (3, 380), (2, 140), (0, 10)]  # fmt: skip


def _labels() -> torch.Tensor:
    return torch.cat([torch.full((n,), c, dtype=torch.long) for c, n in RUNS])


@pytest.fixture
def sf_dir(tmp_path: Path) -> Path:
    gen = torch.Generator().manual_seed(0)
    y = _labels()
    n = len(y)
    torch.save(torch.randn(n, 128, generator=gen), tmp_path / "audio.pt")
    torch.save(torch.randn(n, 16, 16, generator=gen), tmp_path / "vision.pt")
    torch.save(torch.randn(n, 32, generator=gen), tmp_path / "tactile.pt")
    torch.save(torch.randn(n, 3, generator=gen), tmp_path / "gravity.pt")
    torch.save(torch.randn(n, 6, generator=gen), tmp_path / "force.pt")
    torch.save(y, tmp_path / "labels.pt")
    return tmp_path


def test_apportionment_on_the_cache_counts() -> None:
    assert proportional_block_counts([180, 3480, 4280, 1060], 32) == [1, 12, 15, 4]
    assert proportional_block_counts([25, 25, 25, 25], 32) == [8, 8, 8, 8]
    assert proportional_block_counts([1, 1000, 1000, 1000], 6) == [1, 2, 2, 1]
    with pytest.raises(ValueError):
        proportional_block_counts([1, 1, 1, 1], 3)


def test_heldout_rows_are_stratified_blocks_and_match_the_definition() -> None:
    y = _labels()
    idx = stratified_heldout_indices(y)
    assert len(idx) == 512 and len(set(idx.tolist())) == 512
    assert torch.equal(idx, idx.sort().values)
    assert torch.bincount(y[idx], minlength=4).tolist() == [16, 192, 240, 64]
    blocks = label_run_blocks(y, 16)
    gen = torch.Generator().manual_seed(V09_HELDOUT_SEED)
    ref: list[int] = []
    for c, k in enumerate([1, 12, 15, 4]):
        starts = sorted(blocks[c])
        for i in torch.randperm(len(starts), generator=gen)[:k].tolist():
            ref.extend(range(starts[i], starts[i] + 16))
    assert idx.tolist() == sorted(ref)


def test_world_keeps_training_off_the_probe(sf_dir: Path) -> None:
    w = StudyforrestRealWorld(seed=0, data_dir=sf_dir, stratified_probe=True)
    held = set(stratified_heldout_indices(_labels()).tolist())
    assert w.stratified and (w.n_probe, w.n_train) == (512, 9000 - 512)
    for seed in range(300):
        _, idx = w.sample_with_indices(16, seed)
        assert not held & set(idx.tolist())
        assert torch.equal(idx, torch.arange(int(idx[0]), int(idx[0]) + 16))
    _, anchors = check_anchors(w, 3101)
    assert not held & set(anchors.tolist())
    lab = w.dataset_labels
    assert len(lab) == 9000 and (lab[sorted(held)] == -1).all()
    donors, n_self = DonorIndex(lab).draw(anchors, 123)
    assert n_self == 0 and not held & set(donors.flatten().tolist())
    assert torch.equal(lab[donors], _labels()[anchors].expand_as(donors))


def test_probe_is_every_heldout_row_for_any_seed(sf_dir: Path) -> None:
    w = StudyforrestRealWorld(seed=0, data_dir=sf_dir, stratified_probe=True)
    a, ia = w.probe_sample_with_indices(512, 1)
    _, ib = w.probe_sample_with_indices(512, 2)
    assert torch.equal(ia, ib) and torch.equal(ia, stratified_heldout_indices(_labels()))
    assert torch.bincount(a.label, minlength=4).tolist() == [16, 192, 240, 64]
    with pytest.raises(ValueError, match="512"):
        w.probe_sample_with_indices(256, 1)


def test_stratified_probe_excludes_probe_fraction(sf_dir: Path) -> None:
    with pytest.raises(ValueError, match="exclusive"):
        StudyforrestRealWorld(seed=0, data_dir=sf_dir, probe_fraction=0.1, stratified_probe=True)


def test_default_world_unchanged(sf_dir: Path) -> None:
    w = StudyforrestRealWorld(seed=0, data_dir=sf_dir)
    assert not w.stratified and w.n_probe == 0
    assert torch.equal(w.dataset_labels, _labels())


def test_fusion_gain_uses_the_stratified_probe(sf_dir: Path) -> None:
    from bouba_sens.builders import build_loop

    w = StudyforrestRealWorld(seed=0, data_dir=sf_dir, stratified_probe=True)
    loop = build_loop(w, 3101, lock_after=None)
    fg = fusion_gain(loop, train_seed=3101)
    probe, _ = w.probe_sample_with_indices(512, 0)
    with torch.no_grad():
        _, logits = loop._forward(probe)
    assert fg.acc_all == float((logits.argmax(-1) == probe.label).float().mean())
