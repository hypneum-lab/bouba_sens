"""Class-stratified probe draw (ADR-0019, amendment 2026-11-06)."""

from __future__ import annotations

import pytest
import torch

from bouba_sens.world.probe_sampling import (
    sample_probe,
    stratified_allocation,
    stratified_indices,
)


def _scene_labels() -> torch.Tensor:
    # Long runs of one class, like the quantised Studyforrest scene index.
    runs = [(0, 3000), (1, 2500), (2, 2000), (3, 1500)]
    return torch.cat([torch.full((k,), c, dtype=torch.long) for c, k in runs])


def test_balanced_draw_has_all_classes_and_no_repeats() -> None:
    labels = _scene_labels()
    idx = stratified_indices(labels, 512, seed=-99_999)
    assert idx.numel() == 512
    assert torch.unique(idx).numel() == 512  # without replacement
    assert torch.all(idx[1:] > idx[:-1])  # sorted
    counts = torch.bincount(labels[idx], minlength=4).tolist()
    assert counts == [128, 128, 128, 128]


def test_small_class_gives_all_rows_and_shortfall_is_shared() -> None:
    labels = torch.cat(
        [
            torch.zeros(5000, dtype=torch.long),
            torch.ones(3000, dtype=torch.long),
            torch.full((40,), 2, dtype=torch.long),
            torch.full((960,), 3, dtype=torch.long),
        ]
    )
    idx = stratified_indices(labels, 512, seed=7)
    counts = torch.bincount(labels[idx], minlength=4).tolist()
    assert sum(counts) == 512
    assert counts[2] == 40
    assert counts == [158, 157, 40, 157]
    assert torch.unique(idx).numel() == 512


def test_allocation_remainder_goes_by_label_order() -> None:
    assert stratified_allocation({0: 100, 1: 100, 2: 100}, 10) == {0: 4, 1: 3, 2: 3}


def test_draw_is_deterministic_in_seed() -> None:
    labels = _scene_labels()
    a = stratified_indices(labels, 512, seed=3)
    b = stratified_indices(labels, 512, seed=3)
    c = stratified_indices(labels, 512, seed=4)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


class _World:
    def __init__(self, labels: torch.Tensor) -> None:
        self.dataset_labels = labels

    def gather(self, idx: torch.Tensor) -> torch.Tensor:
        return self.dataset_labels[idx]

    def sample(self, batch_size: int, seed: int) -> str:
        return "contiguous"


def test_sample_probe_modes() -> None:
    world = _World(_scene_labels())
    assert sample_probe(world, 512, 0, "contiguous") == "contiguous"
    probe = sample_probe(world, 512, 0, "stratified")
    assert torch.bincount(probe, minlength=4).tolist() == [128] * 4
    with pytest.raises(ValueError):
        sample_probe(world, 512, 0, "other")
    with pytest.raises(ValueError):
        sample_probe(object(), 512, 0, "stratified")
