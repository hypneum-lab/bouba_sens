"""Class-balanced multi-window probe (amendment v0.7 section 4.2, PI 2026-10-25)."""

from __future__ import annotations

import pytest
import torch

from bouba_sens.probe import class_balanced_probe_indices, label_run_blocks


def _runs(*spec: tuple[int, int]) -> torch.Tensor:
    return torch.cat([torch.full((n,), c, dtype=torch.long) for c, n in spec])


# Shape of the Studyforrest cache: 10 runs, class 0 short (170 + 10 frames).
SF_LIKE = _runs(
    (0, 170), (2, 2550), (1, 1620), (2, 140), (3, 680),
    (1, 1860), (2, 1450), (3, 380), (2, 140), (0, 10),
)  # fmt: skip


def test_blocks_tile_each_run_from_its_start_and_drop_remainders() -> None:
    y = _runs((0, 35), (1, 16), (0, 15), (2, 40))
    assert label_run_blocks(y, 16) == {0: [0, 16], 1: [35], 2: [66, 82]}


def test_probe_is_512_rows_balanced_unique_and_single_label_blocks() -> None:
    idx = class_balanced_probe_indices(SF_LIKE, seed=1101 - 99_999)
    assert idx.shape == (512,)
    assert len(set(idx.tolist())) == 512
    lab = SF_LIKE[idx]
    assert torch.bincount(lab, minlength=4).tolist() == [128, 128, 128, 128]
    for b in range(32):
        block = idx[16 * b : 16 * (b + 1)]
        assert torch.equal(block, torch.arange(int(block[0]), int(block[0]) + 16))
        assert len(set(SF_LIKE[block].tolist())) == 1
    assert lab[:128].eq(0).all() and lab[384:].eq(3).all()  # class order


def test_probe_is_deterministic_and_seed_dependent() -> None:
    a = class_balanced_probe_indices(SF_LIKE, seed=5)
    assert torch.equal(a, class_balanced_probe_indices(SF_LIKE, seed=5))
    assert not torch.equal(a, class_balanced_probe_indices(SF_LIKE, seed=6))


def test_probe_matches_its_definition() -> None:
    seed = 1103 - 99_999
    blocks = label_run_blocks(SF_LIKE, 16)
    gen = torch.Generator().manual_seed(seed)
    ref: list[int] = []
    for c in range(4):
        starts = sorted(blocks[c])
        for i in torch.randperm(len(starts), generator=gen)[:8].tolist():
            ref.extend(range(starts[i], starts[i] + 16))
    assert class_balanced_probe_indices(SF_LIKE, seed=seed).tolist() == ref


def test_probe_leaves_global_rng_untouched() -> None:
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    class_balanced_probe_indices(SF_LIKE, seed=0)
    assert torch.equal(torch.rand(3), expected)


def test_too_few_blocks_raises() -> None:
    y = _runs((0, 16 * 7 + 15), (1, 200), (2, 200), (3, 200))
    with pytest.raises(ValueError, match="class 0"):
        class_balanced_probe_indices(y, seed=0)
    with pytest.raises(ValueError, match="class 3"):
        class_balanced_probe_indices(_runs((0, 200), (1, 200), (2, 200)), seed=0)
