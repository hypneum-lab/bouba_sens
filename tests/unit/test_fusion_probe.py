"""v0.9 rev. B: linear-probe fusion gain on frozen representations (PI 2026-10-25)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from bouba_sens import fusion_probe as fp
from bouba_sens.probe import stratified_heldout_indices
from bouba_sens.sensory import MODALITIES
from bouba_sens.world.studyforrest import StudyforrestRealWorld

RUNS = [(0, 170), (2, 2550), (1, 1620), (2, 140), (3, 680),
        (1, 1860), (2, 1450), (3, 380), (2, 140), (0, 10)]  # fmt: skip


def _labels() -> torch.Tensor:
    return torch.cat([torch.full((n,), c, dtype=torch.long) for c, n in RUNS])


def test_probe_training_rows_avoid_heldout_with_buffer_and_cap_class_0() -> None:
    y = _labels()
    held = stratified_heldout_indices(y)
    masked = y.clone()
    masked[held] = -1
    rows, ids, per_class = fp.probe_training_rows(masked, held)
    hs = set(held.tolist())
    assert not set(rows.tolist()) & hs
    dist = np.min(np.abs(rows.numpy()[:, None] - held.numpy()[None, :]), axis=1)
    assert dist.min() > fp.BUFFER
    assert per_class[1] == per_class[2] == per_class[3] == fp.BLOCKS_PER_CLASS
    assert 1 <= per_class[0] < fp.BLOCKS_PER_CLASS  # class 0 keeps what is available
    assert len(rows) == 16 * sum(per_class.values()) and len(set(rows.tolist())) == len(rows)
    for b in torch.unique(ids):
        blk = rows[ids == b]
        assert torch.equal(blk, torch.arange(int(b), int(b) + 16))
        assert len(set(y[blk].tolist())) == 1
    again = fp.probe_training_rows(masked, held)
    assert torch.equal(again[0], rows)


def _synthetic(fused_info: float, seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)

    def draw(n: int):  # type: ignore[no-untyped-def]
        y = np.repeat(np.arange(4), n // 4)
        onehot = np.eye(4)[y]
        r = {m: rng.normal(size=(n, 128)) for m in MODALITIES}
        r["audio"][:, :4] += 1.5 * onehot  # the one informative modality
        r["all"] = rng.normal(size=(n, 896))
        r["all"][:, :4] += fused_info * onehot
        return r, y

    r_tr, y_tr = draw(400)
    r_te, y_te = draw(200)
    groups = np.repeat(np.arange(25), 16)
    return fp.fg_from_representations(r_tr, y_tr, groups, r_te, y_te)


def test_fused_advantage_is_detected_and_best_is_chosen_on_training() -> None:
    strong, none = _synthetic(6.0), _synthetic(0.0)
    assert strong.best_modality == "audio" and none.best_modality == "audio"
    assert strong.fg > 0.3 and none.fg < 0
    assert 0.0 <= strong.bacc_all <= 1.0 and len(strong.recalls_all) == 4
    assert strong.fg_control == pytest.approx(strong.bacc_control - strong.bacc_best)


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


@pytest.mark.usefixtures("deterministic_torch")
def test_linear_probe_fg_runs_on_a_frozen_network(sf_dir: Path) -> None:
    from bouba_sens.builders import build_loop

    w = StudyforrestRealWorld(seed=0, data_dir=sf_dir, stratified_probe=True)
    loop = build_loop(w, 9001, lock_after=None)
    loop.pretrain(2, batch_size=16, seed=9001)
    r = fp.linear_probe_fg(loop, w)
    assert r.best_modality in MODALITIES and np.isfinite(r.fg) and np.isfinite(r.fg_control)
    assert set(r.per_class_training_blocks) == {0, 1, 2, 3}
