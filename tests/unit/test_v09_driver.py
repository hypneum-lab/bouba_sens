"""v0.9 driver: main-grid arms and D variance components (PI 2026-10-25)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

from bouba_sens import v09_driver as drv
from bouba_sens.builders import build_loop
from bouba_sens.decorrelation import V09_PROBE_FRACTION, FusionGain, fusion_gain
from bouba_sens.world.studyforrest import StudyforrestRealWorld

# Run-structured labels (the stratified probe tiles label runs into blocks).
RUNS = [(0, 400), (1, 1200), (2, 1500), (3, 700), (1, 800), (2, 600)]

# Bit-identical comparisons need deterministic CPU kernels (tests/conftest.py).
pytestmark = pytest.mark.usefixtures("deterministic_torch")


@pytest.fixture
def real_dir(tmp_path: Path) -> Path:
    gen = torch.Generator().manual_seed(0)
    y = torch.cat([torch.full((k,), c, dtype=torch.long) for c, k in RUNS])
    n = len(y)
    torch.save(torch.randn(n, 128, generator=gen), tmp_path / "audio.pt")
    torch.save(torch.randn(n, 16, 16, generator=gen), tmp_path / "vision.pt")
    torch.save(torch.randn(n, 32, generator=gen), tmp_path / "tactile.pt")
    torch.save(torch.randn(n, 3, generator=gen), tmp_path / "gravity.pt")
    torch.save(torch.randn(n, 6, generator=gen), tmp_path / "force.pt")
    torch.save(y, tmp_path / "labels.pt")
    return tmp_path


def _world(real_dir: Path, *, stratified: bool = True) -> StudyforrestRealWorld:
    if stratified:
        return StudyforrestRealWorld(seed=0, data_dir=real_dir, stratified_probe=True)
    return StudyforrestRealWorld(seed=0, data_dir=real_dir, probe_fraction=V09_PROBE_FRACTION)


def test_frozen_constants() -> None:
    assert drv.ONSETS == (0, 50, 100, 150, 200)
    assert (drv.WIDTH, drv.RECOVERY, drv.BATCH, drv.LEARNING_RATE) == (50, 200, 16, 1e-3)
    assert drv.PILOT_SEEDS == (3101, 3102, 3103, 3104, 3105)


def test_check_world_needs_the_held_out_probe(real_dir: Path) -> None:
    assert drv.check_world(_world(real_dir)) == "StudyforrestRealWorld"
    with pytest.raises(RuntimeError, match="stratified"):
        drv.check_world(_world(real_dir, stratified=False))
    with pytest.raises(RuntimeError, match="StudyforrestRealWorld"):
        drv.check_world(object())


def test_learning_rate_check(real_dir: Path) -> None:
    loop = build_loop(_world(real_dir), 3101, lock_after=None)
    drv.check_learning_rate(loop)
    loop.opt.param_groups[0]["lr"] = 1e-2
    with pytest.raises(RuntimeError, match="learning rate"):
        drv.check_learning_rate(loop)


def test_run_cell_is_reproducible_and_plain_equals_no_schedule(real_dir: Path) -> None:
    world = _world(real_dir)
    a = drv.run_cell(world, 3101, 0, "decor", recovery=2, width=2)
    b = drv.run_cell(world, 3101, 0, "decor", recovery=2, width=2)
    assert a.fg == b.fg and a.steps == 4
    plain = drv.run_cell(world, 3101, 0, "plain", recovery=2, width=2)
    torch.manual_seed(3101)
    loop = build_loop(world, 3101, lock_after=None)
    loop.pretrain(4, batch_size=16, seed=3101)
    ref = fusion_gain(loop, train_seed=3101)
    assert plain.fg == ref


def test_deficit_matrix_and_variance_components() -> None:
    def cell(s: int, o: int, arm: str, fg: float) -> drv.CellRun:
        return drv.CellRun(s, o, arm, 200, 0, FusionGain(0.0, {}, fg), 0, 0.0)

    rng = np.random.default_rng(0)
    d_true = rng.normal(size=(5, 5)) * 0.05 + rng.normal(size=(5, 1)) * 0.03
    cells = []
    for i, s in enumerate(drv.PILOT_SEEDS):
        for j, o in enumerate(drv.ONSETS):
            cells += [cell(s, o, "joint", 0.3 + d_true[i, j]), cell(s, o, "decor", 0.3)]
    d = drv.deficit_matrix(cells, drv.PILOT_SEEDS)
    np.testing.assert_allclose(d, d_true, atol=1e-12)
    vc = drv.variance_components(d)
    grand = d.mean()
    ms_seed = 5 * ((d.mean(1) - grand) ** 2).sum() / 4
    resid = d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + grand
    ms_res = (resid**2).sum() / 16
    assert vc["df_residual"] == 16
    assert vc["ms_residual"] == pytest.approx(ms_res)
    assert vc["sigma_u2"] == pytest.approx(max(0.0, (ms_seed - ms_res) / 5))
    assert vc["sigma_hat"] == pytest.approx(math.sqrt(vc["sigma_u2"] + ms_res))
    assert vc["ucl_multiplier"] == pytest.approx(math.sqrt(4 / stats.chi2.ppf(0.2, 4)))
    assert vc["ucl_multiplier"] == pytest.approx(1.558, abs=5e-4)
    assert vc["sigma_plan"] == pytest.approx(vc["sigma_hat"] * vc["ucl_multiplier"])


def test_run_cell_requires_deterministic_kernels(real_dir: Path) -> None:
    torch.use_deterministic_algorithms(False)  # the fixture restores it
    with pytest.raises(RuntimeError, match="deterministic"):
        drv.run_cell(_world(real_dir), 3101, 0, "decor", recovery=2, width=2)
