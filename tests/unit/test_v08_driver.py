"""v0.8 driver (amendment v0.8 sections 3 item 11, 3.1 and 7; PI 2026-10-25)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

from bouba_sens import v08_driver as drv
from bouba_sens.metrics.subspace import select_k_from_pilot
from bouba_sens.metrics.subspace_aggregate import LOCKS
from bouba_sens.sensory import MODALITIES
from bouba_sens.subspace_protocol import SNR_FLOORS, BlockCodes
from bouba_sens.world.gaussian import GaussianWorld

_SMALL = {
    "modalities": ("audio", "vision"),
    "snr_floors": (-20.0, 10.0),
    "steps": 2,
    "batch_size": 4,
    "probe_size": 16,
}

# Bit-identical comparisons need deterministic CPU kernels (tests/conftest.py).
pytestmark = pytest.mark.usefixtures("deterministic_torch")


def test_check_world_refuses_anything_but_the_real_bridge() -> None:
    with pytest.raises(RuntimeError, match="StudyforrestRealWorld"):
        drv.check_world(GaussianWorld(seed=0))


def test_run_block_is_reproducible_and_shares_z_pre(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(drv, "STEPS_TRAIN", 2)
    world = GaussianWorld(seed=0)
    a = drv.run_block(world, 2101, 25, **_SMALL).block
    b = drv.run_block(world, 2101, 25, **_SMALL).block
    assert (a.seed, a.lock, a.phase2_seed) == (2101, 25, 2101)
    assert torch.equal(a.z_pre, b.z_pre) and torch.equal(a.z_control, b.z_control)
    assert set(a.z_post) == {(m, f) for m in ("audio", "vision") for f in (-20.0, 10.0)}
    for key in a.z_post:
        assert torch.equal(a.z_post[key], b.z_post[key])
    # With 2 training steps a lock after step 1 freezes the constellation
    # and a lock after step 175 never fires: the checkpoints differ.
    c = drv.run_block(world, 2101, 1, **_SMALL).block
    d = drv.run_block(world, 2101, 175, **_SMALL).block
    assert not torch.equal(c.z_pre, d.z_pre)


def test_checkpoint_depends_on_seed_and_lock_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drv, "STEPS_TRAIN", 2)
    world = GaussianWorld(seed=0)
    a = drv.train_checkpoint(world, 2102, 50)
    torch.rand(10)  # global RNG state before the call must not matter
    b = drv.train_checkpoint(world, 2102, 50)
    for k, v in a.nerve_state.items():
        assert torch.equal(v, b.nerve_state[k])


def test_save_load_roundtrip(tmp_path: Path) -> None:
    blk = _block(2101, 25, np.random.default_rng(0))
    path = drv.block_path(tmp_path, 2101, 25)
    drv.save_block(blk, path)
    got = drv.load_block(path)
    assert (got.seed, got.lock, got.phase2_seed) == (2101, 25, 2101)
    assert torch.equal(got.z_pre, blk.z_pre) and torch.equal(got.probe_labels, blk.probe_labels)
    assert set(got.z_post) == set(blk.z_post)
    for key in blk.z_post:
        assert torch.equal(got.z_post[key], blk.z_post[key])


def test_planning_helpers_match_their_definitions() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=(6, 7)) + rng.normal(size=(6, 1)) + np.linspace(0, 1, 7)
    sd, df = drv.anova_residual_sd(y)
    resid = y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + y.mean()
    assert df == 30 and sd == pytest.approx(math.sqrt((resid**2).sum() / 30))
    assert drv.sd_ucl_df(1.0, 4) == pytest.approx(math.sqrt(4 / stats.chi2.ppf(0.2, 4)))
    # Section 7 values: 0/5 -> 0.275, 0/10 -> 0.149, 0/20 -> 0.077.
    assert drv.clopper_pearson_upper(0, 5) == pytest.approx(0.275, abs=5e-4)
    assert drv.clopper_pearson_upper(0, 10) == pytest.approx(0.149, abs=5e-4)
    assert drv.clopper_pearson_upper(0, 20) == pytest.approx(0.077, abs=5e-4)
    assert drv.clopper_pearson_upper(3, 3) == 1.0


def _block(seed: int, lock: int, rng: np.random.Generator, *, zero_pre: bool = False) -> BlockCodes:
    def z() -> torch.Tensor:
        return torch.from_numpy(rng.normal(size=(64, 24)).astype(np.float32))

    z_pre = torch.zeros(64, 24) if zero_pre else z()
    return BlockCodes(
        seed=seed,
        lock=lock,
        phase2_seed=seed,
        probe_labels=torch.zeros(64, dtype=torch.long),
        z_pre=z_pre,
        z_control=z(),
        z_post={(m, f): z() for m in MODALITIES for f in SNR_FLOORS},
    )


def _pilot_blocks(zero: frozenset[tuple[int, int]] = frozenset()) -> list[BlockCodes]:
    rng = np.random.default_rng(7)
    return [_block(s, lk, rng, zero_pre=(s, lk) in zero) for s in drv.PILOT_SEEDS for lk in LOCKS]


def test_pilot_quantities_fix_k_first_and_compute_planning_inputs() -> None:
    blocks = _pilot_blocks()
    out = drv.pilot_quantities(blocks)
    assert out["k_star"] == select_k_from_pilot([b.z_pre for b in blocks]).k
    assert out["n_complete"] == 10 and out["exclusion_model"] == "level"
    assert out["q_hat_cp80"] == pytest.approx(drv.clopper_pearson_upper(0, 10))
    assert out["sigma_hat_df"] == 54
    assert out["sigma_hat_ucl80"] == pytest.approx(drv.sd_ucl_df(out["sigma_hat_point"], 54))
    assert out["g_hat"] == pytest.approx(out["median_s_split_complete"] / out["sigma_hat_ucl80"])


def test_pilot_exclusions_drive_q_hat_and_the_exclusion_model() -> None:
    one = drv.pilot_quantities(_pilot_blocks(frozenset({(2101, 25)})))
    assert one["n_complete"] == 9 and one["exclusion_model"] == "level"
    assert one["q_hat_cp80"] == pytest.approx(drv.clopper_pearson_upper(1, 10))
    two = drv.pilot_quantities(_pilot_blocks(frozenset({(2101, 25), (2101, 175)})))
    assert two["exclusion_model"] == "seed"


def test_pilot_quantities_refuse_wrong_seeds() -> None:
    blocks = _pilot_blocks()
    with pytest.raises(ValueError, match="pilot"):
        drv.pilot_quantities(blocks[:-7])


def test_run_block_requires_deterministic_kernels() -> None:
    torch.use_deterministic_algorithms(False)  # the fixture restores it
    with pytest.raises(RuntimeError, match="deterministic"):
        drv.run_block(GaussianWorld(seed=0), 2101, 25, **_SMALL)
