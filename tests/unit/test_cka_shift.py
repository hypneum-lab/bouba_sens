"""v0.8 rev. B: 1 - CKA lesion-vs-control outcome and the eigengap k (PI 2026-10-25)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bouba_sens.metrics import cka_shift as ck
from bouba_sens.sensory import MODALITIES
from bouba_sens.subspace_protocol import SNR_FLOORS, BlockCodes


def _rand(n: int = 64, d: int = 20, seed: int = 0) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(seed).normal(size=(n, d)))


def test_linear_cka_properties() -> None:
    x = _rand()
    q, _ = torch.linalg.qr(_rand(20, 20, 1))
    assert ck.linear_cka(x, x) == pytest.approx(1.0)
    assert ck.linear_cka(x, 3.0 * x @ q) == pytest.approx(1.0)  # rotation + isotropic scale
    assert ck.linear_cka(x, _rand(seed=2)) < 0.6
    with pytest.raises(ValueError):
        ck.linear_cka(x, torch.zeros(64, 20))


def _block(post_scale: float = 1.0, degenerate: int = 0, control_zero: bool = False) -> BlockCodes:
    ctrl = torch.zeros(64, 20) if control_zero else _rand(seed=3)
    post = {}
    for i, (m, f) in enumerate((m, f) for m in MODALITIES for f in SNR_FLOORS):
        post[(m, f)] = (
            torch.zeros(64, 20) if i < degenerate else ctrl + post_scale * _rand(seed=10 + i)
        )
    return BlockCodes(9001, 25, 9001, torch.zeros(64, dtype=torch.long), _rand(seed=4), ctrl, post)


def test_block_outcome_and_validity() -> None:
    small, large = ck.evaluate_block_cka(_block(0.1)), ck.evaluate_block_cka(_block(2.0))
    assert small.y is not None and large.y is not None and 0 <= small.y < large.y <= 1
    assert small.n_valid == 15 and small.s_split is None
    r = ck.evaluate_block_cka(_block(degenerate=3))
    assert r.n_valid == 12 and r.y is not None
    assert ck.evaluate_block_cka(_block(degenerate=4)).y is None
    assert ck.evaluate_block_cka(_block(control_zero=True)).n_valid == 0


def test_eigengap_k_finds_the_cliff() -> None:
    rng = np.random.default_rng(5)
    basis = np.linalg.qr(rng.normal(size=(60, 60)))[0]
    lam = np.r_[np.full(14, 1.0) * (1 + 0.02 * rng.random(14)), np.full(46, 1e-3)]
    z = torch.from_numpy(rng.normal(size=(500, 60)) * np.sqrt(lam) @ basis.T)
    k, ratio = ck.eigengap_k(z)
    assert k == 14 and ratio > 100
    kg = ck.k_gap_from_pilot([z, z])
    assert kg.k == 14 and len(kg.per_matrix) == 2


def test_k_gap_skips_degenerate_matrices() -> None:
    flat = torch.full((64, 20), 0.3, dtype=torch.float64) + 1e-9 * _rand()
    assert ck.k_gap_from_pilot([flat, flat]).k == 0
    posts = {(m, f): flat for m in MODALITIES for f in SNR_FLOORS}
    blk = BlockCodes(9001, 25, 9001, torch.zeros(64, dtype=torch.long), flat, flat, posts)
    assert ck.evaluate_block_cka(blk).y is None
