"""Rank checks, 12-of-15 rule, complete blocks, protocol failure (amendment v0.8, 3.1, 6)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from bouba_sens.metrics.subspace import split_half_noise_floor, subspace_shift
from bouba_sens.metrics.subspace_aggregate import (
    LOCKS,
    MIN_VALID_CELLS,
    N_CELLS,
    BlockResult,
    RunHaltedError,
    TechnicalFailureError,
    aggregate,
    evaluate_block,
    replace_technical_failures,
    verdict,
)
from bouba_sens.sensory import MODALITIES
from bouba_sens.subspace_protocol import SNR_FLOORS, BlockCodes

D = 20
K = 3
KEYS = [(m, f) for m in MODALITIES for f in SNR_FLOORS]


def _codes(rank: int, n: int = 64, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, rank, generator=g, dtype=torch.float64)
    b = torch.randn(rank, D, generator=g, dtype=torch.float64)
    return a @ b


def _block(
    *,
    bad_cells: int = 0,
    pre_rank: int = 8,
    ctrl_rank: int = 8,
    seed: int = 2001,
    lock: int = 25,
    n: int = 64,
) -> BlockCodes:
    z_post = {
        key: _codes(K - 1 if i < bad_cells else 8, n=n, seed=100 + i) for i, key in enumerate(KEYS)
    }
    return BlockCodes(
        seed=seed,
        lock=lock,
        phase2_seed=seed,
        probe_labels=torch.zeros(n, dtype=torch.long),
        z_pre=_codes(pre_rank, n=n, seed=1),
        z_control=_codes(ctrl_rank, n=n, seed=2),
        z_post=z_post,
    )


def test_frozen_constants() -> None:
    assert N_CELLS == 15
    assert MIN_VALID_CELLS == 12
    assert LOCKS == (25, 50, 75, 100, 125, 150, 175)


def test_all_valid_block() -> None:
    b = _block()
    r = evaluate_block(b, k=K)
    assert (r.seed, r.lock, r.n_cells, r.n_valid) == (2001, 25, 15, 15)
    c = subspace_shift(b.z_pre, b.z_control, K)
    expected = [subspace_shift(b.z_pre, b.z_post[key], K) - c for key in KEYS]
    assert r.y == pytest.approx(float(np.median(expected)))
    assert r.s_split == pytest.approx(split_half_noise_floor(b.z_pre, K, seed=2001))
    assert all(r.cell_valid.values())
    assert r.delta_s[KEYS[0]] == pytest.approx(expected[0])


@pytest.mark.parametrize(("bad", "defined"), [(1, True), (3, True), (4, False), (15, False)])
def test_twelve_of_fifteen_rule(bad: int, defined: bool) -> None:
    r = evaluate_block(_block(bad_cells=bad), k=K)
    assert r.n_valid == 15 - bad
    assert (r.y is not None) is defined
    assert sum(not v for v in r.cell_valid.values()) == bad


def test_median_is_over_valid_cells_only() -> None:
    b = _block(bad_cells=1)
    r = evaluate_block(b, k=K)
    c = subspace_shift(b.z_pre, b.z_control, K)
    vals = [subspace_shift(b.z_pre, b.z_post[key], K) - c for key in KEYS[1:]]
    assert r.y == pytest.approx(float(np.median(vals)))  # 14 values: mean of the middle two


def test_rank_deficient_pre_invalidates_all_cells() -> None:
    r = evaluate_block(_block(pre_rank=K - 1), k=K)
    assert r.n_valid == 0 and r.y is None and r.s_split is None


def test_rank_deficient_control_invalidates_all_cells() -> None:
    r = evaluate_block(_block(ctrl_rank=K - 1), k=K)
    assert r.n_valid == 0 and r.y is None


def test_constant_control_is_rank_zero_exclusion_not_crash() -> None:
    b = _block()
    b.z_control = torch.ones_like(b.z_control)
    assert evaluate_block(b, k=K).y is None


def test_split_half_rank_failure_invalidates_all_cells() -> None:
    # 4 rows: centred rank 3 passes k=3, but each 2-row half has rank <= 1.
    b = _block(n=4)
    assert evaluate_block(b, k=K).n_valid == 0


@pytest.mark.parametrize("where", ["pre", "control", "cell"])
def test_non_finite_is_a_technical_failure(where: str) -> None:
    b = _block()
    if where == "pre":
        b.z_pre[0, 0] = float("nan")
    elif where == "control":
        b.z_control[0, 0] = float("inf")
    else:
        b.z_post[KEYS[7]][0, 0] = float("nan")
    with pytest.raises(TechnicalFailureError):
        evaluate_block(b, k=K)


def test_wrong_cell_count_is_rejected() -> None:
    b = _block()
    del b.z_post[KEYS[0]]
    with pytest.raises(ValueError, match="cells"):
        evaluate_block(b, k=K)


def test_eigengap_reported() -> None:
    r = evaluate_block(_block(), k=K)
    assert math.isfinite(r.eigengap_pre) or math.isinf(r.eigengap_pre)
    assert set(r.eigengap_post) == set(KEYS)


# ---------------------------------------------------------------- aggregation


def _res(seed: int, lock: int, y: float | None, n_valid: int = 15) -> BlockResult:
    return BlockResult(
        seed=seed,
        lock=lock,
        n_cells=15,
        n_valid=n_valid if y is not None else min(n_valid, 11),
        y=y,
        s_split=0.1 if y is not None else None,
        cell_valid={},
        delta_s={},
        eigengap_pre=2.0,
        eigengap_post={},
    )


def _grid(n: int, *, undefined: dict[int, list[int]] | None = None) -> list[BlockResult]:
    undefined = undefined or {}
    out = []
    for si in range(n):
        for j, lock in enumerate(LOCKS):
            bad = lock in undefined.get(si, [])
            out.append(_res(2001 + si, lock, None if bad else 0.1 * j + 0.01 * si))
    return out


def test_all_complete() -> None:
    agg = aggregate(_grid(20), n_confirmatory=20)
    assert agg.y.shape == (20, 7)
    assert not agg.protocol_failure
    assert agg.seeds_complete == list(range(2001, 2021))
    assert agg.y[3, 2] == pytest.approx(0.2 + 0.03)  # columns follow LOCKS order
    assert agg.complete_fraction == 1.0


def test_incomplete_blocks_dropped_and_counted_per_level() -> None:
    agg = aggregate(_grid(20, undefined={0: [25], 1: [175, 25]}), n_confirmatory=20)
    assert agg.y.shape == (18, 7)
    assert 2001 not in agg.seeds_complete and 2002 not in agg.seeds_complete
    by_lock = {c.lock: c for c in agg.per_level}
    assert by_lock[25].undefined_blocks == 2 and by_lock[25].total_blocks == 20
    assert by_lock[175].undefined_blocks == 1
    assert by_lock[50].undefined_blocks == 0
    assert by_lock[25].total_cells == 20 * 15


def test_invalid_cells_counted_with_denominator() -> None:
    blocks = _grid(20)
    blocks[0] = _res(2001, 25, 0.5, n_valid=13)
    agg = aggregate(blocks, n_confirmatory=20)
    assert {c.lock: c.invalid_cells for c in agg.per_level}[25] == 2


@pytest.mark.parametrize(
    ("n", "complete", "failure"),
    [(20, 16, False), (20, 15, True), (30, 24, False), (30, 23, True)],
)
def test_protocol_failure_threshold(n: int, complete: int, failure: bool) -> None:
    undefined = {i: [25] for i in range(n - complete)}
    agg = aggregate(_grid(n, undefined=undefined), n_confirmatory=n)
    assert len(agg.seeds_complete) == complete
    assert agg.protocol_failure is failure


def test_rank_deficient_seeds_are_never_replaced() -> None:
    # Aggregation takes exactly N confirmatory seeds: no reserve seed can enter.
    with pytest.raises(ValueError, match="confirmatory"):
        aggregate(_grid(21), n_confirmatory=20)
    with pytest.raises(ValueError, match="confirmatory"):
        aggregate(_grid(19), n_confirmatory=20)


def test_missing_level_for_a_seed_is_an_error() -> None:
    blocks = [b for b in _grid(20) if not (b.seed == 2005 and b.lock == 100)]
    with pytest.raises(ValueError, match="2005"):
        aggregate(blocks, n_confirmatory=20)


def test_replace_technical_failures_in_reserve_order() -> None:
    seeds = list(range(2001, 2021))
    reserve = [2021, 2022, 2023, 2024]
    out = replace_technical_failures(seeds, reserve, failed=[2003, 2010])
    assert out[2] == 2021 and out[9] == 2022
    assert len(out) == 20 and 2003 not in out
    assert replace_technical_failures(seeds, reserve, failed=[]) == seeds


def test_more_than_four_technical_replacements_halt() -> None:
    seeds = list(range(2001, 2021))
    with pytest.raises(RunHaltedError):
        replace_technical_failures(seeds, [2021, 2022, 2023, 2024, 2025], failed=seeds[:5])


# -------------------------------------------------------------------- verdict


def _verdict_agg(endpoint_gain: float, s_split: float = 0.1, drop: int = 0):
    blocks = []
    for si in range(20):
        for j, lock in enumerate(LOCKS):
            y = None if si < drop and lock == 25 else endpoint_gain * j / 6
            r = _res(2001 + si, lock, y)
            blocks.append(replace(r, s_split=None if y is None else s_split))
    return aggregate(blocks, n_confirmatory=20)


def test_verdict_protocol_failure_has_priority() -> None:
    assert verdict(_verdict_agg(1.0, drop=5), p_value=0.001) == "protocol_failure"


def test_verdict_supported() -> None:
    assert verdict(_verdict_agg(0.5), p_value=0.049) == "supported"


def test_verdict_detectable_but_sub_floor() -> None:
    assert verdict(_verdict_agg(0.05), p_value=0.001) == "detectable_sub_floor"
    # strict inequality: contrast equal to the floor does not pass the guard
    assert verdict(_verdict_agg(0.5, s_split=0.5), p_value=0.001) == "detectable_sub_floor"


def test_verdict_not_demonstrated() -> None:
    assert verdict(_verdict_agg(0.5), p_value=0.05) == "not_demonstrated"
