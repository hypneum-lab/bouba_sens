"""Block evaluation and aggregation for the v0.8 subspace shift (sections 3.1, 6).

``evaluate_block`` turns the raw code matrices of one (seed, lock) block
(``subspace_protocol.BlockCodes``) into the block outcome ``Y(s, l)``:

* every matrix that enters ``S`` must pass the rank check of section 3.1
  (``Z_pre``, each ``Z_post``, the control end matrix, both 256-row halves
  of ``Z_pre``); non-finite codes are a *technical failure*
  (``TechnicalFailureError``: the whole seed is replaced from the reserve list,
  see ``replace_technical_failures``), every rank failure is an
  outcome-dependent exclusion and is **never** replaced;
* if ``Z_pre``, the control or a split half fails, all 15 cells are invalid;
* ``Y`` is the median of ``dS = S - C`` over valid cells, defined only if
  at least ``MIN_VALID_CELLS`` (12) of the 15 are valid.

``aggregate`` keeps *complete* seed blocks (``Y`` defined at all 7 levels),
counts invalid cells and undefined blocks per level with their
denominators, and flags a protocol failure below 80 % complete blocks.
``verdict`` applies the ordered decision rule of section 6.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from bouba_sens.subspace_protocol import BlockCodes, CellKey
from bouba_sens.metrics.subspace import (
    K_DEFAULT,
    SubspaceRankError,
    eigengap_ratio,
    passes_rank_check,
    split_half_noise_floor,
    subspace_shift,
)

LOCKS: tuple[int, ...] = (25, 50, 75, 100, 125, 150, 175)
N_CELLS = 15  # 5 lesioned modalities x 3 SNR levels
MIN_VALID_CELLS = 12
MIN_COMPLETE_FRACTION = 0.8  # applied in integer form: 5 * complete >= 4 * N
MAX_TECHNICAL_REPLACEMENTS = 4


class TechnicalFailureError(ValueError):
    """Non-finite codes or a crash: the seed is replaced from the reserve list."""


class RunHaltedError(RuntimeError):
    """More than ``MAX_TECHNICAL_REPLACEMENTS`` replacements: halt, file a deviation note."""


@dataclass(frozen=True)
class BlockResult:
    seed: int
    lock: int
    n_cells: int
    n_valid: int
    y: float | None
    s_split: float | None
    cell_valid: dict[CellKey, bool] = field(default_factory=dict)
    delta_s: dict[CellKey, float] = field(default_factory=dict)
    eigengap_pre: float = math.nan
    eigengap_post: dict[CellKey, float] = field(default_factory=dict)


def _all_finite(z: torch.Tensor) -> bool:
    return bool(torch.isfinite(z.detach()).all())


def evaluate_block(
    block: BlockCodes, *, k: int = K_DEFAULT, n_cells: int = N_CELLS
) -> BlockResult:
    """Rank-check one (seed, lock) block and compute ``Y`` (``None`` if undefined).

    ``block`` is a ``subspace_protocol.BlockCodes``. Raises ``TechnicalFailureError``
    on any non-finite matrix and ``ValueError`` if the number of cells differs
    from ``n_cells``.
    """
    if len(block.z_post) != n_cells:
        raise ValueError(f"expected {n_cells} cells per block; got {len(block.z_post)}")
    for name, z in (("Z_pre", block.z_pre), ("control", block.z_control)):
        if not _all_finite(z):
            raise TechnicalFailureError(f"non-finite {name} at seed={block.seed}, lock={block.lock}")
    for key, z in block.z_post.items():
        if not _all_finite(z):
            raise TechnicalFailureError(
                f"non-finite Z_post {key} at seed={block.seed}, lock={block.lock}"
            )

    keys = list(block.z_post)
    shared_ok = passes_rank_check(block.z_pre, k) and passes_rank_check(block.z_control, k)
    s_split: float | None = None
    if shared_ok:
        try:
            s_split = split_half_noise_floor(block.z_pre, k, seed=block.seed)
        except SubspaceRankError:
            shared_ok = False
    cell_valid: dict[CellKey, bool] = dict.fromkeys(keys, False)
    delta_s: dict[CellKey, float] = {}
    if shared_ok:
        control_shift = subspace_shift(block.z_pre, block.z_control, k)
        for key in keys:
            if passes_rank_check(block.z_post[key], k):
                cell_valid[key] = True
                delta_s[key] = subspace_shift(block.z_pre, block.z_post[key], k) - control_shift
    n_valid = sum(cell_valid.values())
    y = float(np.median(list(delta_s.values()))) if n_valid >= MIN_VALID_CELLS else None
    return BlockResult(
        seed=block.seed,
        lock=block.lock,
        n_cells=n_cells,
        n_valid=n_valid,
        y=y,
        s_split=s_split,
        cell_valid=cell_valid,
        delta_s=delta_s,
        eigengap_pre=eigengap_ratio(block.z_pre, k),
        eigengap_post={key: eigengap_ratio(block.z_post[key], k) for key in keys},
    )


@dataclass(frozen=True)
class LevelCounts:
    lock: int
    invalid_cells: int
    total_cells: int
    undefined_blocks: int
    total_blocks: int


@dataclass(frozen=True)
class AggregateResult:
    n_confirmatory: int
    seeds_complete: list[int]
    y: np.ndarray  # (complete seeds, len(locks)), columns in `locks` order
    s_split: np.ndarray  # same shape; shared by the 15 cells of a block
    locks: tuple[int, ...]
    per_level: list[LevelCounts]
    complete_fraction: float
    protocol_failure: bool


def aggregate(
    blocks: Iterable[BlockResult],
    *,
    n_confirmatory: int,
    locks: Sequence[int] = LOCKS,
) -> AggregateResult:
    """Keep complete seed blocks, count exclusions per level, test the 80 % rule.

    ``blocks`` must hold every (seed, lock) result of exactly
    ``n_confirmatory`` seeds, technical replacements already applied;
    rank-excluded seeds stay in the denominator (never replaced).
    """
    locks = tuple(locks)
    by_seed: dict[int, dict[int, BlockResult]] = {}
    for b in blocks:
        by_seed.setdefault(b.seed, {})[b.lock] = b
    if len(by_seed) != n_confirmatory:
        raise ValueError(
            f"expected {n_confirmatory} confirmatory seeds (no replacement of rank exclusions); "
            f"got {len(by_seed)}"
        )
    for seed, per_lock in by_seed.items():
        missing = [lk for lk in locks if lk not in per_lock]
        if missing:
            raise ValueError(f"seed {seed} has no block result for locks {missing}")

    complete = [s for s in sorted(by_seed) if all(by_seed[s][lk].y is not None for lk in locks)]
    y = np.array([[by_seed[s][lk].y for lk in locks] for s in complete], dtype=np.float64)
    splits = np.array([[by_seed[s][lk].s_split for lk in locks] for s in complete], dtype=np.float64)
    per_level = []
    for lk in locks:
        results = [by_seed[s][lk] for s in by_seed]
        per_level.append(
            LevelCounts(
                lock=lk,
                invalid_cells=sum(r.n_cells - r.n_valid for r in results),
                total_cells=sum(r.n_cells for r in results),
                undefined_blocks=sum(r.y is None for r in results),
                total_blocks=len(results),
            )
        )
    n_complete = len(complete)
    return AggregateResult(
        n_confirmatory=n_confirmatory,
        seeds_complete=complete,
        y=y.reshape(n_complete, len(locks)),
        s_split=splits.reshape(n_complete, len(locks)),
        locks=locks,
        per_level=per_level,
        complete_fraction=n_complete / n_confirmatory,
        protocol_failure=5 * n_complete < 4 * n_confirmatory,
    )


def replace_technical_failures(
    seeds: Sequence[int], reserve: Sequence[int], failed: Sequence[int]
) -> list[int]:
    """Replace technically failed seeds by the next reserve seeds, in order.

    Position is preserved. More than ``MAX_TECHNICAL_REPLACEMENTS``
    replacements (or an exhausted reserve) raises ``RunHaltedError``.
    """
    failed_set = set(failed)
    n_fail = sum(s in failed_set for s in seeds)
    if n_fail > MAX_TECHNICAL_REPLACEMENTS or n_fail > len(reserve):
        raise RunHaltedError(
            f"{n_fail} technical replacements needed (max {MAX_TECHNICAL_REPLACEMENTS}); "
            "halt and file a deviation note"
        )
    spare = iter(reserve)
    return [next(spare) if s in failed_set else s for s in seeds]


def verdict(agg: AggregateResult, *, p_value: float, alpha: float = 0.05) -> str:
    """Ordered decision rule of section 6.

    Returns ``protocol_failure``, ``supported``, ``detectable_sub_floor`` or
    ``not_demonstrated``. The magnitude guard is the (heuristic) strict
    inequality: median over complete blocks of ``Y(175) - Y(25)`` greater than
    the median ``S_split`` over complete blocks and all levels.
    """
    if agg.protocol_failure:
        return "protocol_failure"
    if not p_value < alpha:
        return "not_demonstrated"
    lo, hi = agg.locks.index(min(agg.locks)), agg.locks.index(max(agg.locks))
    contrast = float(np.median(agg.y[:, hi] - agg.y[:, lo]))
    floor = float(np.median(agg.s_split))
    return "supported" if contrast > floor else "detectable_sub_floor"
