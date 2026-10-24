"""Block evaluation and aggregation for the v0.8 subspace shift (sections 3.1, 6).

``evaluate_block`` turns the raw code matrices of one (seed, lock) block
(``subspace_protocol.BlockCodes``) into the block outcome ``Y(s, l)``:

* every matrix that enters ``S`` must pass the rank check of section 3.1
  (``Z_pre``, each ``Z_post``, the control end matrix, both 256-row halves
  of ``Z_pre``); non-finite codes are a *technical failure*
  (``TechnicalFailureError``: the whole seed is replaced from the reserve list,
  see ``SeedRoster``), every rank failure is an
  outcome-dependent exclusion and is **never** replaced;
* if ``Z_pre``, the control or a split half fails, all 15 cells are invalid;
* ``Y`` is the median of ``dS = S - C`` over valid cells, defined only if
  at least ``MIN_VALID_CELLS`` (12) of the 15 are valid.

``aggregate`` keeps *complete* seed blocks (``Y`` defined at all 7 levels),
counts invalid cells and undefined blocks per level with their
denominators, and flags a protocol failure below 70 % complete blocks
(PI decision 2026-10-24; was 80 %). It also keeps every block (``y_all``,
NaN where ``Y`` is undefined) for the selection safeguard: rank
exclusions are outcome-dependent, so an analysis of complete blocks
alone can be biased (Codex round-2 counterexample). ``adversarial_matrix``
imputes every undefined ``Y`` against the increasing hypothesis (+inf
at locks <= 100, -inf above), and ``verdict`` requires the Page test and
the magnitude guard to pass on both the complete blocks and the
adversarially imputed full set.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from bouba_sens.metrics.subspace import (
    K_DEFAULT,
    SubspaceRankError,
    eigengap_ratio,
    passes_rank_check,
    split_half_noise_floor,
    subspace_shift,
)
from bouba_sens.subspace_protocol import BlockCodes, CellKey

LOCKS: tuple[int, ...] = (25, 50, 75, 100, 125, 150, 175)
N_CELLS = 15  # 5 lesioned modalities x 3 SNR levels
MIN_VALID_CELLS = 12
MIN_COMPLETE_FRACTION = 0.7  # applied in integer form: 10 * complete >= 7 * N
ADVERSARIAL_SPLIT_LOCK = 100  # undefined Y at lock <= 100 -> +inf, above -> -inf
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


def evaluate_block(block: BlockCodes, *, k: int = K_DEFAULT, n_cells: int = N_CELLS) -> BlockResult:
    """Rank-check one (seed, lock) block and compute ``Y`` (``None`` if undefined).

    ``block`` is a ``subspace_protocol.BlockCodes``. Raises ``TechnicalFailureError``
    on any non-finite matrix and ``ValueError`` if the number of cells differs
    from ``n_cells``.
    """
    if len(block.z_post) != n_cells:
        raise ValueError(f"expected {n_cells} cells per block; got {len(block.z_post)}")
    for name, z in (("Z_pre", block.z_pre), ("control", block.z_control)):
        if not _all_finite(z):
            raise TechnicalFailureError(
                f"non-finite {name} at seed={block.seed}, lock={block.lock}"
            )
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
    seeds_all: list[int] = field(default_factory=list)
    y_all: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))  # NaN = undefined


def aggregate(
    blocks: Iterable[BlockResult],
    *,
    n_confirmatory: int,
    locks: Sequence[int] = LOCKS,
    expected_seeds: Sequence[int] | None = None,
) -> AggregateResult:
    """Keep complete seed blocks, count exclusions per level, test the 70 % rule.

    ``blocks`` must hold every (seed, lock) result of exactly
    ``n_confirmatory`` seeds, technical replacements already applied;
    rank-excluded seeds stay in the denominator (never replaced);
    ``expected_seeds`` (``SeedRoster.active``) rejects any other seed.
    """
    locks = tuple(locks)
    by_seed: dict[int, dict[int, BlockResult]] = {}
    for b in blocks:
        per_lock = by_seed.setdefault(b.seed, {})
        if b.lock in per_lock:
            raise ValueError(f"duplicate block result for seed {b.seed}, lock {b.lock}")
        per_lock[b.lock] = b
    if expected_seeds is not None and set(by_seed) != set(expected_seeds):
        raise ValueError(
            "seeds differ from the authorised roster: "
            f"extra {sorted(set(by_seed) - set(expected_seeds))}, "
            f"missing {sorted(set(expected_seeds) - set(by_seed))}"
        )
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
    splits = np.array(
        [[by_seed[s][lk].s_split for lk in locks] for s in complete], dtype=np.float64
    )
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
        protocol_failure=10 * n_complete < 7 * n_confirmatory,
        seeds_all=sorted(by_seed),
        y_all=np.array(
            [
                [np.nan if by_seed[s][lk].y is None else by_seed[s][lk].y for lk in locks]
                for s in sorted(by_seed)
            ],
            dtype=np.float64,
        ).reshape(len(by_seed), len(locks)),
    )


def adversarial_matrix(agg: AggregateResult) -> np.ndarray:
    """Every block, undefined ``Y`` imputed against an increasing trend.

    Within each block, undefined values at locks <= ``ADVERSARIAL_SPLIT_LOCK``
    take the highest ranks and those above it the lowest ranks; imputed
    values on the same side tie (mid-ranks in the Page test). They are
    written as finite numbers (block maximum + 1 and block minimum - 1;
    1 and -1 for a block with no defined value) so that the Page test,
    which works on ranks, accepts the matrix. Observed values are
    unchanged; with no exclusion this equals ``agg.y``.
    """
    imp = agg.y_all.copy()
    high = np.array([lk <= ADVERSARIAL_SPLIT_LOCK for lk in agg.locks])
    for row in imp:
        miss = np.isnan(row)
        if not miss.any():
            continue
        obs = row[~miss]
        top = float(obs.max()) + 1.0 if obs.size else 1.0
        bottom = float(obs.min()) - 1.0 if obs.size else -1.0
        row[miss & high] = top
        row[miss & ~high] = bottom
    return imp


def adversarial_endpoint_contrast(agg: AggregateResult) -> float:
    """Median over all blocks of ``Y(175) - Y(25)`` on the imputed matrix.

    An undefined endpoint makes the block's contrast -inf.
    """
    lo, hi = agg.locks.index(min(agg.locks)), agg.locks.index(max(agg.locks))
    diff = agg.y_all[:, hi] - agg.y_all[:, lo]
    diff[np.isnan(diff)] = -np.inf
    return float(np.median(diff))


class SeedRoster:
    """Authorised confirmatory seeds with blind, ordered technical replacement.

    ``fail(seed)`` replaces a technically failed active seed (also a
    previous replacement) by the next unused reserve seed, in order; every
    reserve is allocated at most once and failures accumulate across
    calls. A fifth technical failure (or an exhausted reserve) raises
    ``RunHaltedError`` without being absorbed. Rank exclusions never go
    through here.
    """

    def __init__(self, seeds: Sequence[int], reserve: Sequence[int]) -> None:
        if len(set(seeds) | set(reserve)) != len(seeds) + len(reserve):
            raise ValueError("seeds and reserve must be distinct")
        self.active: list[int] = list(seeds)
        self._reserve: list[int] = list(reserve)
        self.consumed: list[int] = []
        self.failed: list[int] = []

    @property
    def n_failures(self) -> int:
        return len(self.failed)

    def fail(self, seed: int) -> int:
        if seed not in self.active:
            raise ValueError(f"seed {seed} is not in the active roster")
        if self.n_failures + 1 > MAX_TECHNICAL_REPLACEMENTS or len(self.consumed) >= len(
            self._reserve
        ):
            raise RunHaltedError(
                f"technical failure #{self.n_failures + 1} (max {MAX_TECHNICAL_REPLACEMENTS} "
                "replacements, reserve of "
                f"{len(self._reserve)}); halt and file a deviation note"
            )
        replacement = self._reserve[len(self.consumed)]
        self.consumed.append(replacement)
        self.failed.append(seed)
        self.active[self.active.index(seed)] = replacement
        return replacement


def verdict(
    agg: AggregateResult, *, p_value: float, p_value_adversarial: float, alpha: float = 0.05
) -> str:
    """Ordered decision rule of section 6, with the selection safeguard.

    ``p_value`` is the Page permutation p-value on the complete blocks,
    ``p_value_adversarial`` the same test on ``adversarial_matrix(agg)``
    (all blocks). Returns ``protocol_failure``, ``not_demonstrated``
    (complete-block Page fails), ``not_robust_to_exclusions`` (complete
    blocks reject, the adversarial analysis does not), ``supported``
    (both Page tests and both guards pass) or ``detectable_sub_floor``.
    The magnitude guard is the (heuristic) strict inequality: median
    endpoint contrast greater than the median ``S_split`` over complete
    blocks and all levels; on the imputed set an undefined endpoint
    counts as -inf.
    """
    if agg.protocol_failure:
        return "protocol_failure"
    if not p_value < alpha:
        return "not_demonstrated"
    if not p_value_adversarial < alpha:
        return "not_robust_to_exclusions"
    lo, hi = agg.locks.index(min(agg.locks)), agg.locks.index(max(agg.locks))
    contrast = float(np.median(agg.y[:, hi] - agg.y[:, lo]))
    floor = float(np.median(agg.s_split))
    guard_ok = contrast > floor and adversarial_endpoint_contrast(agg) > floor
    return "supported" if guard_ok else "detectable_sub_floor"
