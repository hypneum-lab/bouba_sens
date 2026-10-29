"""v0.8 rev. C outcome on the sensory pathways (PI decision 2026-10-28).

Z is the concatenation of five per-modality blocks in ``MODALITIES`` order:
encoder representations (5 x 128, primary) or sender logits (5 x 64,
secondary). Every check is made on the **unnormalised** blocks:

* a block is valid if it is finite, its mean squared row norm is > 0 and
  its centred total variance is > 0 and >= ``REL_TOL`` (1e-6) x that
  norm;
* a matrix is valid if all five blocks are; a cell is valid if Z_pre,
  Z_control and Z_post(c) are; an invalid Z_control makes the block
  undefined; Y(s, l) = median over valid cells of d(c), defined with
  >= 12 valid cells (as before);
* d(c) = 1 - CKA_lin(N(Z_post(c)), N(Z_control)), where N divides each
  block by the square root of its own centred total variance, so that no
  modality dominates the CKA.

Drift floor (gate 4): F = 1 - CKA(N(Z_control), N(Z_control')) for the
second no-lesion control; R = median_c d(c) / F over the valid cells. R is
undefined if either control is invalid, if Y(s, l) is undefined (< 12
valid cells) or if F < ``FLOOR_MIN`` (1e-9); the pilot summary counts an
undefined R as failing (R = 0), never drops it.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from bouba_sens.metrics.cka_shift import linear_cka, total_variance
from bouba_sens.metrics.subspace_aggregate import (
    MIN_VALID_CELLS,
    N_CELLS,
    BlockResult,
    TechnicalFailureError,
)
from bouba_sens.sensory import MODALITIES
from bouba_sens.subspace_protocol import BlockCodes, CellKey

REL_TOL = 1e-6
FLOOR_MIN = 1e-9
ENCODER_DIMS: tuple[int, ...] = (128,) * len(MODALITIES)
LOGIT_DIMS: tuple[int, ...] = (64,) * len(MODALITIES)


def split_blocks(z: Tensor, dims: Sequence[int]) -> list[Tensor]:
    if z.shape[1] != sum(dims):
        raise ValueError(f"expected {sum(dims)} columns; got {z.shape[1]}")
    return list(torch.split(z.detach().double(), list(dims), dim=1))


def block_failures(z: Tensor, dims: Sequence[int]) -> list[str]:
    """Names of the modality blocks of ``z`` that fail the non-degeneracy check."""
    bad: list[str] = []
    for m, b in zip(MODALITIES, split_blocks(z, dims), strict=True):
        if not bool(torch.isfinite(b).all()):
            bad.append(m)
            continue
        msn = float((b**2).sum(1).mean())
        tv = total_variance(b)
        if not (msn > 0.0 and tv > 0.0 and tv >= REL_TOL * msn):
            bad.append(m)
    return bad


def normalise(z: Tensor, dims: Sequence[int]) -> Tensor:
    """Each block divided by the square root of its centred total variance."""
    parts = [b / total_variance(b) ** 0.5 for b in split_blocks(z, dims)]
    return torch.cat(parts, dim=1)


def distance(a: Tensor, b: Tensor, dims: Sequence[int]) -> float:
    return 1.0 - linear_cka(normalise(a, dims), normalise(b, dims))


def evaluate_block_revc(
    block: BlockCodes, dims: Sequence[int], *, n_cells: int = N_CELLS
) -> tuple[BlockResult, dict[str, int]]:
    """Y(s, l) and the per-modality failure counts of one block (module docstring)."""
    if len(block.z_post) != n_cells:
        raise ValueError(f"expected {n_cells} cells per block; got {len(block.z_post)}")
    for name, z in (("Z_pre", block.z_pre), ("control", block.z_control)):
        if not bool(torch.isfinite(z).all()):
            raise TechnicalFailureError(
                f"non-finite {name} at seed={block.seed}, lock={block.lock}"
            )
    fails: dict[str, int] = dict.fromkeys(MODALITIES, 0)
    base_bad = set(block_failures(block.z_pre, dims)) | set(block_failures(block.z_control, dims))
    for m in base_bad:
        fails[m] += 1
    keys = list(block.z_post)
    cell_valid: dict[CellKey, bool] = dict.fromkeys(keys, False)
    d: dict[CellKey, float] = {}
    for key in keys:
        bad = block_failures(block.z_post[key], dims)
        for m in bad:
            fails[m] += 1
        if not base_bad and not bad:
            cell_valid[key] = True
            d[key] = distance(block.z_post[key], block.z_control, dims)
    n_valid = sum(cell_valid.values())
    y = float(np.median(list(d.values()))) if n_valid >= MIN_VALID_CELLS else None
    res = BlockResult(
        seed=block.seed,
        lock=block.lock,
        n_cells=n_cells,
        n_valid=n_valid,
        y=y,
        s_split=None,
        cell_valid=cell_valid,
        delta_s=d,
    )
    return res, fails


def drift_floor(
    block: BlockCodes, res: BlockResult, dims: Sequence[int]
) -> dict[str, float | None]:
    """F and R of one block with a second control (module docstring)."""
    if block.z_control_alt is None:
        raise ValueError("block has no drift-floor control")
    if block_failures(block.z_control, dims) or block_failures(block.z_control_alt, dims):
        return {"F": None, "R": None}
    if res.y is None:  # Y undefined (< 12 valid cells): no ratio (Codex r8)
        return {"F": distance(block.z_control_alt, block.z_control, dims), "R": None}
    f = distance(block.z_control_alt, block.z_control, dims)
    vals = list(res.delta_s.values())
    if f < FLOOR_MIN or not vals:
        return {"F": f, "R": None}
    return {"F": f, "R": float(statistics.median(vals)) / f}
