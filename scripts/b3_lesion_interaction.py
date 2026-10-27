"""B-3 lesion x query interaction test (preregistration 2026-10-26).

Implements `docs/prereg/2026-10-26-b3-lesion-interaction.md` on a grid written
by `scripts/run_grid_probe_v2.sh` (probe v2 per-item outcomes).

Primary test, per world:

- per trio, `P[q, m]` = held-out accuracy of the v2 probe on query q for the
  model lesioned on m; only the 20 off-diagonal cells are used;
- least-squares additive fit `mu + a_q + b_m` of those 20 cells, residual
  matrix `R` (11 degrees of freedom);
- per seed, `Rbar_s` = mean of its trios' residual matrices;
- statistic `S = sum_{q != m} (mean_s Rbar_s[q, m])**2`;
- exact seed-level sign-flip null: all 2**n_seeds patterns,
  p = #{S_null >= S_obs} / 2**n_seeds.

Secondary check (does not enter the decision): within each (trio, q) the four
off-diagonal outcome series are permuted across lesioned modalities by blocks
of `block` consecutive items (10 000 permutations, seed 25102026); statistic
`T` = mean over trios of sum_q sum_{m != q} (P[q, m] - mean_m P[q, m])**2,
i.e. H0 "no lesion effect on off-diagonal decodability".

Usage:
    uv run python scripts/b3_lesion_interaction.py --root GRID --world gaussian --block 1 \\
        --out reports/b3_lesion_interaction_gaussian.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3_probe_v2 import positive_control
from me6_paired_null import N_MODALITIES, discover_trios

SCHEMA = "bouba_sens.b3_lesion_interaction.v1"
SUCCESSES_V2 = "per_query_successes_v2.json"
EXPECTED_TRIOS = 60
ALPHA_PRESENT = 0.01
ALPHA_COMPATIBLE = 0.05
SECONDARY_REPLICATES = 10_000
SECONDARY_SEED = 25_102_026
MAX_EXACT_SEEDS = 16

_OFF = [(q, m) for q in range(N_MODALITIES) for m in range(N_MODALITIES) if q != m]


def _design() -> np.ndarray:
    """(20, 11) design of mu + a_q + b_m for the off-diagonal cells (full dummies)."""
    x = np.zeros((len(_OFF), 1 + 2 * N_MODALITIES))
    for r, (q, m) in enumerate(_OFF):
        x[r, 0] = 1.0
        x[r, 1 + q] = 1.0
        x[r, 1 + N_MODALITIES + m] = 1.0
    return x


_X = _design()
_HAT = _X @ np.linalg.pinv(_X)  # projection onto the additive space


def interaction_residual(p: np.ndarray) -> np.ndarray:
    """Residual of the additive fit on the off-diagonal cells; `p` is `(..., 5, 5)`.

    Returns `(..., 20)` in the order of `_OFF`. The residual space has
    20 - 9 = 11 dimensions.
    """
    y = np.stack([p[..., q, m] for q, m in _OFF], axis=-1)
    return y - y @ _HAT.T


def seed_means(residuals: np.ndarray, seeds: list[int]) -> tuple[np.ndarray, list[int]]:
    """Mean residual per seed: `(n_seeds, 20)` and the sorted seed list."""
    groups: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(seeds):
        groups[s].append(i)
    order = sorted(groups)
    return np.stack([residuals[groups[s]].mean(axis=0) for s in order]), order


def statistic(rbar: np.ndarray) -> float:
    """S = sum over cells of the squared mean (over seeds) residual."""
    return float((rbar.mean(axis=0) ** 2).sum())


def exact_sign_flip(rbar: np.ndarray) -> dict[str, Any]:
    """Exact seed-level sign-flip null of S over all 2**n_seeds patterns."""
    n = rbar.shape[0]
    if n > MAX_EXACT_SEEDS:
        raise ValueError(f"exact enumeration limited to {MAX_EXACT_SEEDS} seeds; got {n}")
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))  # (2**n, n)
    null = ((signs @ rbar / n) ** 2).sum(axis=1)
    observed = statistic(rbar)
    count = int((null >= observed - 1e-15).sum())
    return {
        "observed": observed,
        "patterns": int(signs.shape[0]),
        "null_mean": float(null.mean()),
        "null_p99": float(np.quantile(null, 0.99, method="inverted_cdf")),
        "count_ge_observed": count,
        "p_value": count / signs.shape[0],
    }


def _lesion_effect(p: np.ndarray) -> float:
    """Mean over trios of sum_q sum_{m != q} (P[q, m] - mean_m P[q, m])**2."""
    total = 0.0
    k = p.shape[0]
    for q in range(N_MODALITIES):
        cols = [m for m in range(N_MODALITIES) if m != q]
        row = p[:, q, cols]
        total += float(((row - row.mean(axis=1, keepdims=True)) ** 2).sum())
    return total / k


def block_permutation(
    successes: np.ndarray, *, block: int, replicates: int, seed: int
) -> dict[str, Any]:
    """Secondary: permute off-diagonal series across lesioned columns by blocks."""
    k, _, _, n = successes.shape
    observed = _lesion_effect(successes.mean(-1))
    n_blocks = -(-n // block)
    rng = np.random.default_rng(seed)
    null = np.empty(replicates)
    for r in range(replicates):
        perm = successes.copy()
        for q in range(N_MODALITIES):
            cols = np.array([m for m in range(N_MODALITIES) if m != q])
            keys = rng.random((k, len(cols), n_blocks)).argsort(axis=1)  # (k, 4, blocks)
            idx = np.repeat(keys, block, axis=-1)[..., :n]  # (k, 4, n)
            src = successes[:, q][:, cols]  # (k, 4, n)
            perm[:, q, cols] = np.take_along_axis(src, idx, axis=1)
        null[r] = _lesion_effect(perm.mean(-1))
    count = int((null >= observed - 1e-15).sum())
    return {
        "observed": observed,
        "block": block,
        "replicates": replicates,
        "seed": seed,
        "null_mean": float(null.mean()),
        "null_p99": float(np.quantile(null, 0.99, method="inverted_cdf")),
        "p_value": (count + 1) / (replicates + 1),
    }


def verdict(*, learns: bool, control_passes: bool, complete: bool, p_value: float) -> str:
    """Preregistered decision rule (§7)."""
    if not learns:
        return "inconclusive: no learning network (pilot precondition failed)"
    if not control_passes:
        return "inconclusive: positive control failed"
    if not complete:
        return "inconclusive: incomplete grid"
    if p_value < ALPHA_PRESENT:
        return "lesion x query interaction present"
    if p_value >= ALPHA_COMPATIBLE:
        return "compatible with no interaction"
    return "inconclusive: 0.01 <= p < 0.05"


def report(
    root: Path,
    *,
    world: str,
    block: int,
    learns: bool,
    replicates: int = SECONDARY_REPLICATES,
) -> dict[str, Any]:
    control = positive_control(root)
    trios, incomplete = discover_trios(root, successes_file=SUCCESSES_V2)
    if not trios:
        raise ValueError(f"no complete trio under {root}")
    successes = np.stack([t.successes for t in trios]).astype(np.float64)  # (K, 5, 5, n)
    p = successes.mean(-1)
    residuals = interaction_residual(p)  # (K, 20)
    rbar, seed_list = seed_means(residuals, [int(t.seed) for t in trios])
    primary = exact_sign_flip(rbar)
    secondary = block_permutation(
        successes, block=block, replicates=replicates, seed=SECONDARY_SEED
    )
    complete = len(trios) >= EXPECTED_TRIOS and not incomplete
    mean_resid = rbar.mean(axis=0)
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "world": world,
        "prereg": "docs/prereg/2026-10-26-b3-lesion-interaction.md",
        "precondition_learns": learns,
        "positive_control": control,
        "trios": len(trios),
        "seeds": seed_list,
        "incomplete_trios": incomplete,
        "complete": complete,
        "primary": primary,
        "mean_residual_per_cell": {
            f"q{q}_m{m}": float(v) for (q, m), v in zip(_OFF, mean_resid, strict=True)
        },
        "mean_P_matrix_rows_query_cols_lesioned": p.mean(axis=0).round(4).tolist(),
        "secondary_block_permutation": secondary,
        "verdict": verdict(
            learns=learns,
            control_passes=bool(control["passes"]),
            complete=complete,
            p_value=float(primary["p_value"]),
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument(
        "--pilot-learns",
        choices=["yes", "no"],
        required=True,
        help="outcome of the preregistered pilot precondition for this world",
    )
    parser.add_argument("--replicates", type=int, default=SECONDARY_REPLICATES)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = report(
        args.root,
        world=args.world,
        block=args.block,
        learns=args.pilot_learns == "yes",
        replicates=args.replicates,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("world", "trios", "complete", "primary", "verdict")}
            | {"secondary_p": result["secondary_block_permutation"]["p_value"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
