"""Exploratory decomposition of the B-3 v2 asymmetry (not preregistered, ADR-0023).

Splits the per-trio matrix P[q, m] (rows = query, columns = lesioned) of the
v2 probe into its query and lesion main effects, lists the pairs attaining
Me6, and tests the antisymmetric residual left after removing the additive
query effect against the same sign-flip null (2 000 replicates, seed 0;
block 1 Gaussian, 8 mock as preregistered for the main reading).

Usage (from the repository root):
    uv run python scripts/b3_probe_v2_decompose.py GAUSSIAN_GRID MOCK_GRID OUT_JSON
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from me6_paired_null import (
    _IU,
    _JU,
    MODALITIES,
    PAIR_NAMES,
    discover_trios,
    pair_differences,
    stack_successes,
)

REPLICATES = 2_000


def residual_max(upper: np.ndarray) -> np.ndarray:
    """Per-trio max |residual| of P - P.T after removing its additive query part.

    `upper` is `(K, 10)` (pairs a < b). The antisymmetric matrix A has an
    additive part g_q - g_m with g = row mean of A; the residual is
    A - (g_q - g_m).
    """
    full = np.zeros((upper.shape[0], 5, 5))
    full[:, _IU, _JU] = upper
    full[:, _JU, _IU] = -upper
    g = full.mean(2, keepdims=True)
    resid = full - (g - g.transpose(0, 2, 1))
    return np.abs(resid[:, _IU, _JU]).max(-1)


def decompose(root: Path, *, block: int) -> dict[str, object]:
    trios, _ = discover_trios(root, successes_file="per_query_successes_v2.json")
    successes = stack_successes(trios)
    p = successes.astype(float).mean(-1)  # (K, q, m)
    a = p - p.transpose(0, 2, 1)
    argmax = Counter(PAIR_NAMES[int(np.abs(a[k][_IU, _JU]).argmax())] for k in range(len(trios)))
    d = pair_differences(successes)  # (K, 10, n)
    observed = float(np.median(residual_max(d.mean(-1))))
    rng = np.random.default_rng(0)
    n = d.shape[-1]
    n_blocks = -(-n // block)
    medians = np.empty(REPLICATES)
    for r in range(REPLICATES):
        signs = rng.integers(0, 2, size=(d.shape[0], 10, n_blocks)) * 2 - 1
        signs = np.repeat(signs, block, -1)[..., :n]
        medians[r] = np.median(residual_max((signs * d).mean(-1)))
    return {
        "mean_P_matrix_rows_query_cols_lesioned": np.round(p.mean(0), 3).tolist(),
        "row_means_query": dict(zip(MODALITIES, np.round(p.mean((0, 2)), 3).tolist(), strict=True)),
        "col_means_lesioned": dict(
            zip(MODALITIES, np.round(p.mean((0, 1)), 3).tolist(), strict=True)
        ),
        "argmax_pair_counts": dict(argmax.most_common()),
        "residual_after_row_effect": {
            "observed_median": observed,
            "null_mean": float(medians.mean()),
            "null_p99": float(np.quantile(medians, 0.99)),
            "p": float(((medians >= observed).sum() + 1) / (REPLICATES + 1)),
            "block": block,
            "replicates": REPLICATES,
        },
    }


def main() -> int:
    gaussian, mock, out = (Path(v) for v in sys.argv[1:4])
    result = {"gaussian": decompose(gaussian, block=1), "mock": decompose(mock, block=8)}
    out.write_text(json.dumps(result, indent=2))
    for world, body in result.items():
        print(world, json.dumps(body["residual_after_row_effect"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
