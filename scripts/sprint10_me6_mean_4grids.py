"""Sprint 10 ADR-0017 verification — re-aggregate 4 v0.X grids under Me6_mean.

Reads existing per-cell `per_query_me1.json` files (no cell re-runs needed),
builds the 5x5 perf matrix per (seed, timing, snr) trio, and reports
mid-rank percentile of the pre-registered partition under both the original
max-statistic and the candidate mean-statistic.

The output reproduces ADR-0017's empirical verdict table (4/4 grids,
max vs mean comparison).

DEPENDENCIES — IMPORTANT
========================

This script requires the partition-control machinery, which as of
2026-04-24 lives **off-main** :

1. ``bouba_sens.metrics.partitions`` (declares the 5 modalities,
   ``PERCEPTIVE_PROPRIOCEPTIVE``, and ``generate_random_3_2_partitions``)
   — present on branch ``sprint9/critical-pipeline`` commit ``858ce51``,
   not yet on main.

2. ``bouba_sens.metrics.asymmetry.me6_max_abs_off_diag_partitioned``
   — same sprint9 branch.

3. ``bouba_sens.metrics.asymmetry.me6_mean_off_diag_partitioned``
   — Sprint 10 helper added on the Studio working tree alongside this
   script ; landing on main is pending the broader sprint9 merge.

To reproduce the verdict :

- On Studio (where the working tree carries the helpers) :
    uv run python scripts/sprint10_me6_mean_4grids.py

- On any host post-upstream-merge :
    uv run python scripts/sprint10_me6_mean_4grids.py

- On a fresh checkout of main today : will fail at the imports below.
  The error message points back here ; that is the documented behaviour
  per ADR-0017 §Caveats item 1.

See :
- ADR-0014 — original verdict (4/4 fail under max-stat)
- ADR-0015 — formal ceiling lemma
- ADR-0016 — design proposing mean-stat as Candidate B
- ADR-0017 — empirical refutation of Candidate B (this script's output)
- ADR-0018 — equivalence theorem closing the partition-test framework
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from bouba_sens.metrics.partitions import (
    generate_random_3_2_partitions,
)

from bouba_sens.metrics.asymmetry import (
    me6_max_abs_off_diag_partitioned,
    me6_mean_off_diag_partitioned,
)

MODALITIES = ("audio", "vision", "tactile", "gravity", "force")
CELL_PATTERN = re.compile(r"^seed(?P<seed>\d+)_T(?P<timing>[12])_(?P<modality>\w+)_(?P<snr>\w+)$")

GRIDS: list[tuple[str, str]] = [
    ("ECG", "runs/v04_studyforrest_real_grid"),
    ("Mock", "runs/v04_studyforrest_grid"),
    ("XOR", "runs/v03_xor_grid"),
    ("Sinu", "runs/v03_sinusoid_grid"),
]


def midrank_pctl(prereg_value: float, null_values: list[float]) -> float:
    """Standard mid-rank tie-handling convention (cf ADR-0014 Axe 6)."""
    n_below = sum(1 for v in null_values if v < prereg_value)
    n_tie = sum(1 for v in null_values if v == prereg_value)
    return (n_below + n_tie / 2) / len(null_values) * 100


def collect_perf_matrices(grid_root: Path) -> list[np.ndarray]:
    """For each (seed, timing, snr) trio, build the 5x5 perf matrix
    where row = lesioned modality, col = queried modality."""
    groups: dict[tuple[str, str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for cell_dir in grid_root.iterdir():
        if not cell_dir.is_dir():
            continue
        m = CELL_PATTERN.match(cell_dir.name)
        if not m:
            continue
        per_query_path = cell_dir / "per_query_me1.json"
        if not per_query_path.exists():
            continue
        try:
            per_query = json.loads(per_query_path.read_text())
        except OSError, json.JSONDecodeError:
            continue
        key = (m["seed"], m["timing"], m["snr"])
        groups[key][m["modality"]] = per_query

    matrices = []
    for _, lesion_to_queries in groups.items():
        if len(lesion_to_queries) != 5:
            continue
        try:
            mat = np.zeros((5, 5))
            for i, lesion in enumerate(MODALITIES):
                for j, query in enumerate(MODALITIES):
                    mat[i, j] = lesion_to_queries[lesion][query]
            matrices.append(mat)
        except KeyError, TypeError:
            continue
    return matrices


def main() -> None:
    print("=== Sprint 10 ADR-0017 — Me6_mean vs Me6_max on 4 grids ===")
    print(
        f"{'Grid':<6} {'n_trios':<8} {'mean prereg':<13} "
        f"{'max prereg':<12} {'mean midrank':<14} {'max midrank':<12}"
    )
    print("-" * 80)

    parts = generate_random_3_2_partitions(n=9, seed=0, unique=True)
    rows = []

    for name, grid_root_str in GRIDS:
        grid_root = Path(grid_root_str)
        if not grid_root.exists():
            print(f"{name:<6} GRID MISSING: {grid_root}")
            continue
        matrices = collect_perf_matrices(grid_root)
        if not matrices:
            print(f"{name:<6} NO MATRICES extracted from {grid_root}")
            continue

        mean_prereg = float(
            np.median(
                [me6_mean_off_diag_partitioned(mat, modalities=MODALITIES) for mat in matrices]
            )
        )
        max_prereg = float(
            np.median(
                [me6_max_abs_off_diag_partitioned(mat, modalities=MODALITIES) for mat in matrices]
            )
        )

        mean_nulls, max_nulls = [], []
        for p in parts:
            mean_nulls.append(
                float(
                    np.median(
                        [
                            me6_mean_off_diag_partitioned(mat, modalities=MODALITIES, partition=p)
                            for mat in matrices
                        ]
                    )
                )
            )
            max_nulls.append(
                float(
                    np.median(
                        [
                            me6_max_abs_off_diag_partitioned(
                                mat, modalities=MODALITIES, partition=p
                            )
                            for mat in matrices
                        ]
                    )
                )
            )

        mean_pctl = midrank_pctl(mean_prereg, mean_nulls)
        max_pctl = midrank_pctl(max_prereg, max_nulls)
        rows.append((name, len(matrices), mean_prereg, max_prereg, mean_pctl, max_pctl))
        print(
            f"{name:<6} {len(matrices):<8} "
            f"{mean_prereg:<13.6f} {max_prereg:<12.6f} "
            f"{mean_pctl:<14.1f} {max_pctl:<12.1f}"
        )

    if rows:
        print()
        for name, _, _, _, mean_p, max_p in rows:
            delta = mean_p - max_p
            sign = "+" if delta >= 0 else ""
            print(f"  {name:<6} mean={mean_p:.1f}% vs max={max_p:.1f}% (Δ={sign}{delta:.1f}%)")
        mean_avg = sum(r[4] for r in rows) / len(rows)
        max_avg = sum(r[5] for r in rows) / len(rows)
        print(f"\nGrand mean across grids: mean-stat={mean_avg:.1f}%, max-stat={max_avg:.1f}%")
        print(
            "Max-stat ceiling per ADR-0015 lemma: 72.2% "
            "(structural; unreachable under any matrix design on n=5)."
        )


if __name__ == "__main__":
    main()
