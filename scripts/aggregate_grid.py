"""Sprint 4 Task 4.3 — grid aggregator.

Walks a grid run directory (output of `scripts/run_grid.sh`), groups
per-cell `eval_report.json` files by `(timing, modality, snr)`, applies
`me9_bootstrap` across the 5 seeds per metric, and writes a single
`reports/v0.1_aggregate.json` with B-1 / B-2 / B-3 invariant verdicts.

Usage:
    uv run python scripts/aggregate_grid.py \
        --root runs/v01_grid --out reports/v0.1_aggregate.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from bouba_sens.metrics import me9_bootstrap

# Spec §1.2 invariant thresholds.
B1_ME7_THRESHOLD = 0.05
B2_ME3_DELTA_THRESHOLD = 0.10
B3_ME6_THRESHOLD = 0.02

_CELL_RE = re.compile(
    r"^seed(?P<seed>\d+)_(?P<timing>T[12])_(?P<modality>\w+?)_"
    r"(?P<snr>floor|minus10|plus10)$"
)


def _iter_cell_reports(root: Path) -> Iterable[tuple[dict[str, str], dict[str, Any]]]:
    """Yield (parsed-cell-id, eval_report-dict) for every run directory.

    Cell id keys: `seed`, `timing`, `modality`, `snr`.
    """
    for cell_dir in sorted(root.iterdir()):
        if not cell_dir.is_dir():
            continue
        match = _CELL_RE.match(cell_dir.name)
        if not match:
            continue
        report_path = cell_dir / "eval_report.json"
        if not report_path.exists():
            continue
        yield match.groupdict(), json.loads(report_path.read_text())


def _iter_cell_records(
    root: Path,
) -> Iterable[tuple[dict[str, str], dict[str, Any], dict[str, float] | None]]:
    """Like `_iter_cell_reports` but also returns `per_query_me1.json` if present.

    Sprint 5 / Task 5.2 — the extra file is emitted by the CLI `lesion`
    command; missing it means the cell was produced by a pre-v0.2 run
    and Me6 cannot be computed for that trio.
    """
    for cell_dir in sorted(root.iterdir()):
        if not cell_dir.is_dir():
            continue
        match = _CELL_RE.match(cell_dir.name)
        if not match:
            continue
        report_path = cell_dir / "eval_report.json"
        if not report_path.exists():
            continue
        eval_report = json.loads(report_path.read_text())
        per_query_path = cell_dir / "per_query_me1.json"
        per_query = json.loads(per_query_path.read_text()) if per_query_path.exists() else None
        yield match.groupdict(), eval_report, per_query


def _bootstrap_metric(values: list[float]) -> dict[str, float]:
    mean, lo, hi = me9_bootstrap(values)
    return {"mean": mean, "ci_low": lo, "ci_high": hi}


_MODALITIES: tuple[str, ...] = ("audio", "vision", "tactile", "gravity", "force")


def aggregate(
    root: Path,
    *,
    partition: tuple[frozenset[str], frozenset[str]] | None = None,
) -> dict[str, Any]:
    """Return `{cells, invariants, thresholds}` ready to JSON-dump.

    When ``partition`` is provided (null-model mode), Me6 is computed
    using the partition-aware scorer rather than the pre-reg default.
    """
    records = list(_iter_cell_records(root))

    # Step 1: per-cell bootstrap over seeds (me1, me2, me3_delta).
    cells_seed_values: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for cell_id, body, _ in records:
        key = (cell_id["timing"], cell_id["modality"], cell_id["snr"])
        for metric_name, value in body.items():
            if isinstance(value, (int, float)) and value is not None:
                cells_seed_values[key][metric_name].append(float(value))

    aggregated_cells: dict[str, dict[str, dict[str, float]]] = {}
    for (timing, modality, snr), metrics in cells_seed_values.items():
        cell_label = f"{timing.lower()}_{modality}_{snr}"
        aggregated_cells[cell_label] = {
            m: _bootstrap_metric(vals) for m, vals in metrics.items() if vals
        }

    # Step 2: Me6 per (seed, timing, snr) trio — build 5x5 perf matrix
    # from per_query_me1 files, rows = query modality, cols = lesioned.
    me6_values = _compute_me6(records, partition=partition)

    # Step 3: Me7 per (seed, modality, snr) pair — mean(T1) - mean(T2)
    # using the per-cell me1 (post-lesion accuracy on the *shared* input).
    me7_values = _compute_me7(records)

    invariants = _compute_invariants(aggregated_cells, me6_values=me6_values, me7_values=me7_values)

    return {
        "cells": aggregated_cells,
        "invariants": invariants,
        "thresholds": {
            "b1_me7": B1_ME7_THRESHOLD,
            "b2_me3_delta": B2_ME3_DELTA_THRESHOLD,
            "b3_me6": B3_ME6_THRESHOLD,
        },
    }


def _compute_me6(
    records: list[tuple[dict[str, str], dict[str, Any], dict[str, float] | None]],
    *,
    partition: tuple[frozenset[str], frozenset[str]] | None = None,
) -> list[float]:
    """Me6 (asymmetry max-off-diagonal) per (seed, timing, snr) trio.

    Groups cells by the trio key, stacks the per-query Me1 vectors as
    columns (one column per lesioned-modality cell), then runs
    `me6_asymmetry` + `me6_max_abs_off_diag` on the resulting 5x5
    matrix. Trios with fewer than 5 lesioned-modality cells are skipped.

    When ``partition`` is not None (null-model mode), uses the partition-
    aware ``me6_max_abs_off_diag_partitioned`` instead of the pre-reg default.
    """
    import numpy as np
    import torch

    from bouba_sens.metrics import me6_asymmetry, me6_max_abs_off_diag
    from bouba_sens.metrics.asymmetry import me6_max_abs_off_diag_partitioned

    buckets: dict[tuple[str, str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for cell_id, _, per_query in records:
        if per_query is None:
            continue
        key = (cell_id["seed"], cell_id["timing"], cell_id["snr"])
        buckets[key][cell_id["modality"]] = per_query

    out: list[float] = []
    for _, by_modality in buckets.items():
        if len(by_modality) < len(_MODALITIES):
            continue
        matrix = torch.zeros(len(_MODALITIES), len(_MODALITIES))
        for col_j, lesioned in enumerate(_MODALITIES):
            col = by_modality.get(lesioned)
            if col is None:
                break
            for row_i, query in enumerate(_MODALITIES):
                matrix[row_i, col_j] = float(col.get(query, 0.0))
        else:
            if partition is not None:
                out.append(
                    me6_max_abs_off_diag_partitioned(
                        np.array(matrix.numpy()),
                        modalities=_MODALITIES,
                        partition=partition,
                    )
                )
            else:
                asym = me6_asymmetry(matrix)
                out.append(me6_max_abs_off_diag(asym))
    return out


def _compute_me7(
    records: list[tuple[dict[str, str], dict[str, Any], dict[str, float] | None]],
) -> list[float]:
    """Me7 (congenital gap) per (seed, modality, snr) T1/T2 pair.

    Uses per-cell `me1` for pairing. Sprint 5 / Task 5.2: this is the
    aggregation-level fix for ADR-0003 which had me7 = me1 - me1 = 0.
    """
    t1: dict[tuple[str, str, str], float] = {}
    t2: dict[tuple[str, str, str], float] = {}
    for cell_id, body, _ in records:
        me1 = body.get("me1")
        if me1 is None:
            continue
        key = (cell_id["seed"], cell_id["modality"], cell_id["snr"])
        if cell_id["timing"] == "T1":
            t1[key] = float(me1)
        elif cell_id["timing"] == "T2":
            t2[key] = float(me1)
    return [t1[k] - t2[k] for k in t1 if k in t2]


def _compute_invariants(
    cells: dict[str, dict[str, dict[str, float]]],
    *,
    me6_values: list[float],
    me7_values: list[float],
) -> dict[str, dict[str, Any]]:
    """Distil the three invariant verdicts.

    Sprint 5 Task 5.2 rewrites B-1 and B-3 to consume aggregation-level
    vectors (computed in `_compute_me6` and `_compute_me7`) rather than
    per-cell placeholder columns.
    """
    me3_values = [body.get("me3_delta", {}).get("mean") for body in cells.values()]
    me3_clean = [v for v in me3_values if v is not None]
    median_me3 = _median(me3_clean)

    median_me7 = _median(me7_values)
    median_me6 = _median(me6_values)

    return {
        "b1": {
            "passes": median_me7 > B1_ME7_THRESHOLD,
            "median_me7": median_me7,
            "threshold": B1_ME7_THRESHOLD,
            "cells_counted": len(me7_values),
        },
        "b2": {
            "passes": median_me3 > B2_ME3_DELTA_THRESHOLD,
            "median_me3_delta": median_me3,
            "threshold": B2_ME3_DELTA_THRESHOLD,
            "cells_counted": len(me3_clean),
        },
        "b3": {
            "passes": median_me6 > B3_ME6_THRESHOLD,
            "median_me6_max_abs": median_me6,
            "threshold": B3_ME6_THRESHOLD,
            "cells_counted": len(me6_values),
        },
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_v[mid - 1] + sorted_v[mid]) / 2
    return sorted_v[mid]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--partition-seed",
        type=int,
        default=None,
        help=(
            "Null-model mode: random seed used to generate the partition list. "
            "Requires --partition-index."
        ),
    )
    parser.add_argument(
        "--partition-index",
        type=int,
        default=None,
        help=(
            "Null-model mode: index into the list of random 3+2 partitions "
            "(generated with --partition-seed). Requires --partition-seed."
        ),
    )
    args = parser.parse_args()

    partition: tuple[frozenset[str], frozenset[str]] | None = None
    if args.partition_seed is not None or args.partition_index is not None:
        if args.partition_seed is None or args.partition_index is None:
            parser.error("--partition-seed and --partition-index must be used together")
        from bouba_sens.metrics.partitions import generate_random_3_2_partitions

        partitions = generate_random_3_2_partitions(
            n=args.partition_index + 1, seed=args.partition_seed
        )
        partition = partitions[args.partition_index]
        print(f"null-model partition [{args.partition_index}]: {partition}")

    result = aggregate(args.root, partition=partition)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"aggregate: {len(result['cells'])} cells -> {args.out}")


if __name__ == "__main__":
    main()
