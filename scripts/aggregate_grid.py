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


def _bootstrap_metric(values: list[float]) -> dict[str, float]:
    mean, lo, hi = me9_bootstrap(values)
    return {"mean": mean, "ci_low": lo, "ci_high": hi}


def aggregate(root: Path) -> dict[str, Any]:
    """Return `{cells: {...}, invariants: {...}}` ready to JSON-dump."""
    # Group metric values by (timing, modality, snr).
    cells: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for cell_id, body in _iter_cell_reports(root):
        key = (cell_id["timing"], cell_id["modality"], cell_id["snr"])
        for metric_name, value in body.items():
            if isinstance(value, (int, float)) and value is not None:
                cells[key][metric_name].append(float(value))

    # Bootstrap per-cell per-metric.
    aggregated_cells: dict[str, dict[str, dict[str, float]]] = {}
    for (timing, modality, snr), metrics in cells.items():
        cell_label = f"{timing.lower()}_{modality}_{snr}"
        aggregated_cells[cell_label] = {
            m: _bootstrap_metric(vals) for m, vals in metrics.items() if vals
        }

    # Invariant verdicts.
    invariants = _compute_invariants(aggregated_cells)

    return {
        "cells": aggregated_cells,
        "invariants": invariants,
        "thresholds": {
            "b1_me7": B1_ME7_THRESHOLD,
            "b2_me3_delta": B2_ME3_DELTA_THRESHOLD,
            "b3_me6": B3_ME6_THRESHOLD,
        },
    }


def _compute_invariants(
    cells: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, Any]]:
    """Distil the 3 invariant verdicts from the per-cell bootstrap tables."""
    # B-1: median Me7 across floor-SNR cells must exceed threshold.
    me7_floor = [
        body.get("me7", {}).get("mean") for name, body in cells.items() if "_floor" in name
    ]
    me7_clean = [v for v in me7_floor if v is not None]
    median_me7 = _median(me7_clean)

    # B-2: median me3_delta across ALL cells.
    me3_values = [body.get("me3_delta", {}).get("mean") for body in cells.values()]
    me3_clean = [v for v in me3_values if v is not None]
    median_me3 = _median(me3_clean)

    # B-3: median me6_max_abs across ALL cells.
    me6_values = [body.get("me6_max_abs", {}).get("mean") for body in cells.values()]
    me6_clean = [v for v in me6_values if v is not None]
    median_me6 = _median(me6_clean)

    return {
        "b1": {
            "passes": median_me7 > B1_ME7_THRESHOLD,
            "median_me7": median_me7,
            "threshold": B1_ME7_THRESHOLD,
            "cells_counted": len(me7_clean),
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
            "cells_counted": len(me6_clean),
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
    args = parser.parse_args()

    result = aggregate(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"aggregate: {len(result['cells'])} cells -> {args.out}")


if __name__ == "__main__":
    main()
