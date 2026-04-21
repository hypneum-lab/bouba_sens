"""Sprint 14a — per-seed grid aggregator for seed-stability verification.

Runs the usual (timing, modality, snr) aggregation independently inside
each of the 5 seed buckets, so we can inspect whether a grid-level
headline B-1/B-2/B-3 verdict is driven by a single lucky seed or is
consistent across the replication set.

The per-seed invariant computation re-uses `scripts.aggregate_grid`
helpers: we simply narrow the record list to a single seed before
delegating to `_compute_invariants`. This keeps the pre-registered
thresholds and bootstrap pipeline identical to the main aggregator.

Usage:
    uv run python scripts/aggregate_grid_per_seed.py \
        --root runs/v05_s12_tau0_3 \
        --out reports/v0.5_s14a_tau0_3_per_seed.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from aggregate_grid import (  # noqa: E402
    _bootstrap_metric,
    _compute_invariants,
    _compute_me6,
    _compute_me7,
    _iter_cell_records,
)

SEEDS: tuple[str, ...] = ("0", "1", "2", "3", "4")


def _per_seed_invariants(
    records: list[tuple[dict[str, str], dict[str, Any], dict[str, float] | None]],
) -> dict[str, dict[str, Any]]:
    """Return `{seed: {b1, b2, b3}}` — the three verdicts restricted to each seed."""
    out: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        seed_records = [r for r in records if r[0]["seed"] == seed]
        if not seed_records:
            continue

        cells_seed_values: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for cell_id, body, _ in seed_records:
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

        me6_values = _compute_me6(seed_records)
        me7_values = _compute_me7(seed_records)
        invariants = _compute_invariants(
            aggregated_cells, me6_values=me6_values, me7_values=me7_values
        )
        out[seed] = invariants
    return out


def _cross_seed_summary(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Collapse the 5 per-seed verdicts into mean/std + sign-stability strings."""
    summary: dict[str, Any] = {}
    for inv_name, value_key in (
        ("b1", "median_me7"),
        ("b2", "median_me3_delta"),
        ("b3", "median_me6_max_abs"),
    ):
        values = [float(per_seed[s][inv_name][value_key]) for s in sorted(per_seed)]
        passes = [bool(per_seed[s][inv_name]["passes"]) for s in sorted(per_seed)]
        positives = sum(1 for v in values if v > 0.0)
        negatives = sum(1 for v in values if v < 0.0)
        mean = statistics.fmean(values) if values else 0.0
        stdev = statistics.stdev(values) if len(values) >= 2 else 0.0
        summary[inv_name] = {
            f"{inv_name}_values": values,
            f"{inv_name}_mean": mean,
            f"{inv_name}_std": stdev,
            f"{inv_name}_passes": passes,
            "sign_stability": f"{positives}/{len(values)} positive",
            "sign_negative": f"{negatives}/{len(values)} negative",
            "passes_count": f"{sum(passes)}/{len(passes)}",
        }
    return summary


def aggregate_per_seed(root: Path) -> dict[str, Any]:
    """Return `{per_seed, aggregate}` ready to JSON-dump."""
    records = list(_iter_cell_records(root))
    per_seed = _per_seed_invariants(records)
    aggregate = _cross_seed_summary(per_seed)
    return {"per_seed": per_seed, "aggregate": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = aggregate_per_seed(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"per-seed aggregate: {len(result['per_seed'])} seeds -> {args.out}")


if __name__ == "__main__":
    main()
