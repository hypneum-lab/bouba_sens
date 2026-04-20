"""Compute the percentile of the pre-registered B-3 Me6 median within
the empirical null distribution (10 random 3+2 partitions).

Task 7.1 acceptance: pre-reg median >= 95th percentile of null.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Annotated

import typer

_DEFAULT_OUT = Path("reports/v0.3_critical_validation/null_b3_partitions.json")


def _load_median_me6(aggregate_path: Path) -> float:
    data = json.loads(aggregate_path.read_text())
    return float(data["invariants"]["b3"]["median_me6_max_abs"])


def main(
    null_root: Annotated[Path, typer.Option(help="Parent dir with null grid subdirs")],
    prereg_aggregate: Annotated[
        Path, typer.Option(help="reports/v0.2_aggregate.json (pre-reg partition)")
    ],
    out: Annotated[Path, typer.Option(help="Output summary JSON")] = _DEFAULT_OUT,
) -> None:
    null_values: list[float] = []
    for sub in sorted(null_root.iterdir()):
        agg = sub / "aggregate.json"
        if not agg.exists():
            continue
        try:
            null_values.append(_load_median_me6(agg))
        except (json.JSONDecodeError, KeyError) as e:
            typer.echo(f"skip {agg}: {e}", err=True)
    if not null_values:
        typer.echo(f"ERROR: no aggregate.json found in {null_root}", err=True)
        raise typer.Exit(code=1)
    try:
        prereg = _load_median_me6(prereg_aggregate)
    except (json.JSONDecodeError, KeyError) as e:
        typer.echo(f"ERROR: pre-reg aggregate malformed: {e}", err=True)
        raise typer.Exit(code=1) from e
    null_values_sorted = sorted(null_values)
    rank = sum(1 for v in null_values_sorted if v < prereg)
    percentile = 100.0 * rank / len(null_values_sorted)
    summary = {
        "prereg_me6_median": prereg,
        "null_me6_medians": null_values_sorted,
        "null_median_of_medians": median(null_values_sorted),
        "prereg_rank": rank,
        "n_null": len(null_values_sorted),
        "percentile_of_prereg": percentile,
        "passes_95pct": percentile >= 95.0,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    typer.run(main)
