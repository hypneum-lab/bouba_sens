"""Re-aggregate v0.2 grid reports under three MI estimators.

Task 7.3 acceptance : Gaussian > XOR > Sinusoid decay holds under
at least one alternative estimator (binning or MINE), not just
Kraskov.
"""

from __future__ import annotations

import itertools
import json
import pickle
from collections.abc import Callable
from pathlib import Path
from statistics import median
from typing import Annotated

import numpy as np
import typer

from bouba_sens.metrics.mi_migration import (
    me3_delta,
    me3_delta_binning,
    me3_delta_mine,
)

EstimatorFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]

ESTIMATORS: dict[str, EstimatorFn] = {
    "kraskov": lambda pre, post, y: me3_delta(pre, post, y),
    "binning": lambda pre, post, y: me3_delta_binning(pre, post, y, bins_per_dim=16),
    "mine": lambda pre, post, y: me3_delta_mine(pre, post, y, epochs=300, seed=0),
}


def _iter_cell_reports(root: Path) -> list[Path]:
    return sorted(root.rglob("report.pkl"))


app = typer.Typer()


@app.command()
def main(
    worlds: Annotated[list[str], typer.Argument(help="Ordered world names")],
    runs_root: Annotated[Path, typer.Option()] = Path("runs"),
    out: Annotated[Path, typer.Option()] = Path(
        "reports/v0.3_critical_validation/mi_estimator_comparison.json"
    ),
) -> None:
    per_world: dict[str, dict[str, float]] = {}
    for w in worlds:
        root = runs_root / "v02_grid" if w == "gaussian" else runs_root / f"v02_{w}"
        per_estimator: dict[str, list[float]] = {k: [] for k in ESTIMATORS}
        for cell_report in _iter_cell_reports(root):
            with cell_report.open("rb") as f:
                payload = pickle.load(f)
            if hasattr(payload, "pre_lesion_codes"):
                pre = np.asarray(payload.pre_lesion_codes, dtype=float)
                post = np.asarray(payload.post_lesion_codes, dtype=float)
                y = np.asarray(payload.probe_labels, dtype=int)
            else:
                pre = np.asarray(payload["pre_lesion_codes"], dtype=float)
                post = np.asarray(payload["post_lesion_codes"], dtype=float)
                y = np.asarray(
                    payload.get("probe_labels", payload.get("pre_lesion_labels")),
                    dtype=int,
                )
            for name, fn in ESTIMATORS.items():
                per_estimator[name].append(float(fn(pre, post, y)))
        per_world[w] = (
            {name: median(vals) for name, vals in per_estimator.items()}
            if any(per_estimator.values())
            else {name: float("nan") for name in ESTIMATORS}
        )

    orderings: dict[str, bool] = {}
    for est in ESTIMATORS:
        seq = [per_world[w][est] for w in worlds]
        orderings[est] = all(a > b for a, b in itertools.pairwise(seq))

    result = {
        "per_world_median": per_world,
        "decay_ordering_holds": orderings,
        "worlds_tested_in_order": worlds,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
