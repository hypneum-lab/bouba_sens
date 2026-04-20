"""Bootstrap 95 % CI on Me7 median per world (Sprint 7 Task 7.2).

Loads v0.2 per-world aggregates, reconstructs the 75 paired (T1-T2)
Me7 values, bootstraps the median, compares across worlds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from scipy.stats import bootstrap


def bootstrap_me7_median_ci(sample: np.ndarray, *, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    res = bootstrap(
        (sample,),
        np.median,
        n_resamples=n_boot,
        confidence_level=0.95,
        method="percentile",
        random_state=rng,
    )
    return {
        "median": float(np.median(sample)),
        "ci_low": float(res.confidence_interval.low),
        "ci_high": float(res.confidence_interval.high),
        "n": int(sample.size),
        "n_boot": n_boot,
    }


@dataclass(frozen=True)
class WorldResult:
    world: str
    ci: dict[str, float]


def _load_me7_sample(aggregate_path: Path) -> np.ndarray:
    data = json.loads(aggregate_path.read_text())
    raw = data.get("raw_me7_pairs") or data.get("b1_pairs")
    if raw is None:
        raise KeyError(
            f"{aggregate_path} has no raw Me7 pairs; re-run aggregator with --emit-raw-pairs"
        )
    return np.asarray(raw, dtype=float)


def main(
    gaussian: Annotated[Path, typer.Option()] = Path("reports/v0.2_aggregate.json"),
    xor: Annotated[Path, typer.Option()] = Path("reports/v0.2_aggregate_xor.json"),
    sinusoid: Annotated[Path, typer.Option()] = Path("reports/v0.2_aggregate_sinusoid.json"),
    out: Annotated[Path, typer.Option()] = Path(
        "reports/v0.3_critical_validation/me7_bootstrap.json"
    ),
    n_boot: Annotated[int, typer.Option()] = 10_000,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    results: dict[str, Any] = {}
    for name, path in [("gaussian", gaussian), ("xor", xor), ("sinusoid", sinusoid)]:
        sample = _load_me7_sample(path)
        results[name] = bootstrap_me7_median_ci(sample, n_boot=n_boot, seed=seed)
    # pairwise CI disjointness matrix
    names = list(results)
    disjoint: dict[str, dict[str, bool]] = {}
    for a in names:
        disjoint[a] = {}
        for b in names:
            if a == b:
                disjoint[a][b] = False
                continue
            disjoint[a][b] = (
                results[a]["ci_high"] < results[b]["ci_low"]
                or results[b]["ci_high"] < results[a]["ci_low"]
            )
    payload = {"per_world": results, "pairwise_disjoint": disjoint, "seed": seed}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    typer.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    typer.run(main)
