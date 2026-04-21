"""Bootstrap 95 % CI on Me7 median per world (Sprint 7 Task 7.2).

Loads v0.2 per-world aggregates, reconstructs the 75 paired (T1-T2)
Me7 values, bootstraps the median, compares across worlds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, TypedDict

import numpy as np
import typer
from scipy.stats import bootstrap


class Me7CI(TypedDict):
    median: float
    ci_low: float
    ci_high: float
    n: int
    n_boot: int


def bootstrap_me7_median_ci(sample: np.ndarray, *, n_boot: int, seed: int) -> Me7CI:
    if sample.size < 2:
        raise ValueError(f"sample.size must be >= 2, got {sample.size}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    rng = np.random.default_rng(seed)
    res = bootstrap(
        (sample,),
        np.median,
        n_resamples=n_boot,
        confidence_level=0.95,
        method="percentile",
        random_state=rng,
    )
    return Me7CI(
        median=float(np.median(sample)),
        ci_low=float(res.confidence_interval.low),
        ci_high=float(res.confidence_interval.high),
        n=int(sample.size),
        n_boot=n_boot,
    )


def _load_me7_sample(aggregate_path: Path) -> np.ndarray:
    try:
        text = aggregate_path.read_text()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{aggregate_path} not found; re-run aggregator to produce it"
        ) from e
    data = json.loads(text)
    raw = (
        data.get("raw_me7_pairs")
        or data.get("b1_pairs")
        or data.get("invariants", {}).get("b1", {}).get("raw_me7_pairs")
    )
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
    load_errors: dict[str, str] = {}
    for name, path in [("gaussian", gaussian), ("xor", xor), ("sinusoid", sinusoid)]:
        try:
            sample = _load_me7_sample(path)
        except (FileNotFoundError, KeyError) as e:
            load_errors[name] = str(e)
            continue
        results[name] = bootstrap_me7_median_ci(sample, n_boot=n_boot, seed=seed)
    # pairwise CI disjointness matrix
    names = list(results)
    disjoint: dict[str, dict[str, bool | None]] = {}
    for a in names:
        disjoint[a] = {}
        for b in names:
            if a == b:
                disjoint[a][a] = None
                continue
            disjoint[a][b] = (
                results[a]["ci_high"] < results[b]["ci_low"]
                or results[b]["ci_high"] < results[a]["ci_low"]
            )
    payload = {
        "per_world": results,
        "pairwise_disjoint": disjoint,
        "seed": seed,
        "load_errors": load_errors,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


app = typer.Typer()
app.command()(main)

if __name__ == "__main__":
    app()
