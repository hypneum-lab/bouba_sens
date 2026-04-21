"""Sprint 7 / Task 7.5 — world-complexity audit CLI.

Runs `compute_world_profile` over multiple seeds per world and writes
a JSON table with median + IQR per metric, plus a pairwise-distance
summary so reviewers can see how far apart the 3 synthetic worlds
really are in complexity space.

Usage:
    uv run python scripts/audit_worlds.py \\
        --worlds gaussian,xor,sinusoid \\
        --batch-size 1024 \\
        --seeds 0,1,2,3,4 \\
        --out reports/v0.3_critical_validation/world_complexity_audit.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import median

import numpy as np

from bouba_sens.audit import compute_world_profile
from bouba_sens.world.base import WorldSimulator
from bouba_sens.world.gaussian import GaussianWorld
from bouba_sens.world.sinusoid import SinusoidWorld
from bouba_sens.world.xor import XORWorld

_WORLD_FACTORIES = {
    "gaussian": GaussianWorld,
    "xor": XORWorld,
    "sinusoid": SinusoidWorld,
}


def _build(name: str, seed: int) -> WorldSimulator:
    if name.startswith("studyforrest_real5:"):
        from bouba_sens.world.studyforrest import StudyforrestRealWorld

        path = name.split(":", 1)[1]
        return StudyforrestRealWorld(seed=seed, data_dir=Path(path))
    return _WORLD_FACTORIES[name](seed=seed)


def _iqr(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def audit(
    worlds: list[str], seeds: list[int], batch_size: int
) -> dict[str, dict[str, dict[str, float]]]:
    """Return `{world: {metric: {median, iqr}}}`."""
    results: dict[str, dict[str, dict[str, float]]] = {}
    for world in worlds:
        per_seed: dict[str, list[float]] = defaultdict(list)
        for seed in seeds:
            sim = _build(world, seed)
            sample = sim.sample(batch_size=batch_size, seed=seed)
            profile = compute_world_profile(sample)
            for metric, value in profile.items():
                per_seed[metric].append(value)
        results[world] = {
            metric: {"median": float(median(vs)), "iqr": _iqr(vs)}
            for metric, vs in per_seed.items()
        }
    return results


def _pairwise_gap(
    per_world: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """For each metric, compute the max pairwise median gap across worlds."""
    worlds = list(per_world.keys())
    if len(worlds) < 2:
        return {}
    metrics = list(next(iter(per_world.values())).keys())
    gaps: dict[str, dict[str, float]] = {}
    for m in metrics:
        medians = [per_world[w][m]["median"] for w in worlds]
        gap = float(max(medians) - min(medians))
        rng = float(max(abs(max(medians)), abs(min(medians)))) or 1.0
        gaps[m] = {"gap": gap, "relative_gap": gap / rng}
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", default="gaussian,xor,sinusoid")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    worlds = [w.strip() for w in args.worlds.split(",") if w.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    per_world = audit(worlds, seeds, args.batch_size)
    gaps = _pairwise_gap(per_world)

    max_gap_metric = max(gaps, key=lambda m: gaps[m]["relative_gap"]) if gaps else ""
    summary = {
        "worlds": per_world,
        "pairwise_gaps": gaps,
        "interpretation": {
            "max_relative_gap_metric": max_gap_metric,
            "max_relative_gap_value": gaps.get(max_gap_metric, {}).get("relative_gap", 0.0)
            if max_gap_metric
            else 0.0,
            "note": "Relative gap = range(medians) / max(|medians|) per metric. "
            "Small values (<0.1) mean the worlds cluster tightly on that axis.",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"world-audit: {len(worlds)} worlds x {len(seeds)} seeds -> {args.out}")
    if max_gap_metric:
        gap = gaps[max_gap_metric]
        print(
            f"  max relative gap: {max_gap_metric} "
            f"({gap['relative_gap']:.3f}, absolute {gap['gap']:.4f})"
        )


if __name__ == "__main__":
    main()
