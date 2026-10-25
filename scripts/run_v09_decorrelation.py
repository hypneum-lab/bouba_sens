"""Amendment v0.9 driver CLI: main-grid arms and the pilot D variance.

Runs every (seed, onset, arm) cell of the main grid (onsets 0-200, arms
decor / joint / plain, w = 50, R = 200; ``bouba_sens.v09_driver``) on
``StudyforrestRealWorld(stratified_probe = True)`` and writes one JSON with
the fusion gains, wall times and, for the pilot seeds 3101-3105, the
deficit matrix D and its variance components (section 7). Seeds must
all be pilot seeds or all come from the confirmatory pool (3001-3044).

Usage:

  BOUBA_SENS_STUDYFORREST_DATA=... uv run python scripts/run_v09_decorrelation.py \
    --seeds 3101,3102,3103,3104,3105 --out reports/pilots/v09_pilot_main_grid.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from bouba_sens.v09_driver import (
    CONFIRMATORY_POOL,
    ONSETS,
    PILOT_SEEDS,
    PILOT_SEEDS_REVB,
    check_world,
    deficit_matrix,
    run_main_grid,
    variance_components,
)
from bouba_sens.world.studyforrest import StudyforrestRealWorld


def _seeds(text: str) -> list[int]:
    return [int(t) for t in text.replace(" ", "").split(",") if t]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=_seeds, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--osf-filed", default="", help="OSF DOI of the filed amendment")
    ap.add_argument("--design", choices=("revb", "legacy"), default="revb")
    args = ap.parse_args()
    seeds = args.seeds
    pilot_pool = PILOT_SEEDS_REVB if args.design == "revb" else PILOT_SEEDS
    if not (set(seeds) <= set(pilot_pool) or set(seeds) <= set(CONFIRMATORY_POOL)):
        raise SystemExit("seeds must all be v0.9 pilot seeds or all confirmatory-pool seeds")
    if set(seeds) <= set(CONFIRMATORY_POOL) and not args.osf_filed:
        raise SystemExit("confirmatory seeds need --osf-filed <OSF DOI> (filing gate)")
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite")
    torch.use_deterministic_algorithms(True)
    world = StudyforrestRealWorld(
        seed=seeds[0],
        data_dir=Path(os.environ["BOUBA_SENS_STUDYFORREST_DATA"]),
        stratified_probe=True,
    )
    out: dict[str, object] = {
        "world_class": check_world(world),
        "seeds": seeds,
        "osf_filed": args.osf_filed,
        "design": args.design,
    }
    cells = []
    t0 = time.monotonic()
    for s in seeds:
        for c in run_main_grid(world, s, design=args.design):
            cells.append(c)
            lin = (
                f" FGlin={c.lin.fg:+.4f} bacc_all={c.lin.bacc_all:.3f} best={c.lin.best_modality}"
                f" ctrl={c.lin.fg_control:+.4f}"
                if c.lin is not None
                else ""
            )
            print(
                f"seed={c.seed} onset={c.onset} arm={c.arm} FG={c.fg.fusion_gain:+.4f} "
                f"acc_all={c.fg.acc_all:.4f}{lin} wall={c.wall_s:.1f}s",
                flush=True,
            )
    out["wall_s_total"] = time.monotonic() - t0
    out["cells"] = [
        {
            **{k: v for k, v in asdict(c).items() if k not in ("fg", "lin")},
            "fg": asdict(c.fg),
            "lin": None if c.lin is None else asdict(c.lin),
        }
        for c in cells
    ]
    if sorted(seeds) == sorted(pilot_pool):
        outcome = "linear" if args.design == "revb" else "head"
        d = deficit_matrix(cells, sorted(seeds), outcome=outcome)
        out["onsets"] = list(ONSETS)
        out["D_outcome"] = outcome
        out["D"] = d.tolist()
        out["variance_components"] = variance_components(d)
        if args.design == "revb":
            out["D_head_descriptive"] = deficit_matrix(cells, sorted(seeds)).tolist()
            plain = [c.lin for c in cells if c.arm == "plain"]
            out["headroom"] = {
                "plain_bacc_all_mean": float(np.mean([x.bacc_all for x in plain])),
                "plain_bacc_best_single_mean": float(np.mean([x.bacc_best for x in plain])),
                "fg_lin_mean_all_runs": float(np.mean([c.lin.fg for c in cells])),
                "fg_control_mean_all_runs": float(np.mean([c.lin.fg_control for c in cells])),
                "D_zero_fraction": float(np.mean(d == 0.0)),
                "recalls_all_mean": np.mean([c.lin.recalls_all for c in cells], 0).tolist(),
                "best_modality_counts": {
                    m: sum(c.lin.best_modality == m for c in cells)
                    for m in ("audio", "vision", "tactile", "gravity", "force")
                },
                "all_converged": all(c.lin.converged for c in cells),
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=2))


if __name__ == "__main__":
    main()
