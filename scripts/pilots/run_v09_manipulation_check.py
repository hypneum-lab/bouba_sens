"""v0.9 pilot, manipulation check only (amendment v0.9 section 3.3).

Thin entry point around the merged library function
``bouba_sens.decorrelation.run_manipulation_check`` with its frozen
defaults (200 draws per seed, 10,000 label swaps, generator seed
20261023). The world is ``StudyforrestRealWorld`` with the held-out probe
of section 3.4: ``--split stratified`` (PI decision 2026-10-25, default)
or ``--split tail`` (``probe_fraction = 0.10``, the split of the first
run), so every anchor and donor lies in the training split. This is not the v0.9
driver: the arms, grids and fusion gain are not run here. Usage:

  BOUBA_SENS_STUDYFORREST_DATA=... uv run python \
    scripts/pilots/run_v09_manipulation_check.py --out FILE 3101 ... 3105
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from bouba_sens.decorrelation import (
    CHECK_N_DRAWS,
    GATE_N_PERM,
    GATE_SEED,
    V09_PROBE_FRACTION,
    run_manipulation_check,
)
from bouba_sens.world.studyforrest import StudyforrestRealWorld

PILOT_SEEDS = (3101, 3102, 3103, 3104, 3105)  # first pilot (legacy seeds)
PILOT_SEEDS_REVB = (3201, 3202, 3203, 3204, 3205)  # rev. B, v09-pilot namespace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("seeds", type=int, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=("stratified", "tail"), default="stratified")
    ap.add_argument("--design", choices=("revb", "legacy"), default="revb")
    args = ap.parse_args()
    pool = PILOT_SEEDS_REVB if args.design == "revb" else PILOT_SEEDS
    if tuple(args.seeds) != pool:
        raise SystemExit(f"v0.9 pilot seeds are {pool}; got {tuple(args.seeds)}")
    ns = None
    if args.design == "revb":
        from bouba_sens.seeds import BY_NAME

        ns = BY_NAME["v09-pilot"].code
    data_dir = Path(os.environ["BOUBA_SENS_STUDYFORREST_DATA"])
    # The constructor seed is stored only; sampling takes explicit seeds.
    if args.split == "tail":
        world = StudyforrestRealWorld(
            seed=args.seeds[0], data_dir=data_dir, probe_fraction=V09_PROBE_FRACTION
        )
    else:
        world = StudyforrestRealWorld(seed=args.seeds[0], data_dir=data_dir, stratified_probe=True)
    t0 = time.monotonic()
    gate, k_original = run_manipulation_check(world, list(args.seeds), seed_namespace=ns)
    wall = time.monotonic() - t0
    rec = {
        "amendment": "v0.9 section 3.3 manipulation check (pilot)",
        "world_class": type(world).__name__,
        "split": args.split,
        "design": args.design,
        "seed_namespace": ns,
        "n_frames": world.n_train + world.n_probe,
        "n_train": world.n_train,
        "n_probe": world.n_probe,
        "seeds": list(args.seeds),
        "n_draws_per_seed": CHECK_N_DRAWS,
        "n_perm": GATE_N_PERM,
        "gate_seed": GATE_SEED,
        "gate": asdict(gate),
        "k_original_mean": k_original,
        "wall_time_s": wall,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
