"""Amendment v0.8 driver CLI (subspace shift vs LOCK_AFTER).

capture: train the Phase-1 checkpoint of every (seed, lock) block and
    capture its codes (``bouba_sens.v08_driver.run_block``), one ``.pt``
    file per block in ``--blocks-dir`` plus ``capture_log.json`` (world
    class, wall time per block). Seeds must all be pilot seeds
    (2101-2110) or all come from the confirmatory pool (2001-2044).
pilot: from the 70 pilot blocks, k* and the section 7 planning inputs
    (``bouba_sens.v08_driver.pilot_quantities``).

``--design revc`` (PI decision 2026-10-28): seeds 2211-2220, the real
bridge with the stratified held-out rows (learning gate), straight-through
emission, encoder / logit readouts and the drift-floor control
(``v08_driver.pilot_quantities_revc``).

Usage:

  BOUBA_SENS_STUDYFORREST_DATA=... uv run python scripts/run_v08_subspace.py \
    capture --seeds 2101,...,2110 --blocks-dir runs/v08_pilot
  uv run python scripts/run_v08_subspace.py pilot --blocks-dir runs/v08_pilot \
    --out reports/pilots/v08_pilot_quantities.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from bouba_sens.cli import _build_world
from bouba_sens.metrics.subspace_aggregate import LOCKS
from bouba_sens.v08_driver import (
    CONFIRMATORY_POOL,
    PILOT_SEEDS,
    PILOT_SEEDS_REVB,
    PILOT_SEEDS_REVC,
    block_path,
    check_world,
    check_world_revc,
    load_block,
    pilot_quantities,
    pilot_quantities_revb,
    pilot_quantities_revc,
    run_block,
    save_block,
)


def _seeds(text: str) -> list[int]:
    return [int(t) for t in text.replace(" ", "").split(",") if t]


def capture(args: argparse.Namespace) -> None:
    seeds = args.seeds
    pilot_pool = {"revc": PILOT_SEEDS_REVC, "revb": PILOT_SEEDS_REVB}.get(args.design, PILOT_SEEDS)
    if not (set(seeds) <= set(pilot_pool) or set(seeds) <= set(CONFIRMATORY_POOL)):
        raise SystemExit("seeds must all be v0.8 pilot seeds or all confirmatory-pool seeds")
    if set(seeds) <= set(CONFIRMATORY_POOL) and not args.osf_filed:
        raise SystemExit("confirmatory seeds need --osf-filed <OSF DOI> (filing gate)")
    torch.use_deterministic_algorithms(True)
    if args.design == "revc":
        import os

        from bouba_sens.world.studyforrest import StudyforrestRealWorld

        world = StudyforrestRealWorld(
            seed=seeds[0],
            data_dir=Path(os.environ["BOUBA_SENS_STUDYFORREST_DATA"]),
            stratified_probe=True,
        )
        world_class = check_world_revc(world)
    else:
        world = _build_world("studyforrest", seeds[0])
        world_class = check_world(world)
    log: dict[str, object] = {
        "world_class": world_class,
        "osf_filed": args.osf_filed,
        "design": args.design,
        "blocks": [],
    }
    blocks: list[dict[str, object]] = []
    t0 = time.monotonic()
    for s in seeds:
        for lock in LOCKS:
            path = block_path(args.blocks_dir, s, lock)
            if path.exists():
                raise SystemExit(f"{path} exists; refusing to overwrite")
            run = run_block(world, s, lock, design=args.design)
            save_block(run.block, path)
            blocks.append({"seed": s, "lock": lock, "wall_s": run.wall_s})
            print(f"seed={s} lock={lock} wall={run.wall_s:.1f}s", flush=True)
    log["blocks"] = blocks
    log["wall_s_total"] = time.monotonic() - t0
    (args.blocks_dir / "capture_log.json").write_text(json.dumps(log, indent=2) + "\n")


def pilot(args: argparse.Namespace) -> None:
    if args.design == "revc":
        seeds = PILOT_SEEDS_REVC
        blocks = [load_block(block_path(args.blocks_dir, s, lk)) for s in seeds for lk in LOCKS]
        out = pilot_quantities_revc(blocks)
    elif args.design == "revb":
        seeds = PILOT_SEEDS_REVB
        blocks = [load_block(block_path(args.blocks_dir, s, lk)) for s in seeds for lk in LOCKS]
        out = pilot_quantities_revb(blocks)
    else:
        seeds = PILOT_SEEDS
        blocks = [load_block(block_path(args.blocks_dir, s, lk)) for s in seeds for lk in LOCKS]
        out = pilot_quantities(blocks)
    log = json.loads((args.blocks_dir / "capture_log.json").read_text())
    walls: dict[int, float] = {}
    for b in log["blocks"]:
        walls[b["seed"]] = walls.get(b["seed"], 0.0) + b["wall_s"]
    out["world_class"] = log["world_class"]
    out["wall_s_per_seed"] = walls
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("blocks", "prs")}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--seeds", type=_seeds, required=True)
    c.add_argument("--blocks-dir", type=Path, required=True)
    c.add_argument("--osf-filed", default="", help="OSF DOI of the filed amendment")
    c.add_argument("--design", choices=("revc", "revb", "legacy"), default="revb")
    p = sub.add_parser("pilot")
    p.add_argument("--blocks-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--design", choices=("revc", "revb", "legacy"), default="revb")
    args = ap.parse_args()
    {"capture": capture, "pilot": pilot}[args.cmd](args)


if __name__ == "__main__":
    main()
