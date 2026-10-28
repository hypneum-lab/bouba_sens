"""Amendment v0.9 rev. C pilot (PI decision 2026-10-28, about 19:05).

One world per call: ``--world ccfilm`` (PRIMARY; the 10-fps cache in
``BOUBA_SENS_CCFILM_DATA``) or ``--world studyforrest`` (SECONDARY; the
real bridge in ``BOUBA_SENS_STUDYFORREST_DATA``). Both use the two-channel
label y = 2a + b and the frozen purged segment split
(``bouba_sens.v09_revc``), straight-through emission and the v09-pilot seed
namespace (code 9101), pilot seeds 3206-3210 only.

Steps: the world description (label provenance, split counts), the
manipulation check (section 3.3, 200 draws per seed), then every (seed,
onset, arm) cell for arms decor / joint / plain / label_break, and the gate
summary with the D matrix (``v09_revc.summarise``). The N rule is
``scripts/pilots/derive_v09_pilot.py`` on the output.

Usage:

  BOUBA_SENS_CCFILM_DATA=... uv run python scripts/run_v09_revc.py \
    --world ccfilm --out reports/pilots/v09_revc_ccfilm_pilot.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from bouba_sens.decorrelation import run_manipulation_check
from bouba_sens.seeds import BY_NAME
from bouba_sens.v09_revc import (
    ARMS,
    ONSETS,
    PILOT_SEEDS,
    SegmentSplitWorld,
    run_cell_revc,
    summarise,
)

ENV = {"ccfilm": "BOUBA_SENS_CCFILM_DATA", "studyforrest": "BOUBA_SENS_STUDYFORREST_DATA"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", choices=tuple(ENV), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-check", action="store_true", help="tests only")
    ap.add_argument("--onsets", default=",".join(map(str, ONSETS)), help="tests only")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite")
    seeds = list(PILOT_SEEDS)
    ns = BY_NAME["v09-pilot"].code
    torch.use_deterministic_algorithms(True)
    world = SegmentSplitWorld(args.world, Path(os.environ[ENV[args.world]]))
    out: dict[str, object] = {
        "design": "rev. C",
        "world": args.world,
        "world_description": world.describe(),
        "seeds": seeds,
        "seed_namespace": ns,
        "emission": "straight-through",
    }
    print(json.dumps(out["world_description"], indent=2), flush=True)
    t0 = time.monotonic()
    if not args.skip_check:
        gate, k_orig = run_manipulation_check(world, seeds, seed_namespace=ns)  # type: ignore[arg-type]
        out["manipulation_check"] = {**asdict(gate), "k_original_mean": k_orig}
        print(json.dumps(out["manipulation_check"], indent=2), flush=True)
    out["check_wall_s"] = time.monotonic() - t0
    onsets = [int(x) for x in args.onsets.split(",")]
    cells = []
    for s in seeds:
        for o in onsets:
            for arm in ARMS:
                c = run_cell_revc(world, s, o, arm, seed_namespace=ns)
                cells.append(c)
                print(
                    f"seed={s} onset={o} arm={arm} full={c['task']['full_bacc']:.3f} "
                    f"bestq={c['task']['best_query_bacc']:.3f} bacc_all={c['lin']['bacc_all']:.3f} "
                    f"best={c['lin']['best_modality']}:{c['lin']['bacc_best']:.3f} "
                    f"ctrl={c['lin']['bacc_control']:.3f} wall={c['wall_s']:.1f}s",
                    flush=True,
                )
    out["wall_s_total"] = time.monotonic() - t0
    out["cells"] = cells
    if onsets == list(ONSETS):
        out.update(summarise(cells, seeds))
        print(json.dumps({k: out[k] for k in ("gates", "learning_gate")}, indent=2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
