"""Amendment v0.9 rev. C pilot (PI decision 2026-10-28, about 19:05).

One world per call: ``--world ccfilm`` (PRIMARY; the 10-fps cache in
``BOUBA_SENS_CCFILM_DATA``) or ``--world studyforrest`` (SECONDARY; the
real bridge in ``BOUBA_SENS_STUDYFORREST_DATA``). Both use the two-channel
label y = 2a + b and the frozen purged segment split
(``bouba_sens.v09_revc``), straight-through emission and the v09-pilot seed
namespace (code 9101), pilot seeds only: ``--budget revc`` (recovery R =
200, seeds 3206-3210) or ``--budget revd`` (PI decision 2026-10-29: R =
1,750, fresh seeds 3211-3215).

Stages (stop and report on failure, Codex r8): (1) the manipulation check
(section 3.3, 200 draws per seed); (2) the plain arm at every (seed,
onset) and the gates computable from it (learning at every onset,
isolation, positive control, plain bacc, convergence; Codex r10); (3) the decor, joint and
label_break arms, the gate summary and the D matrix
(``v09_revc.summarise``). A stopped run writes its JSON with ``stopped``
set. The N rule is ``scripts/pilots/derive_v09_pilot.py`` on the output
(no N without the D matrix).

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
    PILOT_SEEDS_REVD,
    RECOVERY,
    RECOVERY_REVD,
    SegmentSplitWorld,
    plain_stage_gates,
    run_cell_revc,
    summarise,
)

ENV = {"ccfilm": "BOUBA_SENS_CCFILM_DATA", "studyforrest": "BOUBA_SENS_STUDYFORREST_DATA"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", choices=tuple(ENV), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--budget",
        choices=("revc", "revd"),
        required=True,
        help="revc: R = 200, seeds 3206-3210; revd (PI 2026-10-29): R = 1750, seeds 3211-3215",
    )
    ap.add_argument("--skip-check", action="store_true", help="tests only")
    ap.add_argument("--onsets", default=",".join(map(str, ONSETS)), help="tests only")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite")
    seeds = list(PILOT_SEEDS if args.budget == "revc" else PILOT_SEEDS_REVD)
    recovery = RECOVERY if args.budget == "revc" else RECOVERY_REVD
    ns = BY_NAME["v09-pilot"].code
    torch.use_deterministic_algorithms(True)
    world = SegmentSplitWorld(args.world, Path(os.environ[ENV[args.world]]))
    out: dict[str, object] = {
        "design": "rev. C" if args.budget == "revc" else "rev. D",
        "recovery": recovery,
        "world": args.world,
        "world_description": world.describe(),
        "seeds": seeds,
        "seed_namespace": ns,
        "emission": "straight-through",
    }
    print(json.dumps(out["world_description"], indent=2), flush=True)
    t0 = time.monotonic()

    def finish(stopped: str | None) -> None:
        out["stopped"] = stopped
        out["wall_s_total"] = time.monotonic() - t0
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n")
        print(f"disposition: {stopped or 'grid complete'}", flush=True)

    # Stage 1: manipulation check (section 3.3); stop and report on failure.
    if not args.skip_check:
        gate, k_orig = run_manipulation_check(world, seeds, seed_namespace=ns)  # type: ignore[arg-type]
        out["manipulation_check"] = {**asdict(gate), "k_original_mean": k_orig}
        print(json.dumps(out["manipulation_check"], indent=2), flush=True)
        if not gate.passed:
            return finish("manipulation check failed: stop and report")
    out["check_wall_s"] = time.monotonic() - t0
    onsets = [int(x) for x in args.onsets.split(",")]
    cells: list[dict[str, object]] = []
    out["cells"] = cells

    def run(arm: str) -> None:
        for s in seeds:
            for o in onsets:
                c = run_cell_revc(world, s, o, arm, seed_namespace=ns, recovery=recovery)
                cells.append(c)
                print(
                    f"seed={s} onset={o} arm={arm} full={c['task']['full_bacc']:.3f} "
                    f"bestq={c['task']['best_query_bacc']:.3f} bacc_all={c['lin']['bacc_all']:.3f} "
                    f"best={c['lin']['best_modality']}:{c['lin']['bacc_best']:.3f} "
                    f"ctrl={c['lin']['bacc_control']:.3f} wall={c['wall_s']:.1f}s",
                    flush=True,
                )

    # Stage 2: plain arm and the learning gate at every onset (D.0).
    run("plain")
    stage = plain_stage_gates(cells, seeds)  # type: ignore[arg-type]
    out["plain_stage"] = stage
    print(json.dumps(stage["plain_gates"], indent=2), flush=True)
    failed = [k for k, v in stage["plain_gates"].items() if not v]
    if failed:
        return finish(f"plain-stage prerequisite failed ({', '.join(failed)}): stop and report")
    # Stage 3: the other arms, then the gates and D.
    for arm in ARMS:
        if arm != "plain":
            run(arm)
    if onsets == list(ONSETS):
        out.update(summarise(cells, seeds))  # type: ignore[arg-type]
        print(json.dumps({k: out[k] for k in ("gates",)}, indent=2), flush=True)
        # eligibility needs a recorded, passed manipulation check (Codex r9):
        # a --skip-check run (tests only) is never eligible
        mc = out.get("manipulation_check")
        out["all_gates_passed"] = bool(
            all(out["gates"].values()) and mc is not None and mc["passed"]  # type: ignore[union-attr, index]
        )
    finish(None)


if __name__ == "__main__":
    main()
