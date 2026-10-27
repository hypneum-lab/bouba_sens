"""Learning criterion of the straight-through emission pilot (ADR-0024 follow-up).

Reads `phase1_seed*/pretrain_losses.json` and `phase1_seed*/probe_v2_meta.json`
(written by `bouba-sens train --probe-v2`) under a root and applies the
criterion fixed in `docs/prereg/2026-10-26-emission-pilot-criterion.md`:

- mean over seeds of the mean Phase-1 loss over the last 20 steps < 1.30
  (ln 4 = 1.386);
- mean over seeds of the v2 probe vision margin over the majority baseline
  >= 0.10.

Usage:
    uv run python scripts/emission_pilot_check.py --root PILOT_ROOT --out pilot.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

LOSS_THRESHOLD = 1.30
LAST_STEPS = 20
CONTROL_MODALITY = "vision"
CONTROL_MARGIN = 0.10
_PHASE1_RE = re.compile(r"^phase1_seed(?P<seed>\d+)$")


def check(root: Path) -> dict[str, Any]:
    seeds: list[dict[str, Any]] = []
    for d in sorted(root.iterdir()):
        match = _PHASE1_RE.match(d.name)
        if not match:
            continue
        losses = json.loads((d / "pretrain_losses.json").read_text())
        meta = json.loads((d / "probe_v2_meta.json").read_text())["modalities"]
        seeds.append(
            {
                "seed": int(match["seed"]),
                "steps": len(losses),
                "loss_first20": sum(losses[:LAST_STEPS]) / LAST_STEPS,
                "loss_last20": sum(losses[-LAST_STEPS:]) / LAST_STEPS,
                "margins": {m: v["accuracy"] - v["baseline"] for m, v in meta.items()},
                "n_codes_test": {m: v["n_codes_test"] for m, v in meta.items()},
            }
        )
    if not seeds:
        raise ValueError(f"no phase1_seed* under {root}")
    loss = sum(s["loss_last20"] for s in seeds) / len(seeds)
    margin = sum(s["margins"][CONTROL_MODALITY] for s in seeds) / len(seeds)
    loss_ok, control_ok = loss < LOSS_THRESHOLD, margin >= CONTROL_MARGIN
    return {
        "root": str(root),
        "criterion": {
            "loss_threshold": LOSS_THRESHOLD,
            "ln4": math.log(4),
            "control_modality": CONTROL_MODALITY,
            "control_margin": CONTROL_MARGIN,
        },
        "mean_loss_last20": loss,
        "mean_control_margin": margin,
        "loss_ok": loss_ok,
        "control_ok": control_ok,
        "learns": loss_ok and control_ok,
        "seeds": seeds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = check(args.root)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(json.dumps({k: v for k, v in result.items() if k != "seeds"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
