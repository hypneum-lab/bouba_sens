"""Re-run the Amedi dose-response curve with multiple seeds.

Q3 from HYPNEUM-PLANS/2026-05-10-niveau8-three-experiments.md.
Pre-registration: docs/milestones/q3-amedi-seeds-2026-05-10.md.

For each seed in --seeds, sweep --lock-after values by invoking
`scripts/run_grid.sh` (single-seed grid: 1 seed * 5 modalities * 2
timings * 3 SNR = 30 cells per LOCK_AFTER) followed by
`scripts/aggregate_grid.py`. Extracts median Me7 (the B-1 curve point)
and the per-seed peak position and height. All per-seed results plus
the meta-aggregate land in a single JSON for downstream Jonckheere-
Terpstra analysis (Task 3).

Usage::

    uv run python scripts/replay_amedi_seeds.py \\
        --seeds 0 17 42 73 101 \\
        --lock-after 50 75 100 125 150 \\
        --out reports/v0.5_amedi_curve_multiseed.json

Pattern A (subprocess) was chosen because the upstream "dose-response
generator" is the run_grid.sh -> aggregate_grid.py pipeline, which is
not a callable Python entry point. The wrapper composes those two
existing scripts rather than re-implementing the grid in Python.

This script is HEAVY (~10-20h on Studio for the full 5-seed sweep)
and is invoked by Task 3, not directly during Task 2 development.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_single_lock_after(
    seed: int,
    lock_after: int,
    runs_root: Path,
    reports_dir: Path,
    studyforrest_data: Path,
    runner: Any = subprocess.run,
) -> float:
    """Run one (seed, lock_after) cell of the grid and return median Me7.

    Invokes ``scripts/run_grid.sh`` with ``SEEDS="<seed>"`` then
    ``scripts/aggregate_grid.py`` to produce the
    ``v0.5_dr_lock{LA}_seed{S}_aggregate.json`` artifact, and reads
    ``data["invariants"]["b1"]["median_me7"]`` from it.

    ``runner`` is injectable for tests (defaults to ``subprocess.run``).
    """
    out_root = runs_root / f"v05_dr_seed{seed}_lock{lock_after}"
    aggregate_json = reports_dir / f"v0.5_dr_lock{lock_after}_seed{seed}_aggregate.json"

    grid_env = {
        **os.environ,
        "SEEDS": str(seed),
        "OUT_ROOT": str(out_root),
        "WORLD": "studyforrest",
        "BOUBA_SENS_STUDYFORREST_DATA": str(studyforrest_data),
        "LOCK_AFTER": str(lock_after),
    }

    runner(
        ["bash", str(REPO_ROOT / "scripts" / "run_grid.sh")],
        env=grid_env,
        check=True,
        cwd=REPO_ROOT,
    )
    runner(
        [
            "uv",
            "run",
            "python",
            str(REPO_ROOT / "scripts" / "aggregate_grid.py"),
            "--root",
            str(out_root),
            "--out",
            str(aggregate_json),
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    data = json.loads(aggregate_json.read_text())
    return float(data["invariants"]["b1"]["median_me7"])


def run_single_seed(
    seed: int,
    lock_after_values: list[int],
    runs_root: Path,
    reports_dir: Path,
    studyforrest_data: Path,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Run the dose-response generator for one seed.

    Returns a dict with ``seed``, ``lock_after_values``, ``b1_values``
    (median Me7 per LOCK_AFTER), ``peak_position`` (LOCK_AFTER at
    argmax), and ``peak_height`` (max median Me7).
    """
    b1_values: list[float] = []
    for lock_after in lock_after_values:
        me7 = _run_single_lock_after(
            seed=seed,
            lock_after=lock_after,
            runs_root=runs_root,
            reports_dir=reports_dir,
            studyforrest_data=studyforrest_data,
            runner=runner,
        )
        b1_values.append(me7)

    peak_idx = max(range(len(b1_values)), key=b1_values.__getitem__)
    return {
        "seed": seed,
        "lock_after_values": list(lock_after_values),
        "b1_values": b1_values,
        "peak_position": int(lock_after_values[peak_idx]),
        "peak_height": float(b1_values[peak_idx]),
    }


def main() -> int:
    """CLI entry point. See module docstring for usage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 17, 42, 73, 101],
        help="Seed list (default = Q3 pre-registered set).",
    )
    parser.add_argument(
        "--lock-after",
        type=int,
        nargs="+",
        default=[50, 75, 100, 125, 150],
        help="LOCK_AFTER values to sweep per seed.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/v0.5_amedi_curve_multiseed.json"),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Parent dir for per-(seed, lock_after) grid run dirs.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Where per-cell aggregate JSON files land.",
    )
    parser.add_argument(
        "--studyforrest-data",
        type=Path,
        default=Path(
            os.environ.get(
                "BOUBA_SENS_STUDYFORREST_DATA",
                "data/sf_phase2_adapted",
            )
        ),
    )
    args = parser.parse_args()

    args.runs_root.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for seed in args.seeds:
        print(f"=== seed {seed} ===", flush=True)
        try:
            result = run_single_seed(
                seed=seed,
                lock_after_values=args.lock_after,
                runs_root=args.runs_root,
                reports_dir=args.reports_dir,
                studyforrest_data=args.studyforrest_data,
            )
        except (subprocess.CalledProcessError, OSError, KeyError) as exc:
            print(f"seed {seed} failed: {exc}", file=sys.stderr)
            return 1
        results.append(result)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "seeds": list(args.seeds),
                "lock_after_values": list(args.lock_after),
                "results": results,
            },
            indent=2,
        )
    )
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
