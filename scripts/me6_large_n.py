"""Me6 at large probe sizes — paired null per n and power curve (ADR-0026).

Analysis fixed by ADR-0026 (preregistered, tag `prereg-me6-large-n-*`).
numpy only, on top of the per-cell files written by
`bouba-sens lesion --probe-sizes 64,256,1024,4096` (v1 zero-masking probe,
`per_query_successes_n{N}.json`) and the historical n = 64 file
(`per_query_successes.json`, one probe seed per cell). Three subcommands:

- `blocks --root PILOT --world W --sizes 64,256,1024,4096` — the block-length
  rule of ADR-0023 (`b3_probe_v2.block_rule`) applied to each per-size file
  and to the historical file of a PILOT grid. Gaussian: block 1 by design.
- `report --root GRID --world W --blocks 64:1,256:1,...,legacy:1` — for each
  size: observed median over the 30 trios of Me6, paired sign-flip null
  (`me6_paired_null.analyse`) with the committed block, p value, frequency
  with which the frozen rule "median > 0.02" fires under the null, the
  ADR-0021 iid reference at the same n, and the reading of ADR-0026.
- `power --sizes ... --p 0.25 --deltas ...` — power curve under the ADR-0021
  iid binomial reference: one pair (audio | vision) carries a true
  asymmetry delta in every trio; power = fraction of grids with paired-test
  p < 0.01 against the iid null at the same n, and fraction of grids where
  the frozen rule "median > 0.02" fires.

Usage:
    uv run python scripts/me6_large_n.py blocks --root PILOT --world mock \\
        --sizes 64,256,1024,4096 --out reports/me6_large_n_pilot_mock.json
    uv run python scripts/me6_large_n.py report --root GRID --world gaussian \\
        --blocks 64:1,256:1,1024:1,4096:1,legacy:1 --out reports/me6_large_n_gaussian.json
    uv run python scripts/me6_large_n.py power --sizes 64,256,1024,4096 --p 0.25 \\
        --out reports/me6_large_n_power.json
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3_probe_v2 import IID_WORLDS, WORLDS, block_rule
from me6_null_calibration import N_MODALITIES, me6_max_abs_off_diag
from me6_paired_null import (
    B3_ME6_THRESHOLD,
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    SUCCESSES_FILE,
    analyse,
    discover_trios,
)

SCHEMA = "bouba_sens.me6_large_n.v1"
LEGACY = "legacy"
DEFAULT_SIZES: tuple[int, ...] = (64, 256, 1024, 4096)
DEFAULT_DELTAS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15)
ALPHA_EXCESS = 0.01
ALPHA_COMPATIBLE = 0.05
EXPECTED_TRIOS = 30
# Pair carrying the planted asymmetry in the power simulation (audio | vision).
POWER_PAIR = (0, 1)


def successes_file_for(size: int | str) -> str:
    """Per-cell outcome file of one probe size; `legacy` = historical n = 64."""
    if size == LEGACY:
        return SUCCESSES_FILE
    return f"per_query_successes_n{int(size)}.json"


def parse_sizes(raw: str) -> list[int]:
    sizes = sorted({int(v) for v in raw.split(",") if v.strip()})
    if not sizes or any(n <= 0 for n in sizes):
        raise ValueError(f"need positive sizes; got {raw!r}")
    return sizes


def parse_blocks(raw: str) -> dict[str, int]:
    """`"64:1,256:1,legacy:1"` -> `{"64": 1, "256": 1, "legacy": 1}`."""
    out: dict[str, int] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        key, _, value = part.partition(":")
        key = key.strip()
        if key != LEGACY:
            key = str(int(key))
        block = int(value)
        if block < 1:
            raise ValueError(f"block must be >= 1; got {part!r}")
        out[key] = block
    if not out:
        raise ValueError("no block given")
    return out


def reading(*, complete: bool, p_value: float) -> str:
    """ADR-0026 reading of one (world, size) row, applied in this order."""
    if not complete:
        return "inconclusive: incomplete grid"
    if p_value < ALPHA_EXCESS:
        return "excess over the paired null"
    if p_value >= ALPHA_COMPATIBLE:
        return "compatible with the paired null"
    return "inconclusive"


def blocks(root: Path, *, world: str, sizes: Sequence[int]) -> dict[str, Any]:
    """Block-length rule of ADR-0023 on each per-size file of a pilot grid."""
    rows: dict[str, Any] = {}
    for key in [*(str(n) for n in sizes), LEGACY]:
        name = successes_file_for(key)
        row = block_rule(root, world=world, successes_file=name)
        row["successes_file"] = name
        rows[key] = row
    return {
        "schema": SCHEMA,
        "kind": "blocks",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "root": str(root),
        "world": world,
        "blocks": {k: int(v["block"]) for k, v in rows.items()},
        "rows": rows,
    }


def report(
    root: Path,
    *,
    world: str,
    blocks_by_size: dict[str, int],
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Paired null per size, the frozen 0.02 rule for comparison, iid reference."""
    if world not in WORLDS:
        raise ValueError(f"unknown world {world!r}")
    if world in IID_WORLDS and any(b != 1 for b in blocks_by_size.values()):
        raise ValueError("iid world: every block must be 1 (ADR-0023 rule)")
    rows: dict[str, Any] = {}
    for key, block in blocks_by_size.items():
        name = successes_file_for(key)
        trios, incomplete = discover_trios(root, successes_file=name)
        if not trios:
            rows[key] = {"successes_file": name, "trios": 0, "reading": "not run"}
            continue
        res = analyse(
            trios,
            replicates=replicates,
            seed=seed,
            incomplete=incomplete,
            root=root,
            block=block,
            successes_file=name,
        )
        med = res["median"]
        complete = bool(res["params"]["complete"]) and len(trios) == EXPECTED_TRIOS
        ref = res.get("binomial_reference", {})
        rows[key] = {
            "successes_file": name,
            "n_items": res["params"]["n_items"],
            "block": block,
            "trios": len(trios),
            "complete": complete,
            "grand_mean_accuracy": res["grand_mean_accuracy"],
            "observed_median": med["observed"],
            "null": med["null"],
            "p_value": med["p_value"],
            "frac_ge_observed": med["frac_ge_observed"],
            "rule_0p02_observed_passes": med["passes_fixed_threshold"],
            "rule_0p02_rate_under_null": med["null"]["frac_above_threshold"],
            "iid_reference_median": ref.get("me6_median_over_cells"),
            "iid_reference_tail": ref.get("median_tail"),
            "trios_with_p_below_0p05": int(
                sum(1 for c in res["cells"] if float(c["p_value"]) < 0.05)
            ),
            "reading": reading(complete=complete, p_value=float(med["p_value"])),
        }
    return {
        "schema": SCHEMA,
        "kind": "report",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "root": str(root),
        "world": world,
        "replicates": replicates,
        "seed": seed,
        "threshold_frozen_rule": B3_ME6_THRESHOLD,
        "rows": rows,
    }


def _simulate_medians(
    rng: np.random.Generator, *, n: int, p: float, delta: float, replicates: int, cells: int
) -> np.ndarray:
    """Median over `cells` trios of Me6 for iid binomial entries, one planted pair."""
    probs = np.full((N_MODALITIES, N_MODALITIES), p)
    a, b = POWER_PAIR
    probs[a, b] = p + delta / 2
    probs[b, a] = p - delta / 2
    if probs.min() <= 0 or probs.max() >= 1:
        raise ValueError(f"p +/- delta/2 out of (0, 1): p={p}, delta={delta}")
    out = np.empty(replicates)
    chunk = 1000
    for start in range(0, replicates, chunk):
        g = min(chunk, replicates - start)
        counts = rng.binomial(n, probs, size=(g, cells, N_MODALITIES, N_MODALITIES))
        out[start : start + g] = np.median(me6_max_abs_off_diag(counts / n), axis=1)
    return out


def power(
    *,
    sizes: Sequence[int],
    p: float,
    deltas: Sequence[float],
    null_replicates: int = 20_000,
    alt_replicates: int = 2_000,
    cells: int = EXPECTED_TRIOS,
    seed: int = DEFAULT_SEED,
    alpha: float = ALPHA_EXCESS,
) -> dict[str, Any]:
    """Power of the paired test (iid null) and of the frozen 0.02 rule, per n."""
    rows: list[dict[str, Any]] = []
    for n in sizes:
        null = np.sort(
            _simulate_medians(
                np.random.default_rng([seed, n, 0]),
                n=n,
                p=p,
                delta=0.0,
                replicates=null_replicates,
                cells=cells,
            )
        )
        q99 = float(np.quantile(null, 0.99, method="inverted_cdf"))
        row: dict[str, Any] = {
            "n": n,
            "null_mean": float(null.mean()),
            "null_p99": q99,
            "rule_0p02_rate_under_null": float((null > B3_ME6_THRESHOLD).mean()),
            "alternatives": [],
        }
        for i, delta in enumerate(deltas, start=1):
            alt = _simulate_medians(
                np.random.default_rng([seed, n, i]),
                n=n,
                p=p,
                delta=delta,
                replicates=alt_replicates,
                cells=cells,
            )
            count_ge = null.size - np.searchsorted(null, alt, side="left")
            p_values = (count_ge + 1) / (null.size + 1)
            row["alternatives"].append(
                {
                    "delta": delta,
                    "median_mean": float(alt.mean()),
                    "power_paired_test": float((p_values < alpha).mean()),
                    "rule_0p02_fires": float((alt > B3_ME6_THRESHOLD).mean()),
                }
            )
        rows.append(row)
    return {
        "schema": SCHEMA,
        "kind": "power",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reference": "ADR-0021 iid binomial entries; one planted pair (audio|vision)",
        "p": p,
        "deltas": list(deltas),
        "alpha": alpha,
        "cells": cells,
        "null_replicates": null_replicates,
        "alt_replicates": alt_replicates,
        "seed": seed,
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_blocks = sub.add_parser("blocks", help="block-length rule per size on a pilot grid")
    p_blocks.add_argument("--root", type=Path, required=True)
    p_blocks.add_argument("--world", required=True, choices=sorted(WORLDS))
    p_blocks.add_argument("--sizes", default=",".join(str(n) for n in DEFAULT_SIZES))
    p_blocks.add_argument("--out", type=Path)
    p_report = sub.add_parser("report", help="paired null per size")
    p_report.add_argument("--root", type=Path, required=True)
    p_report.add_argument("--world", required=True, choices=sorted(WORLDS))
    p_report.add_argument("--blocks", required=True, help='e.g. "64:1,256:1,legacy:1"')
    p_report.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    p_report.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_report.add_argument("--out", type=Path, required=True)
    p_power = sub.add_parser("power", help="power curve under the iid reference")
    p_power.add_argument("--sizes", default=",".join(str(n) for n in DEFAULT_SIZES))
    p_power.add_argument("--p", type=float, default=0.25)
    p_power.add_argument("--deltas", default=",".join(str(d) for d in DEFAULT_DELTAS))
    p_power.add_argument("--null-replicates", type=int, default=20_000)
    p_power.add_argument("--alt-replicates", type=int, default=2_000)
    p_power.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_power.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "blocks":
        result = blocks(args.root, world=args.world, sizes=parse_sizes(args.sizes))
        printable: Any = result["blocks"]
    elif args.cmd == "report":
        result = report(
            args.root,
            world=args.world,
            blocks_by_size=parse_blocks(args.blocks),
            replicates=args.replicates,
            seed=args.seed,
        )
        printable = {
            k: {
                f: v.get(f)
                for f in ("observed_median", "p_value", "rule_0p02_rate_under_null", "reading")
            }
            for k, v in result["rows"].items()
        }
    else:
        result = power(
            sizes=parse_sizes(args.sizes),
            p=args.p,
            deltas=[float(d) for d in args.deltas.split(",") if d.strip()],
            null_replicates=args.null_replicates,
            alt_replicates=args.alt_replicates,
            seed=args.seed,
        )
        printable = result["rows"]
    text = json.dumps(result, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
