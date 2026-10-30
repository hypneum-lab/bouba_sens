"""B-3 probe v2 — positive control, block-length rule and preregistered verdict.

Implements the analysis fixed by `docs/prereg/2026-10-25-b3-probe-v2.md`
(ADR-0023). Three subcommands, numpy only on top of the saved per-cell files
written by `bouba-sens train` / `bouba-sens lesion` (probe v2):

- `control --root GRID` — positive control. For every intact network
  (`phase1_seed*/probe_v2_meta.json`), held-out accuracy of the v2 probe on
  the control modality minus the majority-class baseline; the control passes
  when the mean margin over seeds is >= 0.10. The untrained reference
  (`init_probe_v2_meta.json`) is reported, descriptive only.
- `block --root PILOT --world W` — block-length rule on a PILOT grid: lag-k
  autocorrelation of the paired differences d_i (mean over series with
  non-zero variance, the estimator of ADR-0022), k = 1 .. n / 4. Block =
  1 for an iid world; otherwise the smallest k with ACF(k) < 0.1, rounded up
  to a power of two (k = 1 -> block 1); n / 4 if the ACF never falls below
  0.1.
- `report --root GRID --world W --block B` — control + paired sign-flip null
  of `scripts/me6_paired_null.py` on `per_query_successes_v2.json` with the
  preregistered block, and the verdict of the preregistration.

Usage:
    uv run python scripts/b3_probe_v2.py block --root PILOT --world mock --out pilot.json
    uv run python scripts/b3_probe_v2.py report --root GRID --world gaussian --block 1 \\
        --out reports/b3_probe_v2_gaussian.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from me6_paired_null import (
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    analyse,
    discover_trios,
    pair_differences,
    stack_successes,
)

SCHEMA = "bouba_sens.b3_probe_v2.v1"
SUCCESSES_V2 = "per_query_successes_v2.json"
META_V2 = "probe_v2_meta.json"
INIT_META_V2 = "init_probe_v2_meta.json"

# Preregistered constants (docs/prereg/2026-10-25-b3-probe-v2.md).
CONTROL_MODALITY = "vision"
CONTROL_MARGIN = 0.10
ACF_THRESHOLD = 0.1
ALPHA_PRESENT = 0.01
ALPHA_COMPATIBLE = 0.05
IID_WORLDS = frozenset({"gaussian", "xor", "sinusoid"})
WORLDS = frozenset({"gaussian", "xor", "sinusoid", "mock"})
EXPECTED_TRIOS = 30

_PHASE1_RE = re.compile(r"^phase1_seed(?P<seed>\d+)$")


def acf_mean(series: np.ndarray, lags: list[int]) -> tuple[dict[int, float], int]:
    """Mean over series of the lag-k autocorrelation; `series` is `(S, n)`.

    Each series is centred on its own mean; series with zero variance carry
    no autocorrelation and are dropped. Returns `({k: acf}, n_series_kept)`;
    the dict is empty when no series has variance.
    """
    x = np.asarray(series, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected (S, n); got {x.shape}")
    x = x - x.mean(axis=-1, keepdims=True)
    var = (x**2).mean(axis=-1)
    keep = var > 0
    x, var = x[keep], var[keep]
    if x.shape[0] == 0:
        return {}, 0
    out: dict[int, float] = {}
    for k in lags:
        if k <= 0 or k >= x.shape[1]:
            raise ValueError(f"lag {k} out of range for n = {x.shape[1]}")
        out[k] = float(((x[:, :-k] * x[:, k:]).mean(axis=-1) / var).mean())
    return out, int(x.shape[0])


def block_from_acf(acf: dict[int, float], *, n_items: int) -> tuple[int, int | None]:
    """Preregistered rule: smallest lag with ACF < 0.1, rounded up to a power of 2.

    Returns `(block, k_star)`; `k_star` is None when the ACF never falls below
    the threshold over the lags given, in which case the block is n / 4.
    """
    for k in sorted(acf):
        if acf[k] < ACF_THRESHOLD:
            return (1 if k == 1 else 2 ** math.ceil(math.log2(k))), k
    return max(1, n_items // 4), None


def block_rule(root: Path, *, world: str, successes_file: str = SUCCESSES_V2) -> dict[str, Any]:
    """Apply the block-length rule to the v2 outcomes of a pilot grid.

    `successes_file` selects another per-cell outcome file with the same
    layout (ADR-0026: the v1 probe at several sizes); the default keeps the
    ADR-0023 behaviour.
    """
    if world not in WORLDS:
        raise ValueError(f"unknown world {world!r}")
    trios, incomplete = discover_trios(root, successes_file=successes_file)
    successes = stack_successes(trios)  # (K, 5, 5, n)
    n_items = int(successes.shape[-1])
    differences = pair_differences(successes)  # (K, 10, n)
    lags = list(range(1, n_items // 4 + 1))
    acf_d, kept_d = acf_mean(differences.reshape(-1, n_items), lags)
    acf_s, kept_s = acf_mean(successes.reshape(-1, n_items), lags)
    if world in IID_WORLDS:
        block, k_star, rule = 1, None, "iid world: block = 1 by design"
    elif not acf_d:
        block, k_star, rule = 1, None, "no paired difference has variance: block = 1"
    else:
        block, k_star = block_from_acf(acf_d, n_items=n_items)
        rule = (
            f"smallest lag with ACF(d) < {ACF_THRESHOLD} = {k_star}, rounded up to a power of 2"
            if k_star is not None
            else f"ACF(d) never < {ACF_THRESHOLD} up to n/4: block = n/4"
        )
    return {
        "root": str(root),
        "world": world,
        "trios": len(trios),
        "incomplete_trios": incomplete,
        "n_items": n_items,
        "acf_paired_differences": {str(k): round(v, 4) for k, v in acf_d.items()},
        "acf_paired_differences_series": kept_d,
        "acf_successes": {str(k): round(v, 4) for k, v in acf_s.items()},
        "acf_successes_series": kept_s,
        "k_star": k_star,
        "block": block,
        "rule": rule,
    }


def positive_control(root: Path) -> dict[str, Any]:
    """Control-modality margin over the majority baseline on intact networks."""
    seeds: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        match = _PHASE1_RE.match(path.name)
        if not match or not (path / META_V2).exists():
            continue
        meta = json.loads((path / META_V2).read_text())["modalities"]
        row: dict[str, Any] = {
            "seed": int(match["seed"]),
            "margins": {m: v["accuracy"] - v["baseline"] for m, v in meta.items()},
            "accuracy": {m: v["accuracy"] for m, v in meta.items()},
            "baseline": meta[CONTROL_MODALITY]["baseline"],
            "n_codes_test": {m: v["n_codes_test"] for m, v in meta.items()},
        }
        init_path = path / INIT_META_V2
        if init_path.exists():
            init = json.loads(init_path.read_text())["modalities"]
            row["init_margins"] = {m: v["accuracy"] - v["baseline"] for m, v in init.items()}
        seeds.append(row)
    if not seeds:
        raise ValueError(f"no phase1_seed*/{META_V2} under {root}")
    margins = [s["margins"][CONTROL_MODALITY] for s in seeds]
    mean_margin = float(np.mean(margins))
    out: dict[str, Any] = {
        "modality": CONTROL_MODALITY,
        "threshold_margin": CONTROL_MARGIN,
        "n_seeds": len(seeds),
        "mean_accuracy": float(np.mean([s["accuracy"][CONTROL_MODALITY] for s in seeds])),
        "mean_baseline": float(np.mean([s["baseline"] for s in seeds])),
        "mean_margin": mean_margin,
        "passes": mean_margin >= CONTROL_MARGIN,
        "mean_margin_per_modality": {
            m: float(np.mean([s["margins"][m] for s in seeds])) for m in seeds[0]["margins"]
        },
        "seeds": seeds,
    }
    if all("init_margins" in s for s in seeds):
        out["untrained_reference_mean_margin_per_modality"] = {
            m: float(np.mean([s["init_margins"][m] for s in seeds]))
            for m in seeds[0]["init_margins"]
        }
    return out


def verdict(*, control_passes: bool, complete: bool, p_value: float) -> str:
    """Preregistered decision rule."""
    if not control_passes:
        return "inconclusive: positive control failed"
    if not complete:
        return "inconclusive: incomplete grid"
    if p_value < ALPHA_PRESENT:
        return "asymmetry present"
    if p_value >= ALPHA_COMPATIBLE:
        return "compatible with the null"
    return "inconclusive: 0.01 <= p < 0.05"


def report(root: Path, *, world: str, block: int, replicates: int, seed: int) -> dict[str, Any]:
    """Positive control + paired null on the v2 outcomes + verdict."""
    control = positive_control(root)
    trios, incomplete = discover_trios(root, successes_file=SUCCESSES_V2)
    paired = analyse(
        trios,
        replicates=replicates,
        seed=seed,
        block=block,
        incomplete=incomplete,
        root=root,
        successes_file=SUCCESSES_V2,
    )
    complete = len(trios) == EXPECTED_TRIOS and not incomplete
    med = paired["median"]
    successes = stack_successes(trios)
    per_query = {
        q: float(successes[:, i].mean())
        for i, q in enumerate(("audio", "vision", "tactile", "gravity", "force"))
    }
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "world": world,
        "prereg": "docs/prereg/2026-10-25-b3-probe-v2.md (tag prereg-b3-probe-v2-20261025)",
        "block": block,
        "positive_control": control,
        "b3_v2": {
            "statistic": "median over trios of max_{q != m} |P[q, m] - P[m, q]|",
            "observed": med["observed"],
            "null_mean": med["null"]["mean"],
            "null_p99": med["null"]["p99"],
            "p_value": med["p_value"],
            "trios": len(trios),
            "complete": complete,
            "mean_accuracy_per_query": per_query,
            "verdict": verdict(
                control_passes=bool(control["passes"]),
                complete=complete,
                p_value=float(med["p_value"]),
            ),
        },
        "paired_null": paired,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_control = sub.add_parser("control", help="positive control on intact networks")
    p_control.add_argument("--root", type=Path, required=True)
    p_control.add_argument("--out", type=Path)
    p_block = sub.add_parser("block", help="block-length rule on a pilot grid")
    p_block.add_argument("--root", type=Path, required=True)
    p_block.add_argument("--world", required=True, choices=sorted(WORLDS))
    p_block.add_argument("--out", type=Path)
    p_report = sub.add_parser("report", help="control + paired null + verdict")
    p_report.add_argument("--root", type=Path, required=True)
    p_report.add_argument("--world", required=True, choices=sorted(WORLDS))
    p_report.add_argument("--block", type=int, required=True)
    p_report.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    p_report.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_report.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "control":
        result = positive_control(args.root)
    elif args.cmd == "block":
        result = block_rule(args.root, world=args.world)
    else:
        result = report(
            args.root,
            world=args.world,
            block=args.block,
            replicates=args.replicates,
            seed=args.seed,
        )
    text = json.dumps(result, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    summary = result.get("b3_v2", result)
    printable = {k: v for k, v in summary.items() if k not in ("seeds", "incomplete_trios")}
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
