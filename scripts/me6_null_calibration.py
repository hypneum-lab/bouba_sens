"""Me6 null calibration — what does B-3 look like when nothing is there?

The B-3 verdict (`scripts/aggregate_grid.py`) declares "passes" when the
median, over the K = 30 (seed, timing, SNR) trios of a grid, of
`me6_max_abs_off_diag(me6_asymmetry(M))` exceeds `B3_ME6_THRESHOLD = 0.02`.
Each entry of the 5x5 matrix `M` is a `query_accuracy` estimated on a
probe batch of `batch_size = 64` samples (`src/bouba_sens/loop.py:306-337`,
`src/bouba_sens/cli.py:374`). Me6 is the maximum of the 10 absolute
off-diagonal differences `|M[i, j] - M[j, i]|`.

This script simulates the null hypothesis "every entry of `M` has the same
true accuracy `p`": the 25 entries are independent `Binomial(n, p) / n`
draws. It reports the null distribution of Me6 per trio and of the
median-over-K-trios statistic actually used for the verdict, for several
probe sizes `n` and base accuracies `p`, and the smallest `n` at which each
historically observed B-3 median would exceed the null P99.

It does not re-run any cell, does not touch checkpoints, and modifies no
metric, verdict or threshold. numpy only.

Usage:
    uv run python scripts/me6_null_calibration.py \
        --replicates 20000 --out reports/me6_null_calibration.json
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "bouba_sens.me6_null_calibration.v1"

# Spec §1.2 B-3 threshold, as applied in scripts/aggregate_grid.py:28.
B3_ME6_THRESHOLD = 0.02
N_MODALITIES = 5
# scripts/run_grid.sh: 5 seeds x 2 timings x 3 SNR = 30 trios per grid
# (= `cells_counted` for b3 in every 150-cell aggregate).
DEFAULT_CELLS = 30
# AdaptationLoop.query_accuracy default (src/bouba_sens/loop.py:310).
PROBE_BATCH = 64

DEFAULT_N_GRID: tuple[int, ...] = (64, 256, 1024, 4096)
DEFAULT_FINE_N_GRID: tuple[int, ...] = (
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    8192,
)
DEFAULT_P_GRID: tuple[float, ...] = (0.2, 0.25, 0.3, 0.5, 0.7, 0.9)

# Historically reported B-3 medians (median over 30 trios, n = 64 per
# entry). Values are copied verbatim from the cited ADRs; none is
# recomputed here.
DEFAULT_OBSERVED: tuple[tuple[str, float, str], ...] = (
    ("gaussian", 0.1484, "ADR-0004 (v0.2 GaussianWorld grid, 30 trios)"),
    ("xor", 0.1406, "ADR-0005-v03 (v0.3 XORWorld grid, 30 trios)"),
    ("sinusoid", 0.1406, "ADR-0005-v03 (v0.3 SinusoidWorld grid, 30 trios)"),
    ("mock", 0.3125, "ADR-0008 (v0.4 Studyforrest mock grid, 30 trios)"),
    ("ecg", 0.4453, "ADR-0009 (v0.4 real MIT-BIH ECG grid, 30 trios; headline retracted)"),
)

# Only tracked aggregate carrying per-cell me1 means (query_accuracy on
# the audio query, n = 64, averaged over 5 seeds) and its own B-3 median.
DEFAULT_AGGREGATE = Path("reports/v0.5_s14c_joint_aggregate.json")

_IU, _JU = np.triu_indices(N_MODALITIES, k=1)


def me6_max_abs_off_diag(perf: np.ndarray) -> np.ndarray:
    """numpy twin of `me6_max_abs_off_diag(me6_asymmetry(M))`.

    `perf` has shape `(..., 5, 5)`; returns shape `(...)`. Uses the 10
    unordered off-diagonal pairs, which is exactly the set of non-zero
    entries of the antisymmetric matrix `M - M.T`.
    """
    if perf.shape[-2:] != (N_MODALITIES, N_MODALITIES):
        raise ValueError(f"perf must end with ({N_MODALITIES}, {N_MODALITIES}); got {perf.shape}")
    diff = np.abs(perf[..., _IU, _JU] - perf[..., _JU, _IU])
    return diff.max(axis=-1)


def _rng_for(seed: int, n: int, p: float) -> np.random.Generator:
    """One independent stream per (n, p) so results do not depend on order."""
    return np.random.default_rng([seed, n, round(p * 1_000_000)])


def simulate_null(
    n: int,
    p: float,
    *,
    replicates: int,
    cells: int,
    seed: int,
    chunk: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate `replicates` grids of `cells` trios under H0.

    Returns `(me6_per_trio, median_over_cells)` with shapes
    `(replicates * cells,)` and `(replicates,)`.
    """
    if n <= 0 or not 0.0 < p < 1.0:
        raise ValueError(f"need n > 0 and 0 < p < 1; got n={n}, p={p}")
    rng = _rng_for(seed, n, p)
    per_trio: list[np.ndarray] = []
    medians: list[np.ndarray] = []
    remaining = replicates
    while remaining > 0:
        g = min(chunk, remaining)
        counts = rng.binomial(n, p, size=(g, cells, N_MODALITIES, N_MODALITIES))
        me6 = me6_max_abs_off_diag(counts / n)  # (g, cells)
        per_trio.append(me6.reshape(-1))
        medians.append(np.median(me6, axis=1))
        remaining -= g
    return np.concatenate(per_trio), np.concatenate(medians)


def summarise(values: np.ndarray, *, threshold: float = B3_ME6_THRESHOLD) -> dict[str, float]:
    """mean / median / P95 / P99 plus the fraction above the B-3 threshold.

    Quantiles use `inverted_cdf` so that P99 is an attained value of the
    discrete distribution (multiples of 1/n or 1/(2n)).
    """
    q95, q99 = np.quantile(values, [0.95, 0.99], method="inverted_cdf")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(q95),
        "p99": float(q99),
        "frac_above_threshold": float((values > threshold).mean()),
    }


def read_aggregate(path: Path) -> dict[str, Any]:
    """Pull per-cell me1 means, K and the B-3 median from an aggregate JSON.

    Returns `{"found": False, "reason": ...}` when the file is absent or
    does not carry the expected keys; nothing is invented.
    """
    if not path.exists():
        return {"found": False, "source": str(path), "reason": "file not found"}
    body = json.loads(path.read_text())
    cells = body.get("cells", {})
    me1 = {
        name: float(metrics["me1"]["mean"])
        for name, metrics in cells.items()
        if isinstance(metrics, dict) and "me1" in metrics and "mean" in metrics["me1"]
    }
    if not me1:
        return {"found": False, "source": str(path), "reason": "no cells[*].me1.mean"}
    b3 = body.get("invariants", {}).get("b3", {})
    vals = np.array(list(me1.values()))
    return {
        "found": True,
        "source": str(path),
        "n_cells": len(me1),
        "min": float(vals.min()),
        "mean": float(vals.mean()),
        "max": float(vals.max()),
        "per_cell": me1,
        "b3_median_me6_max_abs": b3.get("median_me6_max_abs"),
        "b3_cells_counted": b3.get("cells_counted"),
    }


def n_min_for(
    value: float,
    p: float,
    table: dict[tuple[int, float], dict[str, Any]],
    *,
    key: str,
) -> int | None:
    """Smallest n in `table` (for this p) whose null P99 is strictly below `value`."""
    for n in sorted(n for (n, pp) in table if pp == p):
        if value > table[(n, p)][key]["p99"]:
            return n
    return None


def calibrate(
    *,
    replicates: int,
    fine_replicates: int,
    cells: int,
    seed: int,
    n_grid: Sequence[int],
    fine_n_grid: Sequence[int],
    p_grid: Sequence[float],
    observed: Iterable[tuple[str, float, str]],
    real_me1: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the whole calibration and return a JSON-serialisable dict."""
    t0 = time.perf_counter()
    p_values = sorted({round(float(p), 6) for p in p_grid})
    p_real: list[float] = []
    if real_me1 and real_me1.get("found"):
        p_real = sorted({round(real_me1[k], 4) for k in ("min", "mean", "max")})
        p_values = sorted(set(p_values) | set(p_real))

    main_n = sorted(set(n_grid))
    all_n = sorted(set(main_n) | set(fine_n_grid))

    table: dict[tuple[int, float], dict[str, Any]] = {}
    for n in all_n:
        reps = replicates if n in main_n else fine_replicates
        for p in p_values:
            per_trio, med = simulate_null(n, p, replicates=reps, cells=cells, seed=seed)
            table[(n, p)] = {
                "n": n,
                "p": p,
                "replicates": reps,
                "in_main_grid": n in main_n,
                "me6_per_trio": summarise(per_trio),
                "me6_median_over_cells": summarise(med),
            }

    obs_list = [{"name": name, "value": float(v), "source": src} for name, v, src in observed]
    n_min_rows = [
        {
            "name": o["name"],
            "value": o["value"],
            "p": p,
            "n_min_median_statistic": n_min_for(o["value"], p, table, key="me6_median_over_cells"),
            "n_min_per_trio": n_min_for(o["value"], p, table, key="me6_per_trio"),
        }
        for o in obs_list
        for p in p_values
    ]

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "params": {
            "replicates": replicates,
            "fine_replicates": fine_replicates,
            "cells": cells,
            "seed": seed,
            "n_grid": main_n,
            "fine_n_grid": sorted(set(fine_n_grid)),
            "p_grid": sorted({round(float(p), 6) for p in p_grid}),
            "p_from_real_me1": p_real,
            "threshold": B3_ME6_THRESHOLD,
            "n_modalities": N_MODALITIES,
            "probe_batch_in_production": PROBE_BATCH,
        },
        "real_me1": real_me1 if real_me1 is not None else {"found": False, "reason": "not read"},
        "observed": obs_list,
        "null": [table[k] for k in sorted(table)],
        "n_min": n_min_rows,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


def _print_report(result: dict[str, Any]) -> None:
    params = result["params"]
    print(
        f"Me6 null calibration — K={params['cells']} trios, "
        f"{params['replicates']} grids per (n, p) on the main grid, seed={params['seed']}"
    )
    print()
    print(
        f"{'n':>6} {'p':>6} | {'Me6/trio mean':>13} {'P95':>7} {'P99':>7} | "
        f"{'median-of-K mean':>16} {'P95':>7} {'P99':>7} {'P(>0.02)':>9}"
    )
    print("-" * 92)
    for row in result["null"]:
        if not row["in_main_grid"]:
            continue
        t, m = row["me6_per_trio"], row["me6_median_over_cells"]
        print(
            f"{row['n']:>6} {row['p']:>6.3f} | {t['mean']:>13.4f} {t['p95']:>7.4f} "
            f"{t['p99']:>7.4f} | {m['mean']:>16.4f} {m['p95']:>7.4f} {m['p99']:>7.4f} "
            f"{m['frac_above_threshold']:>9.4f}"
        )
    print()
    print("Smallest n at which the observed median would exceed the null P99")
    print(f"{'world':>10} {'value':>7} {'p':>6} | {'median-of-K':>12} {'per trio':>9}")
    print("-" * 56)
    for row in result["n_min"]:
        nm = row["n_min_median_statistic"]
        nt = row["n_min_per_trio"]
        print(
            f"{row['name']:>10} {row['value']:>7.4f} {row['p']:>6.3f} | "
            f"{str(nm) if nm is not None else '>max':>12} "
            f"{str(nt) if nt is not None else '>max':>9}"
        )
    rm = result["real_me1"]
    print()
    if rm.get("found"):
        print(
            f"real me1 from {rm['source']}: {rm['n_cells']} cells, "
            f"min {rm['min']:.4f} / mean {rm['mean']:.4f} / max {rm['max']:.4f}"
        )
    else:
        print(f"real me1: not found ({rm.get('reason')})")
    print(f"elapsed {result['elapsed_s']} s")


def _parse_observed(items: Sequence[str]) -> list[tuple[str, float, str]]:
    out: list[tuple[str, float, str]] = []
    for item in items:
        name, _, value = item.partition("=")
        if not name or not value:
            raise argparse.ArgumentTypeError(f"--observed expects name=value, got {item!r}")
        out.append((name, float(value), "command line"))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--replicates", type=int, default=20_000, help="grids per (n, p), main n")
    parser.add_argument(
        "--fine-replicates", type=int, default=5_000, help="grids per (n, p), extra n for n_min"
    )
    parser.add_argument(
        "--cells",
        type=int,
        default=None,
        help=f"trios per grid (K); default: b3.cells_counted of --aggregate, else {DEFAULT_CELLS}",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, nargs="+", default=list(DEFAULT_N_GRID))
    parser.add_argument("--fine-n", type=int, nargs="+", default=list(DEFAULT_FINE_N_GRID))
    parser.add_argument("--p", type=float, nargs="+", default=list(DEFAULT_P_GRID))
    parser.add_argument(
        "--observed",
        nargs="*",
        default=None,
        help="name=value pairs replacing the built-in ADR values",
    )
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=DEFAULT_AGGREGATE,
        help="aggregate_grid JSON to read real me1 means and K from (missing file is reported)",
    )
    parser.add_argument("--no-aggregate", action="store_true", help="do not read --aggregate")
    parser.add_argument("--out", type=Path, default=Path("reports/me6_null_calibration.json"))
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    real_me1 = None if args.no_aggregate else read_aggregate(args.aggregate)
    cells = args.cells
    if cells is None:
        cells = DEFAULT_CELLS
        if real_me1 and real_me1.get("found") and real_me1.get("b3_cells_counted"):
            cells = int(real_me1["b3_cells_counted"])

    observed: list[tuple[str, float, str]] = (
        list(DEFAULT_OBSERVED) if args.observed is None else _parse_observed(args.observed)
    )
    if real_me1 and real_me1.get("found") and real_me1.get("b3_median_me6_max_abs") is not None:
        observed.append(
            (
                Path(real_me1["source"]).stem,
                float(real_me1["b3_median_me6_max_abs"]),
                f"{real_me1['source']} invariants.b3.median_me6_max_abs",
            )
        )

    result = calibrate(
        replicates=args.replicates,
        fine_replicates=args.fine_replicates,
        cells=cells,
        seed=args.seed,
        n_grid=args.n,
        fine_n_grid=args.fine_n,
        p_grid=args.p,
        observed=observed,
        real_me1=real_me1,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if not args.quiet:
        _print_report(result)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
