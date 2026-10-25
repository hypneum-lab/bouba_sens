"""Me6 paired null — the B-3 median against a sign-flip null on the saved per-item outcomes.

## What is tested

The B-3 verdict (`scripts/aggregate_grid.py`) takes, per (seed, timing, SNR)
trio, the 5x5 matrix `P[q, m]` = accuracy of the model lesioned on `m` when
queried on `q` alone (rows = query, columns = lesioned modality, as in
`aggregate_grid._compute_me6`), computes `Me6 = max_{q != m} |P[q, m] - P[m, q]|`
and declares PASS when the median of Me6 over the K = 30 trios of a grid
exceeds 0.02. Each accuracy is the mean of n = 64 per-item outcomes
(`AdaptationLoop.query_successes`, written by `bouba-sens lesion` to
`per_query_successes.json`, ADR-0022).

## Null hypothesis (H0)

Symmetry of the expected performances: `E[P[q, m]] = E[P[m, q]]` for every
pair. Under H0 the per-item paired difference `d_i = s_qm(i) - s_mq(i)` is
symmetric about zero, so its sign carries no information.

## Procedure (paired sign-flip)

For every complete trio, every unordered pair (m, q) with m < q, and every
probe item i, swap `s_mq(i) <-> s_qm(i)` with probability 1/2 — i.e. flip the
sign of `d_i`. Recompute P, Me6 per trio, then the median over trios.
Repeat B = 20 000 times (seed 25102026, numpy only). The null keeps the
observed per-item outcomes (|d_i|, the discordant items, the empirical
accuracies) and the within-column sharing of the probe batch; only the
direction of each paired difference is randomised. The iid binomial
reference of `scripts/me6_null_calibration.py` (ADR-0021) is recomputed on
the same K and n at the grand-mean accuracy for comparison.

## Reading the output (`reports/me6_paired_null.json`)

- `median.observed` — the B-3 statistic of the grid.
- `median.null.{mean,p99}` — what the statistic does under H0.
- `median.frac_ge_observed` = P(median_null >= observed);
  `median.p_value` = (count + 1) / (B + 1). Small (< 0.05) = the observed
  median is in the upper tail of the paired null, i.e. not what pure
  sign-symmetry produces. Large = compatible with H0.
- `median.null.frac_above_threshold` = P(B-3 passes | H0) with the fixed
  0.02 rule.
- `cells[*]` — the same quantities for each trio's own maximum.
- `binomial_reference` — the ADR-0021 iid reference at the same K, n, p.

Known limits (ADR-0022): items within a probe are assumed exchangeable
(true for the iid-sampled Gaussian / XOR / Sinusoid worlds, not for the
contiguous Studyforrest windows, which need block flips); the signs of the
10 pair differences of a trio are flipped independently, so correlations
between pairs sharing a cell are not reproduced beyond the observed |d_i|.

Usage:
    uv run python scripts/me6_paired_null.py --root runs/v01_grid \\
        --replicates 20000 --out reports/me6_paired_null.json
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import socket
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "bouba_sens.me6_paired_null.v1"

# Spec §1.2 B-3 threshold, as applied in scripts/aggregate_grid.py.
B3_ME6_THRESHOLD = 0.02
# Same tuple and order as aggregate_grid._MODALITIES (checked by the tests).
MODALITIES: tuple[str, ...] = ("audio", "vision", "tactile", "gravity", "force")
N_MODALITIES = len(MODALITIES)
# Same pattern as aggregate_grid._CELL_RE (checked by the tests).
CELL_RE = re.compile(
    r"^seed(?P<seed>\d+)_(?P<timing>T[12])_(?P<modality>\w+?)_(?P<snr>floor|minus10|plus10)$"
)
SUCCESSES_FILE = "per_query_successes.json"
DEFAULT_REPLICATES = 20_000
DEFAULT_SEED = 25_102_026
# Replicates per RNG chunk. Fixed (not a CLI option) so that a given seed
# always yields the same draws whatever the machine's memory.
_CHUNK = 250

# The 10 unordered off-diagonal pairs (a < b): exactly the non-zero entries
# of the antisymmetric matrix M - M.T used by me6_asymmetry.
_IU, _JU = np.triu_indices(N_MODALITIES, k=1)
PAIR_NAMES: tuple[str, ...] = tuple(
    f"{MODALITIES[a]}|{MODALITIES[b]}" for a, b in zip(_IU, _JU, strict=True)
)


@dataclass(frozen=True)
class Trio:
    """One (seed, timing, snr) cell of the B-3 statistic.

    `successes` has shape `(5, 5, n)`: `successes[q, m, i]` is the outcome
    (1 = correct) on probe item `i` of the model lesioned on modality `m`
    (column) when queried on modality `q` alone (row).
    """

    seed: str
    timing: str
    snr: str
    successes: np.ndarray

    @property
    def label(self) -> str:
        return f"seed{self.seed}_{self.timing}_{self.snr}"

    @property
    def n_items(self) -> int:
        return int(self.successes.shape[-1])


def discover_trios(root: Path) -> tuple[list[Trio], list[dict[str, Any]]]:
    """Group `per_query_successes.json` files by (seed, timing, snr).

    Same directory-name regex and grouping as `aggregate_grid._compute_me6`;
    only the successes file is required (no `eval_report.json`). Returns the
    complete trios (all five lesioned modalities present) and a description
    of the incomplete ones. Inconsistent files (missing query, differing
    item counts, non-binary values) raise `ValueError`.
    """
    buckets: dict[tuple[str, str, str], dict[str, dict[str, list[int]]]] = defaultdict(dict)
    for cell_dir in sorted(root.iterdir()):
        if not cell_dir.is_dir():
            continue
        match = CELL_RE.match(cell_dir.name)
        if not match:
            continue
        path = cell_dir / SUCCESSES_FILE
        if not path.exists():
            continue
        body = json.loads(path.read_text())
        if not isinstance(body, dict):
            raise ValueError(f"{path}: expected an object {{modality: [0/1, ...]}}")
        key = (match["seed"], match["timing"], match["snr"])
        buckets[key][match["modality"]] = body

    trios: list[Trio] = []
    incomplete: list[dict[str, Any]] = []
    for key in sorted(buckets, key=lambda k: (int(k[0]), k[1], k[2])):
        by_lesioned = buckets[key]
        missing = [m for m in MODALITIES if m not in by_lesioned]
        if missing:
            incomplete.append(
                {
                    "cell": f"seed{key[0]}_{key[1]}_{key[2]}",
                    "present": sorted(by_lesioned),
                    "missing_lesioned": missing,
                }
            )
            continue
        n_items = _item_count(by_lesioned, key)
        successes = np.empty((N_MODALITIES, N_MODALITIES, n_items), dtype=np.int8)
        for col_j, lesioned in enumerate(MODALITIES):
            column = by_lesioned[lesioned]
            for row_i, query in enumerate(MODALITIES):
                values = column.get(query)
                if values is None:
                    raise ValueError(f"{key}: lesioned {lesioned} lacks query {query}")
                arr = np.asarray(values)
                if arr.shape != (n_items,) or not np.isin(arr, (0, 1)).all():
                    raise ValueError(
                        f"{key}: lesioned {lesioned} / query {query} must be {n_items} 0/1 values"
                    )
                successes[row_i, col_j, :] = arr
        trios.append(Trio(key[0], key[1], key[2], successes))
    return trios, incomplete


def _item_count(by_lesioned: dict[str, dict[str, list[int]]], key: tuple[str, str, str]) -> int:
    lengths = {len(values) for column in by_lesioned.values() for values in column.values()}
    if len(lengths) != 1:
        raise ValueError(f"{key}: item counts differ across entries: {sorted(lengths)}")
    (n_items,) = lengths
    if n_items == 0:
        raise ValueError(f"{key}: empty successes")
    return n_items


def stack_successes(trios: list[Trio]) -> np.ndarray:
    """`(K, 5, 5, n)` int8 array; all trios must share the same n."""
    if not trios:
        raise ValueError("no complete trio")
    n_items = {t.n_items for t in trios}
    if len(n_items) != 1:
        raise ValueError(f"trios have different item counts: {sorted(n_items)}")
    return np.stack([t.successes for t in trios])


def pair_differences(successes: np.ndarray) -> np.ndarray:
    """`d[..., pair, i] = s[a, b, i] - s[b, a, i]` for the 10 pairs a < b.

    Input `(..., 5, 5, n)`, output `(..., 10, n)` in {-1, 0, 1}. The mean
    over `i` of each row is the signed off-diagonal entry
    `P[a, b] - P[b, a]` of the antisymmetric matrix.
    """
    if successes.shape[-3:-1] != (N_MODALITIES, N_MODALITIES):
        raise ValueError(f"expected (..., 5, 5, n); got {successes.shape}")
    return (successes[..., _IU, _JU, :] - successes[..., _JU, _IU, :]).astype(np.int8)


def me6_from_successes(successes: np.ndarray) -> np.ndarray:
    """Observed Me6 per trio: `max_pairs |mean_i d|`. Input `(..., 5, 5, n)`."""
    signed = pair_differences(successes).mean(axis=-1)  # (..., 10)
    return np.abs(signed).max(axis=-1)


def _rng_for(seed: int, start: int) -> np.random.Generator:
    return np.random.default_rng([seed, start])


def paired_null(
    differences: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sign-flip null of Me6 per trio and of the median over trios.

    `differences` is the `(K, 10, n)` output of `pair_differences`. For each
    replicate, each (trio, pair, item) sign is flipped independently with
    probability 1/2 — the swap `s_mq(i) <-> s_qm(i)` — and Me6 is recomputed
    per trio, then the median over the K trios. Returns
    `(me6_per_trio, median_over_trios)` of shapes `(replicates, K)` and
    `(replicates,)`. Deterministic given `seed`.
    """
    if differences.ndim != 3 or differences.shape[1] != len(PAIR_NAMES):
        raise ValueError(f"expected (K, 10, n); got {differences.shape}")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    n_trios, _, n_items = differences.shape
    per_trio = np.empty((replicates, n_trios))
    start = 0
    while start < replicates:
        g = min(_CHUNK, replicates - start)
        rng = _rng_for(seed, start)
        signs = rng.integers(0, 2, size=(g, *differences.shape), dtype=np.int8) * 2 - 1
        flipped_sum = (signs * differences).sum(axis=-1, dtype=np.int32)  # (g, K, 10)
        per_trio[start : start + g] = np.abs(flipped_sum).max(axis=-1) / n_items
        start += g
    return per_trio, np.median(per_trio, axis=1)


def summarise(values: np.ndarray, *, threshold: float = B3_ME6_THRESHOLD) -> dict[str, float]:
    """mean / median / P95 / P99 and the fraction above the B-3 threshold.

    Quantiles use `inverted_cdf`, as in me6_null_calibration.py, so P99 is
    an attained value of the discrete distribution.
    """
    q95, q99 = np.quantile(values, [0.95, 0.99], method="inverted_cdf")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(q95),
        "p99": float(q99),
        "frac_above_threshold": float((values > threshold).mean()),
    }


def tail(values: np.ndarray, observed: float) -> dict[str, float | int]:
    """Upper-tail count and p-value of `observed` under the empirical `values`."""
    count = int((values >= observed).sum())
    total = int(values.size)
    return {
        "count_ge_observed": count,
        "frac_ge_observed": count / total,
        "p_value": (count + 1) / (total + 1),
    }


def binomial_reference(
    *, n_items: int, p: float, replicates: int, cells: int, seed: int
) -> dict[str, Any]:
    """ADR-0021 iid binomial reference at the same K, n and grand-mean p."""
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from me6_null_calibration import simulate_null

    p_clipped = min(max(p, 1e-6), 1 - 1e-6)
    per_trio, medians = simulate_null(
        n_items, p_clipped, replicates=replicates, cells=cells, seed=seed
    )
    return {
        "source": "scripts/me6_null_calibration.py simulate_null (ADR-0021)",
        "n": n_items,
        "p": p_clipped,
        "replicates": replicates,
        "cells": cells,
        "me6_per_trio": summarise(per_trio),
        "me6_median_over_cells": summarise(medians),
        "_medians": medians,
    }


def analyse(
    trios: list[Trio],
    *,
    replicates: int,
    seed: int,
    with_binomial: bool = True,
    incomplete: list[dict[str, Any]] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Observed B-3 statistic, paired null, per-trio nulls, binomial reference."""
    t0 = time.perf_counter()
    successes = stack_successes(trios)  # (K, 5, 5, n)
    n_trios, _, _, n_items = successes.shape
    differences = pair_differences(successes)  # (K, 10, n)
    signed = differences.mean(axis=-1)  # (K, 10) = P[a, b] - P[b, a]
    observed_cells = me6_from_successes(successes)  # (K,)
    observed_median = float(np.median(observed_cells))
    grand_mean_p = float(successes.mean())

    null_cells, null_medians = paired_null(differences, replicates=replicates, seed=seed)

    cells_out: list[dict[str, Any]] = []
    for k, trio in enumerate(trios):
        argmax = int(np.abs(signed[k]).argmax())
        cells_out.append(
            {
                "cell": trio.label,
                "seed": int(trio.seed),
                "timing": trio.timing,
                "snr": trio.snr,
                "n_items": n_items,
                "accuracy_matrix_mean": float(successes[k].mean()),
                "observed_me6": float(observed_cells[k]),
                "observed_argmax_pair": PAIR_NAMES[argmax],
                "observed_signed_pairs": dict(
                    zip(PAIR_NAMES, (float(v) for v in signed[k]), strict=True)
                ),
                "discordant_items_per_pair": dict(
                    zip(
                        PAIR_NAMES,
                        (int(v) for v in np.abs(differences[k]).sum(axis=-1)),
                        strict=True,
                    )
                ),
                "null": summarise(null_cells[:, k]),
                **tail(null_cells[:, k], float(observed_cells[k])),
            }
        )

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "params": {
            "root": str(root) if root is not None else None,
            "replicates": replicates,
            "seed": seed,
            "threshold": B3_ME6_THRESHOLD,
            "n_modalities": N_MODALITIES,
            "n_items": n_items,
            "cells_counted": n_trios,
            "cells_expected_full_grid": 30,
            "complete": n_trios == 30 and not incomplete,
            "null": "paired sign-flip of s_mq(i) <-> s_qm(i), independent per (trio, pair, item)",
        },
        "incomplete_trios": incomplete or [],
        "median": {
            "observed": observed_median,
            "passes_fixed_threshold": observed_median > B3_ME6_THRESHOLD,
            "null": summarise(null_medians),
            **tail(null_medians, observed_median),
        },
        "cells": cells_out,
        "per_trio_pooled_null": summarise(null_cells.reshape(-1)),
        "grand_mean_accuracy": grand_mean_p,
    }
    if with_binomial:
        ref = binomial_reference(
            n_items=n_items, p=grand_mean_p, replicates=replicates, cells=n_trios, seed=seed
        )
        ref_medians = ref.pop("_medians")
        ref["median_tail"] = tail(ref_medians, observed_median)
        result["binomial_reference"] = ref
    result["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return result


def _print_report(result: dict[str, Any]) -> None:
    params, med = result["params"], result["median"]
    print(
        f"Me6 paired null — K={params['cells_counted']} trios"
        f"{'' if params['complete'] else ' (INCOMPLETE grid)'}, n={params['n_items']} items, "
        f"B={params['replicates']}, seed={params['seed']}"
    )
    if result["incomplete_trios"]:
        print(f"  incomplete trios skipped: {len(result['incomplete_trios'])}")
    print()
    print(
        f"median over trios: observed {med['observed']:.4f} | null mean {med['null']['mean']:.4f} "
        f"median {med['null']['median']:.4f} P95 {med['null']['p95']:.4f} P99 {med['null']['p99']:.4f} "
        f"| P(null >= obs) {med['frac_ge_observed']:.4f} (p = {med['p_value']:.4f}) "
        f"| P(B-3 passes | H0) {med['null']['frac_above_threshold']:.4f}"
    )
    ref = result.get("binomial_reference")
    if ref:
        rm = ref["me6_median_over_cells"]
        print(
            f"binomial iid reference (p={ref['p']:.4f}): null mean {rm['mean']:.4f} "
            f"P99 {rm['p99']:.4f} | P(null >= obs) {ref['median_tail']['frac_ge_observed']:.4f}"
        )
    print()
    print(f"{'cell':>22} {'obs':>7} {'null mean':>9} {'P99':>7} {'P(>=)':>7} {'argmax pair':>18}")
    print("-" * 76)
    for cell in result["cells"]:
        print(
            f"{cell['cell']:>22} {cell['observed_me6']:>7.4f} {cell['null']['mean']:>9.4f} "
            f"{cell['null']['p99']:>7.4f} {cell['frac_ge_observed']:>7.4f} "
            f"{cell['observed_argmax_pair']:>18}"
        )
    print(f"elapsed {result['elapsed_s']} s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="grid directory (output of scripts/run_grid.sh)"
    )
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--no-binomial", action="store_true", help="skip the ADR-0021 iid binomial reference"
    )
    parser.add_argument("--out", type=Path, default=Path("reports/me6_paired_null.json"))
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.root.is_dir():
        print(f"error: --root {args.root} is not a directory", file=sys.stderr)
        return 2
    trios, incomplete = discover_trios(args.root)
    if not trios:
        print(
            f"error: no complete (seed, timing, snr) trio with {SUCCESSES_FILE} under {args.root}"
            f" ({len(incomplete)} incomplete)",
            file=sys.stderr,
        )
        return 2
    result = analyse(
        trios,
        replicates=args.replicates,
        seed=args.seed,
        with_binomial=not args.no_binomial,
        incomplete=incomplete,
        root=args.root,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if not args.quiet:
        _print_report(result)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
