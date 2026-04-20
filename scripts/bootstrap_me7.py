"""Task 7.2 — bootstrap 95 % CI on Me7 median per world.

Usage (standalone):
    uv run python scripts/bootstrap_me7.py \\
        reports/v0.2_aggregate_gaussian.json \\
        reports/v0.2_aggregate_xor.json \\
        reports/v0.2_aggregate_sinusoid.json

The script loads each aggregate JSON, extracts the 75 raw Me7
paired-values (congenital - late_acquired), bootstraps ``n_boot``
resamples, and prints the 95 % CI per world.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class Me7CI(TypedDict):
    """Bootstrap 95 % CI for a single world's Me7 median."""

    median: float
    ci_low: float
    ci_high: float


def bootstrap_me7_median_ci(
    values: NDArray[np.float64],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Me7CI:
    """Return bootstrap 95 % CI for the median of *values*.

    Parameters
    ----------
    values:
        1-D array of raw Me7 paired-values (congenital - late_acquired).
    n_boot:
        Number of bootstrap resamples.
    seed:
        RNG seed for reproducibility.
    alpha:
        Two-tailed significance level; default 0.05 yields 95 % CI.

    Returns
    -------
    Me7CI
        Dict with keys ``median``, ``ci_low``, ``ci_high`` (all ``float``).
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = arr.size
    boot_medians = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        resample = rng.choice(arr, size=n, replace=True)
        boot_medians[i] = float(np.median(resample))
    lo = float(np.percentile(boot_medians, 100 * alpha / 2))
    hi = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))
    return Me7CI(median=float(np.median(arr)), ci_low=lo, ci_high=hi)


def _extract_me7_values(aggregate: dict[str, object]) -> NDArray[np.float64]:
    """Extract raw Me7 paired-values from an aggregate JSON dict."""
    cells: list[dict[str, object]] = aggregate.get("cells", [])  # type: ignore[assignment]
    vals: list[float] = []
    for cell in cells:
        me7 = cell.get("me7_congenital_minus_late")
        if me7 is not None:
            vals.append(float(me7))  # type: ignore[arg-type]
    return np.array(vals, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap 95 % CI on Me7 median per world.")
    parser.add_argument("reports", nargs="+", type=Path, help="Aggregate JSON files.")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for path in args.reports:
        aggregate = json.loads(path.read_text())
        values = _extract_me7_values(aggregate)
        if values.size == 0:
            print(f"{path.name}: no me7 values found")
            continue
        ci = bootstrap_me7_median_ci(values, n_boot=args.n_boot, seed=args.seed)
        print(
            f"{path.stem}: median={ci['median']:.4f} "
            f"95%CI=[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
        )


if __name__ == "__main__":
    main()
