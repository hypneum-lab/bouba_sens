"""Confirmatory TOST analysis for amendment v0.7 (sections 3, 6 and 9).

Input JSON (one value per confirmatory seed; collected from the grid
aggregates, ``invariants.b1.median_me7`` per (seed, LOCK_AFTER), and from
the E1 runner, ``ross_full_delta_bits`` per seed)::

    {"e1": [{"seed": 1001, "locks": {"50": v, "75": v, "100": v,
             "125": v, "150": v}, "nolock": v}, ...],
     "e2": [{"seed": 1001, "ross_full_delta_bits": v}, ...]}

Decision (frozen): TOST with two one-sided one-sample t-tests at
alpha = 0.05, i.e. the 90 % two-sided CI inside +-bound; bounds 0.025
(B-1) and 0.05 bit (B-2). Each primary test gets exactly one of the six
section-9 labels from its 90 % CI. B-1 fixed sequence E-1 -> E-1b (IUT
over the five locks) -> E-1c (no-lock). Reported, no decision: 95 % CI,
a 10 000-resample percentile bootstrap TOST and a Wilcoxon signed-rank
TOST.

    uv run python scripts/analyse_v07_tost.py --input v07_values.json --out v07_tost.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

ALPHA = 0.05
B1_BOUND = 0.025
B2_BOUND = 0.05
LOCKS: tuple[str, ...] = ("50", "75", "100", "125", "150")
N_BOOT = 10_000
BOOT_SEED = 20261023


def classify(lo: float, hi: float, bound: float) -> str:
    """Section-9 label of a 90 % CI [lo, hi] against {-bound, 0, +bound}."""
    if -bound < lo and hi < bound:
        return "equivalent"
    if lo >= bound:
        return "beyond_bound_registered"
    if hi <= -bound:
        return "beyond_bound_reverse"
    if 0 < lo < bound <= hi:
        return "effect_detected_size_unresolved"
    if lo <= -bound < hi < 0:
        return "reverse_effect_detected_size_unresolved"
    return "inconclusive"


def _t_ci(x: np.ndarray, level: float) -> tuple[float, float]:
    n = x.size
    se = x.std(ddof=1) / math.sqrt(n)
    q = stats.t.ppf(0.5 + level / 2, n - 1)
    m = float(x.mean())
    return m - q * se, m + q * se


def tost(values: Sequence[float], bound: float, *, n_boot: int = N_BOOT) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 3 or not np.all(np.isfinite(x)):
        raise ValueError("need >= 3 finite per-seed values")
    n = x.size
    se = x.std(ddof=1) / math.sqrt(n)
    p_lower = float(stats.t.sf((x.mean() + bound) / se, n - 1))
    p_upper = float(stats.t.cdf((x.mean() - bound) / se, n - 1))
    lo, hi = _t_ci(x, 1 - 2 * ALPHA)
    rng = np.random.default_rng(BOOT_SEED)
    boot = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    b_lo, b_hi = (float(v) for v in np.percentile(boot, [100 * ALPHA, 100 * (1 - ALPHA)]))
    w_lower = float(stats.wilcoxon(x + bound, alternative="greater").pvalue)
    w_upper = float(stats.wilcoxon(x - bound, alternative="less").pvalue)
    return {
        "n": int(n),
        "bound": bound,
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "ci90": [lo, hi],
        "ci95": list(_t_ci(x, 0.95)),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "label": classify(lo, hi, bound),
        "bootstrap": {"ci90": [b_lo, b_hi], "label": classify(b_lo, b_hi, bound)},
        "wilcoxon": {
            "p_lower": w_lower,
            "p_upper": w_upper,
            "equivalent": bool(max(w_lower, w_upper) < ALPHA),
        },
    }


def _unique_seeds(rows: Sequence[dict[str, Any]]) -> None:
    seeds = [r["seed"] for r in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds in input")


def analyse_e1(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """E-1 (mean over the five locks), then E-1b (IUT), then E-1c (no-lock)."""
    _unique_seeds(rows)
    for r in rows:
        if set(r["locks"]) != set(LOCKS):
            raise ValueError(f"seed {r['seed']}: locks must be exactly {LOCKS}")
    per_lock = {k: [float(r["locks"][k]) for r in rows] for k in LOCKS}
    e1_vals = [float(np.mean([r["locks"][k] for k in LOCKS])) for r in rows]
    out: dict[str, Any] = {"E-1": tost(e1_vals, B1_BOUND)}
    if out["E-1"]["label"] != "equivalent":
        out["E-1b"] = {"tested": False}
        out["E-1c"] = {"tested": False}
        return out
    by_lock = {k: tost(v, B1_BOUND) for k, v in per_lock.items()}
    iut = all(r["label"] == "equivalent" for r in by_lock.values())
    out["E-1b"] = {"tested": True, "equivalent": iut, "per_lock": by_lock}
    if not iut:
        out["E-1c"] = {"tested": False}
        return out
    out["E-1c"] = {"tested": True, **tost([float(r["nolock"]) for r in rows], B1_BOUND)}
    return out


def analyse_e2(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    _unique_seeds(rows)
    return tost([float(r["ross_full_delta_bits"]) for r in rows], B2_BOUND)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.input.read_text())
    result = {}
    if "e1" in data:
        result["B-1"] = analyse_e1(data["e1"])
    if "e2" in data:
        result["B-2"] = {"E-2": analyse_e2(data["e2"])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
