#!/usr/bin/env python3
"""Q3 / Q3+ Amedi inverted-U verdict via per-seed quadratic regression.

Per N8/N9 critic MAJOR #4 : Jonckheere-Terpstra tests *monotonic*
trend, but the §5.5 hypothesis is *inverted-U* (non-monotone peak).
The proper test is fitting `y = a + b*x + c*x^2` per seed and
testing `c < 0` (concave-down).

Loads `runs/v05_dr_seed{S}_lock{L}/v0.1_aggregate.json` for each
combination of seeds x LOCK_AFTER values, extracts
`invariants.b1.median_me7`, then :

1. Reports per-seed peak position (argmax over LOCK).
2. Fits `y = a + b*x + c*x^2` per seed, reports c sign + extremum.
3. Sign test on (c < 0) across seeds (binomial vs 0.5).
4. One-sample t-test on mean(c) < 0.
5. Verdict per pre-reg brackets : Survives / Reframe / Retract.

Writes `reports/v0.5_amedi_curve_multiseed_<N>seed_final.json`.

Usage:
    uv run python scripts/analyse_amedi_quadratic.py \
        --seeds 0,17,42,73,101,7,23,31,53,89 \
        --locks 50,75,100,125,150 \
        --out reports/v0.5_amedi_curve_multiseed_10seed_final.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
from scipy import stats


def load_matrix(repo: Path, seeds: list[int], locks: list[int]) -> dict[int, dict[int, float]]:
    matrix: dict[int, dict[int, float]] = {}
    for s in seeds:
        matrix[s] = {}
        for la in locks:
            agg_path = repo / f"runs/v05_dr_seed{s}_lock{la}/v0.1_aggregate.json"
            agg = json.loads(agg_path.read_text())
            matrix[s][la] = agg["invariants"]["b1"]["median_me7"]
    return matrix


def quadratic_per_seed(
    locks: list[int], matrix: dict[int, dict[int, float]], seeds: list[int]
) -> tuple[list[float], list[float | None]]:
    """Per-seed quadratic fit. Returns (c_coeffs, peak_estimates)."""
    locks_arr = np.array(locks, dtype=float)
    c_coeffs: list[float] = []
    peak_estimates: list[float | None] = []
    for s in seeds:
        y = np.array([matrix[s][la] for la in locks], dtype=float)
        c, b, _a = np.polyfit(locks_arr, y, deg=2)
        c_coeffs.append(float(c))
        peak_estimates.append(float(-b / (2 * c)) if c < 0 else None)
    return c_coeffs, peak_estimates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("."),
        help="Path to bouba_sens repo (default: cwd).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,17,42,73,101,7,23,31,53,89",
        help="Comma-separated seed list.",
    )
    parser.add_argument(
        "--seeds-base",
        type=str,
        default="0,17,42,73,101",
        help="Original 5-seed cohort (for matrix annotation).",
    )
    parser.add_argument(
        "--locks",
        type=str,
        default="50,75,100,125,150",
        help="Comma-separated LOCK_AFTER values.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/v0.5_amedi_curve_multiseed_10seed_final.json"),
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    seeds_base = [int(s) for s in args.seeds_base.split(",") if s.strip()]
    seeds_plus = [s for s in seeds if s not in seeds_base]
    locks = [int(la) for la in args.locks.split(",") if la.strip()]

    matrix = load_matrix(args.repo, seeds, locks)

    print(f"=== median_me7 matrix {len(seeds)} seeds x {len(locks)} LOCK_AFTER ===")
    print(f"{'seed':>6}", " ".join(f"{la:>9}" for la in locks))
    for s in seeds:
        marker = "*" if s in seeds_plus else " "
        print(f"{marker}{s:>5}", " ".join(f"{matrix[s][la]:>+9.4f}" for la in locks))

    peak_positions: list[int] = []
    peak_heights: list[float] = []
    for s in seeds:
        vals = [matrix[s][la] for la in locks]
        peak_idx = int(np.argmax(vals))
        peak_positions.append(locks[peak_idx])
        peak_heights.append(vals[peak_idx])

    median_peak = st.median(peak_positions)
    drift_pct = (max(peak_positions) - min(peak_positions)) / 100 * 100

    print()
    print(f"Peak positions (argmax): {peak_positions}")
    print(f"Median peak: {median_peak} | drift: {drift_pct:.1f}%")

    c_coeffs, peak_estimates = quadratic_per_seed(locks, matrix, seeds)

    print()
    print("=== Per-seed quadratic fit (test c<0 = inverted-U) ===")
    for s, c, pe in zip(seeds, c_coeffs, peak_estimates, strict=True):
        marker = "*" if s in seeds_plus else " "
        if pe is not None:
            print(f"{marker}seed={s:>4}: c={c:+.6f} peak@{pe:>6.1f}")
        else:
            print(f"{marker}seed={s:>4}: c={c:+.6f} NO PEAK (c>=0)")

    n_concave = sum(1 for c in c_coeffs if c < 0)
    n_total = len(c_coeffs)
    sign_p = float(stats.binomtest(n_concave, n_total, p=0.5, alternative="greater").pvalue)
    t_c, p_two = stats.ttest_1samp(c_coeffs, 0.0)
    p_one = float(p_two / 2 if t_c < 0 else 1 - p_two / 2)

    print()
    print(f"Sign test (c<0 in m/n seeds): {n_concave}/{n_total} (binomial p={sign_p:.4f})")
    print(f"t-test mean(c)<0: t={t_c:.3f} one-tailed p={p_one:.4f}")

    valid_peaks = [p for p in peak_estimates if p is not None]
    if valid_peaks:
        print()
        print(f"Peak position estimates (c<0 seeds, n={len(valid_peaks)}):")
        print(f"  median: {np.median(valid_peaks):.1f}")
        print(
            f"  IQR: [{np.percentile(valid_peaks, 25):.1f}, {np.percentile(valid_peaks, 75):.1f}]"
        )

    non_monotone_significant = (sign_p < 0.05) or (p_one < 0.05)
    peak_in_band = 75 <= median_peak <= 125

    if non_monotone_significant and peak_in_band and drift_pct <= 25:
        verdict = "Survives"
        narrative = "Survives (peak stable @ 100 +/- 25%, non-monotone confirmed)"
    elif non_monotone_significant:
        verdict = "Reframe"
        narrative = f"Reframe (peak drift {drift_pct:.1f}%, non-monotone confirmed via quadratic)"
    else:
        verdict = "Retract"
        narrative = (
            f"Retract (non-monotone NOT confirmed: sign p={sign_p:.4f}, mean(c)<0 p={p_one:.4f})"
        )
    print()
    print(f"VERDICT: {narrative}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "matrix": {str(s): {str(la): matrix[s][la] for la in locks} for s in seeds},
                "all_seeds": seeds,
                "seeds_base": seeds_base,
                "seeds_plus": seeds_plus,
                "peak_positions": peak_positions,
                "peak_heights": peak_heights,
                "median_peak_argmax": median_peak,
                "drift_pct": drift_pct,
                "quadratic_per_seed": [
                    {"seed": s, "c": float(c), "peak_estimate": float(p) if p is not None else None}
                    for s, c, p in zip(seeds, c_coeffs, peak_estimates, strict=True)
                ],
                "n_concave_down": n_concave,
                "sign_test_p": sign_p,
                "t_test_mean_c": {"t": float(t_c), "p_one_tailed": p_one},
                "valid_peaks_median": float(np.median(valid_peaks)) if valid_peaks else None,
                "valid_peaks_iqr": (
                    [float(np.percentile(valid_peaks, 25)), float(np.percentile(valid_peaks, 75))]
                    if valid_peaks
                    else None
                ),
                "verdict": verdict,
                "narrative": narrative,
                "test": "quadratic_regression_per_critic",
            },
            indent=2,
        )
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
