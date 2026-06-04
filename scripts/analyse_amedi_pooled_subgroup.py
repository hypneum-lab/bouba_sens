#!/usr/bin/env python3
"""Q3 pooled v2 : me7 per (seed, modality, snr) at each LOCK_AFTER.

me7 = me1_t1 - me1_t2 paired by (modality, snr).
Per grid: 5 mod x 3 snr = 15 me7 values.
Cross 10 seeds: 150 me7 values per LOCK_AFTER.
Total: 750 me7 measurements across 5 LOCK_AFTER.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).parent.parent
ALL_SEEDS = [0, 17, 42, 73, 101, 7, 23, 31, 53, 89]
LOCKS = [50, 75, 100, 125, 150]
MODS = ["audio", "vision", "tactile", "gravity", "force"]
SNRS = ["floor", "minus10", "plus10"]


def me7_per_grid(agg_data):
    """Returns dict[(modality, snr)] -> me7 value (or None) for one grid."""
    cells = agg_data["cells"]
    out = {}
    for m in MODS:
        for snr in SNRS:
            t1_key = f"t1_{m}_{snr}"
            t2_key = f"t2_{m}_{snr}"
            t1 = cells.get(t1_key, {}).get("me1", {}).get("mean")
            t2 = cells.get(t2_key, {}).get("me1", {}).get("mean")
            if t1 is not None and t2 is not None:
                out[(m, snr)] = t1 - t2
            else:
                out[(m, snr)] = None
    return out


# Load all 50 grids
all_data = {}
for s in ALL_SEEDS:
    for la in LOCKS:
        agg_path = REPO / f"runs/v05_dr_seed{s}_lock{la}/v0.1_aggregate.json"
        if agg_path.exists():
            agg = json.loads(agg_path.read_text())
            all_data[(s, la)] = me7_per_grid(agg)

n_loaded = sum(len([v for v in d.values() if v is not None]) for d in all_data.values())
print(f"Loaded {len(all_data)}/{len(ALL_SEEDS) * len(LOCKS)} grids, {n_loaded} me7 values total")
print()

# === LEVEL 1: pooled per LOCK_AFTER ===
print("=" * 70)
print("LEVEL 1: Pooled across 10 seeds x 5 mod x 3 snr = 150 me7 per LOCK")
print("=" * 70)
print(
    f"{'LOCK':>6} {'N':>5} {'median':>10} {'mean':>10} {'std':>10} {'CI95_low':>10} {'CI95_hi':>10}"
)

pooled = {la: [] for la in LOCKS}
for (_s, la), d in all_data.items():
    for v in d.values():
        if v is not None:
            pooled[la].append(v)

medians_per_lock = {}
for la in LOCKS:
    vals = pooled[la]
    arr = np.array(vals)
    # Bootstrap CI on median
    bs_meds = []
    for _ in range(2000):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        bs_meds.append(np.median(sample))
    med = float(np.median(arr))
    ci_lo = float(np.percentile(bs_meds, 2.5))
    ci_hi = float(np.percentile(bs_meds, 97.5))
    medians_per_lock[la] = med
    print(
        f"{la:>6} {len(vals):>5} {med:>+10.4f} {arr.mean():>+10.4f} {arr.std(ddof=1):>10.4f} {ci_lo:>+10.4f} {ci_hi:>+10.4f}"
    )

# Quadratic on pooled
flat_x, flat_y = [], []
for la in LOCKS:
    for v in pooled[la]:
        flat_x.append(la)
        flat_y.append(v)
flat_x = np.array(flat_x, dtype=float)
flat_y = np.array(flat_y, dtype=float)
coeffs = np.polyfit(flat_x, flat_y, deg=2)
c, b, a = coeffs

# Bootstrap c
bs_c = []
for _ in range(2000):
    idx = np.random.choice(len(flat_x), size=len(flat_x), replace=True)
    bs_c.append(np.polyfit(flat_x[idx], flat_y[idx], deg=2)[0])
se_c = float(np.std(bs_c))
ci_lo_c = float(np.percentile(bs_c, 2.5))
ci_hi_c = float(np.percentile(bs_c, 97.5))

print()
print(f"Pooled quadratic fit: y = {a:+.4f} {b:+.4f}*x {c:+.6f}*x^2")
print(f"  c (concavity): {c:+.5e}, SE={se_c:.5e}, 95% CI=[{ci_lo_c:+.5e}, {ci_hi_c:+.5e}]")
peak_x = -b / (2 * c) if c < 0 else None
print(f"  Peak estimate: LOCK={peak_x:.1f}" if peak_x else "  No peak (c >= 0)")

# Test: is c < 0 with significance?
p_one = float(stats.norm.cdf(c / se_c)) if se_c > 0 else float("nan")
print(f"  p(c<0) one-tailed: {p_one:.4f}")
verdict_pooled = (
    "non-monotone CONFIRMED" if p_one < 0.05 and c < 0 else "non-monotone NOT confirmed"
)
print(f"  POOLED VERDICT: {verdict_pooled}")

# === LEVEL 2: per (modality, snr) subgroup ===
print()
print("=" * 70)
print("LEVEL 2: per (modality, snr) subgroup (15 sub-pops, N=10 seeds each per LOCK)")
print("=" * 70)
print(
    f"{'mod':>10} {'snr':>8} {'L50':>8} {'L75':>8} {'L100':>8} {'L125':>8} {'L150':>8} {'c':>11} {'p':>7} {'verdict':>15}"
)

interesting = []
for m in MODS:
    for snr in SNRS:
        row_meds = []
        sub_x, sub_y = [], []
        for la in LOCKS:
            vals = []
            for s in ALL_SEEDS:
                v = all_data.get((s, la), {}).get((m, snr))
                if v is not None:
                    vals.append(v)
                    sub_x.append(la)
                    sub_y.append(v)
            row_meds.append(float(np.median(vals)) if vals else None)
        if len(sub_x) >= 5:
            try:
                sub_coeffs = np.polyfit(
                    np.array(sub_x, dtype=float), np.array(sub_y, dtype=float), deg=2
                )
                sub_c, sub_b, sub_a = sub_coeffs
                # Bootstrap
                bs_sub = []
                for _ in range(500):
                    idx = np.random.choice(len(sub_x), size=len(sub_x), replace=True)
                    with contextlib.suppress(Exception):
                        bs_sub.append(
                            np.polyfit(
                                np.array([sub_x[i] for i in idx], dtype=float),
                                np.array([sub_y[i] for i in idx], dtype=float),
                                deg=2,
                            )[0]
                        )
                if bs_sub:
                    sub_se = float(np.std(bs_sub))
                    sub_p = float(stats.norm.cdf(sub_c / sub_se)) if sub_se > 0 else float("nan")
                    sub_peak = -sub_b / (2 * sub_c) if sub_c < 0 else None
                    is_sig = sub_p < 0.05 and sub_c < 0
                    if is_sig:
                        interesting.append(
                            (
                                m,
                                snr,
                                float(sub_c),
                                float(sub_p),
                                float(sub_peak) if sub_peak else None,
                            )
                        )
                    row_str = " ".join(
                        f"{med:>+8.4f}" if med is not None else "    n/a" for med in row_meds
                    )
                    print(
                        f"{m:>10} {snr:>8} {row_str} {sub_c:>+11.2e} {sub_p:>7.3f} {'PEAK ✓' if is_sig else 'n.s.':>15}"
                    )
            except Exception as exc:
                print(f"{m:>10} {snr:>8} fit error: {exc}")

print()
print("=" * 70)
print(f"Subgroups with significant non-monotone (p<0.05, c<0): {len(interesting)}/15")
print("=" * 70)
for m, snr, c_val, p_val, peak in interesting:
    peak_str = f"@{peak:.1f}" if peak else "no peak"
    print(f"  {m:>10} {snr:>8}: c={c_val:+.3e} p={p_val:.4f} peak{peak_str}")

# Save
out = REPO / "reports" / "v0.5_amedi_curve_pooled_v2.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(
        {
            "pooled_per_lock_n": {str(la): len(pooled[la]) for la in LOCKS},
            "pooled_medians": {str(la): medians_per_lock[la] for la in LOCKS},
            "pooled_quadratic": {
                "c": float(c),
                "b": float(b),
                "a": float(a),
                "se_c": se_c,
                "p_one": p_one,
                "ci_c": [ci_lo_c, ci_hi_c],
                "peak_estimate": float(peak_x) if peak_x else None,
                "verdict": verdict_pooled,
            },
            "interesting_subgroups": [
                {"modality": m, "snr": snr, "c": cv, "p": pv, "peak": pk}
                for m, snr, cv, pv, pk in interesting
            ],
            "n_subgroups_significant": len(interesting),
            "n_subgroups_total": 15,
        },
        indent=2,
    )
)
print()
print(f"Wrote {out}")
