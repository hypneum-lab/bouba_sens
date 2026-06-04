#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, SIM105
"""N12 verdict : tactile-floor + force-plus10 subgroup replication on 10 NEW seeds.

Per pre-registration (per critic v2 fix : N=10 NEW seeds only enter the verdict).
Tests : per-subgroup quadratic regression with c<0 + Bonferroni α=0.025 (2 pre-registered subgroups).
Honest framing : even significant result is hypothesis-generating per family-wise
note (15-subgroup exploration in N9 → effective α=0.05/15=0.0033 for true control).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).parent.parent
N12_SEEDS = [3, 11, 19, 29, 37, 41, 47, 59, 67, 79]
LOCKS = [50, 75, 100, 125, 150]
TARGET_SUBGROUPS = [("tactile", "floor"), ("force", "plus10")]


# Load N12 me7 values per (seed, modality, snr)
def me7_per_grid(agg_data):
    cells = agg_data["cells"]
    out = {}
    for m in ["audio", "vision", "tactile", "gravity", "force"]:
        for snr in ["floor", "minus10", "plus10"]:
            t1 = cells.get(f"t1_{m}_{snr}", {}).get("me1", {}).get("mean")
            t2 = cells.get(f"t2_{m}_{snr}", {}).get("me1", {}).get("mean")
            out[(m, snr)] = t1 - t2 if t1 is not None and t2 is not None else None
    return out


# Build matrix: subgroup -> lock -> list of me7 across 10 NEW seeds
data = {sg: {la: [] for la in LOCKS} for sg in TARGET_SUBGROUPS}
all_n12 = {}
for s in N12_SEEDS:
    for la in LOCKS:
        agg_path = REPO / f"runs/v05_dr_n12_seed{s}_lock{la}/v0.1_aggregate.json"
        if not agg_path.exists():
            print(f"MISSING n12 seed={s} lock={la}")
            continue
        agg = json.loads(agg_path.read_text())
        per_grid = me7_per_grid(agg)
        all_n12[(s, la)] = per_grid
        for sg in TARGET_SUBGROUPS:
            v = per_grid.get(sg)
            if v is not None:
                data[sg][la].append(v)

print(f"Loaded {len(all_n12)}/50 N12 grids")
print()

# === Per-subgroup quadratic regression on NEW seeds only ===
print("=" * 70)
print("N12 verdict: quadratic regression on 10 NEW seeds per subgroup")
print("=" * 70)
print(
    f"{'subgroup':>20} {'L50':>9} {'L75':>9} {'L100':>9} {'L125':>9} {'L150':>9} {'c':>11} {'p':>7} {'verdict':>15}"
)

ALPHA_BONFERRONI = 0.025  # 0.05 / 2 pre-registered subgroups
results = []

for sg in TARGET_SUBGROUPS:
    m, snr = sg
    row_meds = []
    flat_x, flat_y = [], []
    for la in LOCKS:
        vals = data[sg][la]
        row_meds.append(float(np.median(vals)) if vals else None)
        for v in vals:
            flat_x.append(la)
            flat_y.append(v)
    if len(flat_x) < 5:
        print(f"{m + '-' + snr:>20} insufficient data")
        continue
    flat_x = np.array(flat_x, dtype=float)
    flat_y = np.array(flat_y, dtype=float)
    coeffs = np.polyfit(flat_x, flat_y, deg=2)
    c, b, a = coeffs
    # Bootstrap SE on c
    rng = np.random.default_rng(42)
    bs_c = []
    for _ in range(2000):
        idx = rng.choice(len(flat_x), size=len(flat_x), replace=True)
        try:
            bs_c.append(np.polyfit(flat_x[idx], flat_y[idx], deg=2)[0])
        except Exception:
            pass
    se_c = float(np.std(bs_c))
    p_one = float(stats.norm.cdf(c / se_c)) if se_c > 0 else float("nan")
    sig_bonferroni = p_one < ALPHA_BONFERRONI and c < 0
    peak_x = -b / (2 * c) if c < 0 else None
    in_band = peak_x is not None and 85 <= peak_x <= 115
    verdict_sub = (
        "REPLICATES ✓"
        if (sig_bonferroni and in_band)
        else ("c<0 but peak out" if sig_bonferroni else "n.s.")
    )
    row_str = " ".join(f"{med:>+9.4f}" if med is not None else "    n/a" for med in row_meds)
    peak_str = f"@{peak_x:.1f}" if peak_x else "no peak"
    print(f"{m + '-' + snr:>20} {row_str} {c:>+11.2e} {p_one:>7.3f} {verdict_sub:>15} {peak_str}")
    results.append(
        {
            "subgroup": f"{m}-{snr}",
            "c": float(c),
            "se_c": se_c,
            "p_one_tailed": p_one,
            "peak_estimate": float(peak_x) if peak_x else None,
            "bonferroni_alpha": ALPHA_BONFERRONI,
            "sig_bonferroni": bool(sig_bonferroni),
            "peak_in_band_85_115": bool(in_band),
            "verdict_subgroup": verdict_sub,
        }
    )

# Overall verdict
n_replicate = sum(1 for r in results if r["sig_bonferroni"] and r["peak_in_band_85_115"])
n_sig_but_drift = sum(1 for r in results if r["sig_bonferroni"] and not r["peak_in_band_85_115"])
print()
print("=" * 70)
print(f"VERDICT: {n_replicate}/2 subgroups replicate at Bonferroni α={ALPHA_BONFERRONI}")
print("=" * 70)
if n_replicate >= 1:
    overall = "N12-survive"
    print("Per pre-reg: N12-survive — §5.5 reframed as 'modality-specific Amedi signature'")
    print("Family-wise note: even this 'significant' result is hypothesis-generating;")
    print("true family-wise control across 15 N9-exploration subgroups would require α=0.0033.")
elif n_sig_but_drift >= 1 or any(r["p_one_tailed"] < 0.05 for r in results):
    overall = "N12-tie"
    print("Per pre-reg: N12-tie — §5.5 reframed as 'preliminary, requires N≥30 dedicated'")
else:
    overall = "N12-loses"
    print("Per pre-reg: N12-loses — §5.5 retract confirmed at subgroup level. ADR-0019 finalized.")
    print("TMLR submission proceeds without §5.5 headline ; pivot to other findings.")

# Save
out = REPO / "reports" / "v0.5_amedi_n12_verdict.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(
        {
            "n12_seeds": N12_SEEDS,
            "target_subgroups": [f"{m}-{snr}" for m, snr in TARGET_SUBGROUPS],
            "alpha_bonferroni": ALPHA_BONFERRONI,
            "results_per_subgroup": results,
            "n_replicate": n_replicate,
            "verdict_overall": overall,
        },
        indent=2,
    )
)
print()
print(f"Wrote {out}")
