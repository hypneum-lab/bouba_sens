"""Sprint 10 / paper v0.2 figure: me7 dose-response vs LOCK_AFTER.

Reads the 5 dose-response aggregates plus the 4.5-modal no-lock
baseline, produces a publication-quality scatter+line plot of
median Me7 as a function of LOCK_AFTER, with a horizontal line
at the pre-registered 0.05 threshold.

Output: reports/v0.5_amedi_curve.png (matplotlib)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--out", type=Path, default=Path("reports/v0.5_amedi_curve.png"))
    args = parser.parse_args()

    lock_values = [50, 100, 200, 400, 800]
    me7s: list[float] = []
    for la in lock_values:
        data = json.loads((args.reports_dir / f"v0.5_dr_lock{la}_aggregate.json").read_text())
        me7s.append(data["invariants"]["b1"]["median_me7"])

    baseline = json.loads((args.reports_dir / "v0.5_sf45_nolock_aggregate.json").read_text())
    me7_nolock = baseline["invariants"]["b1"]["median_me7"]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.array(lock_values, dtype=float)
    y = np.array(me7s, dtype=float)
    ax.plot(x, y, marker="o", linewidth=2.0, markersize=9, label="4.5-modal real")
    ax.axhline(y=me7_nolock, linestyle="--", linewidth=1.0, alpha=0.6, label="no-lock baseline")
    ax.axhline(
        y=0.05,
        linestyle=":",
        linewidth=1.5,
        color="crimson",
        label="B-1 pre-registered threshold (0.05)",
    )
    ax.axhline(y=0.0, color="grey", linewidth=0.5, alpha=0.5)

    ax.set_xscale("log")
    ax.set_xticks(lock_values)
    ax.set_xticklabels([str(v) for v in lock_values])
    ax.set_xlabel("LOCK_AFTER (mux plasticity_step at which constellation freezes)")
    ax.set_ylabel(r"median Me7 = $me1(T1) - me1(T2)$  (bits)")
    ax.set_title(
        "Amedi dose-response curve on 4.5-modal Studyforrest bridge\n"
        "(peak at LOCK_AFTER=100 = 50% of STEPS_TRAIN=200)"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotate the peak.
    peak_idx = int(np.argmax(y))
    ax.annotate(
        f"peak = +{y[peak_idx]:.4f}",
        xy=(x[peak_idx], y[peak_idx]),
        xytext=(x[peak_idx] * 1.3, y[peak_idx] * 0.85),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
        fontsize=10,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
