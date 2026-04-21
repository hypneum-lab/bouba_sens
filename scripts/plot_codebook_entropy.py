"""Sprint 14c — codebook entropy trajectory plot for CBFREEZE mechanism.

Reads per-cell `report.pkl` artifacts (`AdaptationReport` dataclass) for
two or more grid roots, stacks the `codebook_entropy_trajectory` lists
across (seed, timing, modality, snr) cells, and plots the cross-cell
median + IQR per step for each root.

Goal: visualise that the LOCK=100 (Amedi-peak) codebook has rising
entropy during the lesion / re-distribution phase (P2), whereas
LOCK=100 + CBFREEZE stays flat — the mechanism behind the Sprint 13b
finding that freezing the codebook changes the B-1 verdict.

Usage:
    uv run python scripts/plot_codebook_entropy.py \
        --roots runs/v05_s13_cbfreeze,runs/v0.5_dr_lock100 \
        --out reports/v0.5_s14c_codebook_entropy.png
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_CELL_RE = re.compile(
    r"^seed(?P<seed>\d+)_(?P<timing>T[12])_(?P<modality>\w+?)_"
    r"(?P<snr>floor|minus10|plus10)$"
)


def _load_root_trajectories(root: Path) -> np.ndarray:
    """Return array of shape (n_cells, n_steps) from every cell under `root`.

    Cells that lack `report.pkl` or whose trajectory length differs from
    the modal length are skipped. This avoids a single aborted cell
    collapsing the array shape.
    """
    curves: list[list[float]] = []
    for cell_dir in sorted(root.iterdir()):
        if not cell_dir.is_dir() or not _CELL_RE.match(cell_dir.name):
            continue
        report_path = cell_dir / "report.pkl"
        if not report_path.exists():
            continue
        with report_path.open("rb") as fh:
            report = pickle.load(fh)
        traj = getattr(report, "codebook_entropy_trajectory", None)
        if not traj:
            continue
        curves.append([float(v) for v in traj])

    if not curves:
        return np.empty((0, 0), dtype=np.float64)

    lengths = [len(c) for c in curves]
    # Keep only the modal-length curves so stacking is well-defined.
    modal_len = max(set(lengths), key=lengths.count)
    stacked = np.array([c for c in curves if len(c) == modal_len], dtype=np.float64)
    return stacked


def _root_label(root: Path) -> str:
    """Short human-readable tag for the legend."""
    return root.name


def plot(roots: list[Path], out: Path) -> dict[str, dict[str, float]]:
    """Write a line-plot PNG; return per-root (final, peak, delta) stats."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    stats: dict[str, dict[str, float]] = {}

    for root in roots:
        curves = _load_root_trajectories(root)
        if curves.size == 0:
            print(f"WARNING: no usable report.pkl under {root}")
            continue
        n_cells, n_steps = curves.shape
        steps = np.arange(n_steps)
        median = np.median(curves, axis=0)
        q25 = np.percentile(curves, 25.0, axis=0)
        q75 = np.percentile(curves, 75.0, axis=0)
        label = f"{_root_label(root)}  (n={n_cells})"
        line = ax.plot(steps, median, linewidth=2.0, label=label)[0]
        ax.fill_between(steps, q25, q75, alpha=0.2, color=line.get_color())

        stats[root.name] = {
            "n_cells": float(n_cells),
            "entropy_step0": float(median[0]),
            "entropy_final": float(median[-1]),
            "entropy_delta": float(median[-1] - median[0]),
            "entropy_peak": float(median.max()),
        }

    ax.set_xlabel("P2 snapshot index (lesion phase trajectory)")
    ax.set_ylabel("codebook entropy (nats)")
    ax.set_title(
        "Codebook entropy trajectory during P2 (lesion re-distribution)\n"
        "median across cells, shaded band = IQR"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        type=str,
        required=True,
        help="Comma-separated list of grid run directories.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    roots = [Path(p.strip()) for p in args.roots.split(",") if p.strip()]
    stats = plot(roots, args.out)
    for root_name, row in stats.items():
        print(
            f"{root_name}: n={int(row['n_cells'])} "
            f"H0={row['entropy_step0']:.4f} "
            f"Hf={row['entropy_final']:.4f} "
            f"Hpeak={row['entropy_peak']:.4f} "
            f"dH={row['entropy_delta']:+.4f}"
        )


if __name__ == "__main__":
    main()
