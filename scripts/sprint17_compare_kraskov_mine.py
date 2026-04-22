"""Sprint 17 — Kraskov vs MINE side-by-side Me3 delta estimator.

Pools `pre/post_lesion_codes` and `probe_labels` across all cells of
each seed in a v0.5 grid, then computes ``me3_delta`` (Kraskov kNN)
and ``me3_delta_mine`` (MINE DV) on the same pooled arrays. Emits a
5-row CSV (one per seed) with both estimates + absolute difference.

Sprint 14a retracted the B-2 "migration peak" claim because Kraskov
returns 0.000 at every seed; paper §6.3 now asks whether that null is
load-bearing or an estimator blind spot. This script produces the
cross-estimator evidence.

Usage:
    uv run python scripts/sprint17_compare_kraskov_mine.py \
        --root runs/v05_dr_lock100 \
        --out reports/sprint17_kraskov_vs_mine.csv
"""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import torch

from bouba_sens.loop import AdaptationReport
from bouba_sens.metrics.mi_migration import me3_delta, me3_delta_mine

SEED_DIR_PREFIX = "seed"


def _seed_of(cell_dir: Path) -> str | None:
    """Extract `"0".."4"` from a cell dir name like ``seed3_T1_audio_floor``."""
    name = cell_dir.name
    if not name.startswith(SEED_DIR_PREFIX):
        return None
    rest = name[len(SEED_DIR_PREFIX) :]
    digits: list[str] = []
    for ch in rest:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return "".join(digits) if digits else None


def _load_report(cell_dir: Path) -> AdaptationReport | None:
    report_path = cell_dir / "report.pkl"
    if not report_path.exists():
        return None
    with report_path.open("rb") as fh:
        report: AdaptationReport = pickle.load(fh)
    return report


def _pool_seed(
    cell_dirs: list[Path],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Concatenate pre/post codes + labels across all cells of one seed."""
    pres: list[torch.Tensor] = []
    posts: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for cell_dir in cell_dirs:
        report = _load_report(cell_dir)
        if report is None:
            continue
        if (
            report.pre_lesion_codes is None
            or report.post_lesion_codes is None
            or report.probe_labels is None
        ):
            continue
        pres.append(report.pre_lesion_codes.detach().cpu().flatten())
        posts.append(report.post_lesion_codes.detach().cpu().flatten())
        labels.append(report.probe_labels.detach().cpu().flatten())
    if not pres:
        return None
    return (
        torch.cat(pres, dim=0),
        torch.cat(posts, dim=0),
        torch.cat(labels, dim=0),
    )


def _group_by_seed(root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for cell_dir in sorted(root.iterdir()):
        if not cell_dir.is_dir():
            continue
        seed = _seed_of(cell_dir)
        if seed is None:
            continue
        if cell_dir.name.startswith(f"{SEED_DIR_PREFIX}{seed}_T"):
            groups[seed].append(cell_dir)
    return groups


def compare(root: Path, out: Path) -> list[dict[str, float | int | str]]:
    """Walk grid, compute Kraskov vs MINE Me3 delta per seed, write CSV."""
    groups = _group_by_seed(root)
    if not groups:
        raise RuntimeError(f"no per-cell seed directories found under {root}")

    rows: list[dict[str, float | int | str]] = []
    for seed in sorted(groups):
        pooled = _pool_seed(groups[seed])
        if pooled is None:
            continue
        pre, post, labels = pooled
        n_samples = int(pre.shape[0])

        kraskov = me3_delta(pre, post, labels)
        mine = me3_delta_mine(pre, post, labels)
        row: dict[str, float | int | str] = {
            "seed": seed,
            "n_samples": n_samples,
            "me3_delta_kraskov": kraskov,
            "me3_delta_mine": mine,
            "abs_diff": abs(mine - kraskov),
        }
        rows.append(row)
        print(
            f"seed={seed}  N={n_samples}  "
            f"Kraskov={kraskov:+.4f}  MINE={mine:+.4f}  "
            f"|diff|={abs(mine - kraskov):.4f}",
            flush=True,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["seed", "n_samples", "me3_delta_kraskov", "me3_delta_mine", "abs_diff"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {out}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="grid root (e.g. runs/v05_dr_lock100)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="CSV output path (e.g. reports/sprint17_kraskov_vs_mine.csv)",
    )
    args = parser.parse_args()
    compare(args.root, args.out)


if __name__ == "__main__":
    main()
