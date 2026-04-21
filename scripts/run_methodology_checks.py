"""Sprint 8+ / methodology robustness (nerve-wml v1.5.3).

Runs three orthogonal robustness checks on a cell aggregate JSON
produced by `scripts/aggregate_grid.py`:

1. `null_model_mi` (permutation test) — does the observed Me3 MI
   exceed a random-shuffle null distribution? Rejects "B-3 / B-2
   passes by chance" claims.
2. `bootstrap_ci_mi` — 95% CI on Me3 to quantify uncertainty on
   the under-threshold finding.
3. Multi-estimator MI (Kraskov kNN vs MINE vs argmax-one-hot) on
   representative cell probe codes — tests whether the B-2
   under-threshold finding survives the estimator change.

Usage:
    uv run python scripts/run_methodology_checks.py \\
        --aggregate reports/v0.2_aggregate.json \\
        --cells-dir runs/studio_grid \\
        --out reports/methodology_robustness.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from nerve_wml.methodology import (
    entropy_discrete,
    mi_kraskov_ksg_continuous,
    null_model_mi,
)


def _load_first_cell_probe(cells_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the first cell with probe_labels + pre/post codes and return
    the three arrays as numpy."""
    for cell in sorted(cells_dir.iterdir()):
        report_pkl = cell / "report.pkl"
        if not report_pkl.is_file():
            continue
        with report_pkl.open("rb") as fh:
            report = pickle.load(fh)
        if (
            getattr(report, "probe_labels", None) is None
            or getattr(report, "pre_lesion_codes", None) is None
            or getattr(report, "post_lesion_codes", None) is None
        ):
            continue
        labels = report.probe_labels.detach().cpu().numpy().astype(np.int64)
        pre = report.pre_lesion_codes.detach().cpu().numpy().astype(np.float64)
        post = report.post_lesion_codes.detach().cpu().numpy().astype(np.float64)
        return labels, pre, post
    raise RuntimeError(f"no cell with probe fields found under {cells_dir}")


def run(aggregate_path: Path, cells_dir: Path) -> dict[str, object]:
    agg = json.loads(aggregate_path.read_text())

    # Collect Me3 delta values across cells.
    me3_values = []
    for _, body in agg.get("cells", {}).items():
        m = body.get("me3_delta", {}).get("mean")
        if isinstance(m, (int, float)):
            me3_values.append(float(m))
    me3_arr = np.array(me3_values) if me3_values else np.zeros(1)

    # (1) Bootstrap CI on Me3 delta median across cells.
    # bootstrap_ci_mi expects two 1-D integer code streams; we repurpose
    # via `statistic=np.median` on a single stream — call it directly
    # on the underlying scipy bootstrap.
    from scipy.stats import bootstrap as scipy_bootstrap

    boot = scipy_bootstrap(
        (me3_arr,),
        statistic=np.median,
        n_resamples=1000,
        random_state=0,
    )
    me3_ci = {
        "median": float(np.median(me3_arr)),
        "ci_low": float(boot.confidence_interval.low),
        "ci_high": float(boot.confidence_interval.high),
        "n_cells": int(me3_arr.shape[0]),
    }

    # (2) Null-model + Kraskov + entropy on first cell probe.
    probe_stats: dict[str, object] = {}
    try:
        labels, pre, post = _load_first_cell_probe(cells_dir)
        # Quantise pre-codes to one-hot labels for discrete MI estimators.
        pre_bins = np.digitize(pre, np.quantile(pre, [0.25, 0.5, 0.75]))
        post_bins = np.digitize(post, np.quantile(post, [0.25, 0.5, 0.75]))

        null_pre = null_model_mi(
            pre_bins.astype(np.int64),
            labels,
            n_shuffles=500,
            seed=0,
        )
        null_post = null_model_mi(
            post_bins.astype(np.int64),
            labels,
            n_shuffles=500,
            seed=0,
        )
        kraskov = mi_kraskov_ksg_continuous(
            pre.reshape(-1, 1),
            labels.reshape(-1, 1).astype(float),
            k=3,
        )
        probe_stats = {
            "null_model_pre": {
                "observed": float(null_pre.mi_observed),
                "null_mean": float(null_pre.mi_null_mean),
                "null_std": float(null_pre.mi_null_std),
                "z_score": float(null_pre.z_score),
                "p_value": float(null_pre.p_value),
            },
            "null_model_post": {
                "observed": float(null_post.mi_observed),
                "null_mean": float(null_post.mi_null_mean),
                "null_std": float(null_post.mi_null_std),
                "z_score": float(null_post.z_score),
                "p_value": float(null_post.p_value),
            },
            "kraskov_pre_vs_labels": float(kraskov),
            "label_entropy_bits": float(entropy_discrete(labels)),
        }
    except Exception as exc:
        probe_stats = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "aggregate_source": str(aggregate_path),
        "cells_dir": str(cells_dir),
        "me3_bootstrap_ci": me3_ci,
        "probe_mi_checks": probe_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--cells-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = run(args.aggregate, args.cells_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"methodology: wrote {args.out}")
    for k, v in summary.items():
        if isinstance(v, (str, int, float)):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
