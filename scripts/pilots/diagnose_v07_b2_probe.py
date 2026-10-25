"""Diagnostic for the v0.7 B-2 pilot (pilot seeds only, descriptive).

Re-runs the frozen ``--preset v07-e2`` capture for each listed pilot
seed and reports what the Ross estimator sees: probe label counts,
distinct rows of each readout (hard / Gumbel, pre / post), the number
of non-constant columns and whether pre equals post. No estimator, no
outcome statistic. Usage:

  BOUBA_SENS_STUDYFORREST_DATA=... uv run python \
    scripts/pilots/diagnose_v07_b2_probe.py 1101 1102 1103 1104 1105
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from bouba_sens.cli import _build_world

_RUNNER = Path(__file__).resolve().parents[1] / "spikes" / "e1_b2_dimensional_mine.py"
_spec = importlib.util.spec_from_file_location("e1_runner", _RUNNER)
e1 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(e1)
F = e1.PRESETS["v07-e2"]


def describe(z: np.ndarray) -> dict:
    uniq = np.unique(z, axis=0)
    return {
        "n_distinct_rows": int(uniq.shape[0]),
        "n_nonconstant_columns": int((np.ptp(z, axis=0) > 0).sum()),
        "max_abs": float(np.abs(z).max()),
    }


def main(seeds: list[int]) -> None:
    torch.use_deterministic_algorithms(True)  # as the runner on --seed-list
    out = []
    for s in seeds:
        labels, ro = e1._capture(
            F["world"], s, F["lock_after"], F["steps_train"], F["steps_lesion"],
            train_batch=F["train_batch"], lesion_batch=F["lesion_batch"],
            probe_n=F["probe_n"], k_steps=F["k_steps"], rng_seed=s,
            readouts=("gumbel", "hard"),
        )  # fmt: skip
        vals, cnt = np.unique(labels, return_counts=True)
        rec = {
            "seed": s,
            "world_class": type(_build_world(F["world"], s)).__name__,
            "probe_label_counts": {int(v): int(c) for v, c in zip(vals, cnt, strict=True)},
        }
        for r in ("hard", "gumbel"):
            pre, post = ro[r]
            rec[r] = {
                "pre": describe(pre),
                "post": describe(post),
                "pre_equals_post": bool(np.array_equal(pre, post)),
            }
        out.append(rec)
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
