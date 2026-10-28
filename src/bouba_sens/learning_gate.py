"""Learning gate of the rev. C designs (PI decision 2026-10-28, P1).

ADR-0024: with the default argmax emission the network does not learn the
task, so every rev. C pilot first checks that the network does. The gate is
measured on the task, on held-out rows of the campaign's own label:

* ``full_bacc``: balanced accuracy of the full network (all five
  modalities live, head on the fused code);
* ``query_bacc[m]``: the same with only modality m live (the other four
  zeroed, as in ``AdaptationLoop.query_accuracy``).

Both use ``loop._forward``, whose fused code draws Gumbel noise from the
global RNG; the evaluation runs inside ``torch.random.fork_rng`` under the
fixed seed ``GATE_EVAL_SEED``, so it neither depends on nor changes the
training RNG stream. Balanced accuracy is the mean recall over the classes
present in the held-out rows (an input-independent predictor scores 0.25
with 4 classes present).

Pass rule (``learning_gate``): mean over seeds of the per-seed mean
``full_bacc`` >= 0.40 and every seed >= 0.30; for v0.9 also the mean over
seeds of (full_bacc - max_m query_bacc) >= 0.05.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from bouba_sens.sensory import MODALITIES
from bouba_sens.world.base import WorldSample

GATE_EVAL_SEED = 20261028
GATE_MEAN = 0.40
GATE_PER_SEED = 0.30
GATE_OVER_SINGLE = 0.05


def balanced_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    recalls = [float(np.mean(pred[y == c] == c)) for c in np.unique(y)]
    return float(np.mean(recalls))


def _predict(loop: Any, sample: WorldSample) -> np.ndarray:
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(GATE_EVAL_SEED)
        _, logits = loop._forward(sample)
    return logits.argmax(-1).numpy()  # type: ignore[no-any-return]


def heldout_task_metrics(loop: Any, sample: WorldSample) -> dict[str, Any]:
    """Full-network and single-modality balanced accuracies on ``sample``."""
    y = sample.label.numpy()
    out: dict[str, Any] = {"full_bacc": balanced_accuracy(_predict(loop, sample), y)}
    q: dict[str, float] = {}
    for m in MODALITIES:
        masked = replace(
            sample,
            **{
                k: (getattr(sample, k) if k == m else torch.zeros_like(getattr(sample, k)))
                for k in MODALITIES
            },
        )
        q[m] = balanced_accuracy(_predict(loop, masked), y)
    out["query_bacc"] = q
    out["best_query_bacc"] = max(q.values())
    out["n_rows"] = len(y)
    return out


def learning_gate(
    per_seed: Mapping[int, Sequence[Mapping[str, Any]]], *, need_over_single: bool
) -> dict[str, Any]:
    """Apply the pass rule to the metrics of every seed (module docstring)."""
    seed_mean = {s: float(np.mean([m["full_bacc"] for m in ms])) for s, ms in per_seed.items()}
    over = {
        s: float(np.mean([m["full_bacc"] - m["best_query_bacc"] for m in ms]))
        for s, ms in per_seed.items()
    }
    mean = float(np.mean(list(seed_mean.values())))
    ok = mean >= GATE_MEAN and min(seed_mean.values()) >= GATE_PER_SEED
    over_mean = float(np.mean(list(over.values())))
    if need_over_single:
        ok = ok and over_mean >= GATE_OVER_SINGLE
    return {
        "per_seed_full_bacc": seed_mean,
        "mean_full_bacc": mean,
        "min_seed_full_bacc": min(seed_mean.values()),
        "per_seed_over_best_single": over,
        "mean_over_best_single": over_mean,
        "rule": (
            f"mean >= {GATE_MEAN}, every seed >= {GATE_PER_SEED}"
            + (f", mean over best single >= {GATE_OVER_SINGLE}" if need_over_single else "")
        ),
        "passed": bool(ok),
    }
