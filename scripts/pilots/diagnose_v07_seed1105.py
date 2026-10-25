"""Read-only diagnosis of v0.7 pilot round 2 (seed 1105 outlier).

Re-runs the frozen ``--preset v07-e2`` capture of each pilot seed step by
step (same calls as ``_capture`` in the E1 runner, keeping the lesion
report) and reports, per seed: reproduction of the committed Ross
delta (Gumbel), Phase-2 loss / accuracy curves, the post-lesion gate
values and codebook entropy, and spectral / class-structure summaries of
the Gumbel Z_pre and Z_post (participation ratio, top-1 and top-3
eigenvalue shares, between-class share of variance, distinct rows,
nearest-neighbour label agreement). No rule, no outcome statistic is
changed. Usage:

  BOUBA_SENS_STUDYFORREST_DATA=... uv run python \
    scripts/pilots/diagnose_v07_seed1105.py 1101 1102 1103 1104 1105
  ... scripts/pilots/diagnose_v07_seed1105.py redraw 1101 1105
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from bouba_sens.lesion import LesionSpec, m2_snr_schedule
from bouba_sens.metrics.mi_ross import mi_ross_discrete_continuous

_RUNNER = Path(__file__).resolve().parents[1] / "spikes" / "e1_b2_dimensional_mine.py"
_spec = importlib.util.spec_from_file_location("e1_runner", _RUNNER)
e1 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(e1)
F = e1.PRESETS["v07-e2"]


def spectrum(z: np.ndarray, y: np.ndarray) -> dict:
    zc = z - z.mean(0)
    lam = np.linalg.svd(zc.astype(np.float64), compute_uv=False) ** 2 / (len(z) - 1)
    tot = lam.sum()
    means = np.stack([z[y == c].mean(0) for c in np.unique(y)])
    counts = np.array([(y == c).sum() for c in np.unique(y)])
    between = float((counts[:, None] * (means - z.mean(0)) ** 2).sum()) / (len(z) - 1)
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = d2.argmin(1)
    return {
        "pr": float(tot**2 / (lam**2).sum()),
        "top1_share": float(lam[0] / tot),
        "top3_share": float(lam[:3].sum() / tot),
        "total_var": float(tot),
        "between_class_share": between / float(tot),
        "nn_label_agreement": float((y[nn] == y).mean()),
        "n_distinct_rows": int(np.unique(z, axis=0).shape[0]),
    }


def main(seeds: list[int]) -> None:
    torch.use_deterministic_algorithms(True)
    for s in seeds:
        torch.manual_seed(s)
        loop = e1.build_loop(F["world"], s, F["lock_after"])
        loop.pretrain(F["steps_train"], batch_size=F["train_batch"], seed=s)
        probe = e1.draw_probe(loop.world, F["probe_n"], s - 99_999, F["probe_scheme"])
        pre = e1._full_fused(loop, probe, hard=False).numpy()
        e1._full_fused(loop, probe, hard=True)  # the runner's hard capture (no RNG)
        y = probe.label.numpy()
        lesion = LesionSpec(
            modality="audio",
            mode="M2",
            timing="T1",
            schedule=lambda t: m2_snr_schedule(t, snr_init=0.0, snr_floor=-20.0, k=F["k_steps"]),
        )
        rep = loop.lesion_phase(
            lesion, steps=F["steps_lesion"], batch_size=F["lesion_batch"], seed=s,
            probe_size=F["probe_n"],
        )  # fmt: skip
        post = e1._full_fused(loop, probe, hard=False).numpy()
        r_pre = mi_ross_discrete_continuous(pre, y, k=3)
        r_post = mi_ross_discrete_continuous(post, y, k=3)
        loss = np.array(rep.loss_curve)
        acc = np.array(rep.accuracy_curve)
        rec = {
            "seed": s,
            "ross_delta_bits": r_post - r_pre,
            "ross_pre_bits": r_pre,
            "ross_post_bits": r_post,
            "loss_finite": bool(np.isfinite(loss).all()),
            "loss_first10_mean": float(loss[:10].mean()),
            "loss_last10_mean": float(loss[-10:].mean()),
            "loss_max": float(loss.max()),
            "acc_last10_mean": float(acc[-10:].mean()),
            "gate_end": rep.gate_trajectory[-1],
            "codebook_entropy_start_end": [
                rep.codebook_entropy_trajectory[0],
                rep.codebook_entropy_trajectory[-1],
            ],
            "transducers_active_end": rep.transducer_activation_trajectory[-1],
            "pre": spectrum(pre, y),
            "post": spectrum(post, y),
        }
        print(
            json.dumps(rec, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)),
            flush=True,
        )


def redraws(seeds: list[int], n: int = 5) -> None:
    """Gumbel re-draws of the post-lesion readout on the same trained network.

    Trains and lesions each seed exactly as the runner, then captures the
    Gumbel Z_post ``n`` more times with the global RNG re-seeded with
    10_000 + r, and the hard Z_post once: is the outlier a property of
    the network or of one Gumbel draw?
    """
    torch.use_deterministic_algorithms(True)
    for s in seeds:
        torch.manual_seed(s)
        loop = e1.build_loop(F["world"], s, F["lock_after"])
        loop.pretrain(F["steps_train"], batch_size=F["train_batch"], seed=s)
        probe = e1.draw_probe(loop.world, F["probe_n"], s - 99_999, F["probe_scheme"])
        y = probe.label.numpy()
        pre = e1._full_fused(loop, probe, hard=False).numpy()
        e1._full_fused(loop, probe, hard=True)
        lesion = LesionSpec(
            modality="audio",
            mode="M2",
            timing="T1",
            schedule=lambda t: m2_snr_schedule(t, snr_init=0.0, snr_floor=-20.0, k=F["k_steps"]),
        )
        loop.lesion_phase(
            lesion, steps=F["steps_lesion"], batch_size=F["lesion_batch"], seed=s,
            probe_size=F["probe_n"],
        )  # fmt: skip
        out = []
        for r in range(n):
            torch.manual_seed(10_000 + r)
            post = e1._full_fused(loop, probe, hard=False).numpy()
            sp = spectrum(post, y)
            out.append(
                {
                    "redraw": r,
                    "ross_post_bits": mi_ross_discrete_continuous(post, y, k=3),
                    "pr": sp["pr"],
                    "top1_share": sp["top1_share"],
                    "between_class_share": sp["between_class_share"],
                }
            )
        hard = e1._full_fused(loop, probe, hard=True).numpy()
        uniq, inv = np.unique(hard, axis=0, return_inverse=True)
        rows = [
            {str(c): int(((inv.ravel() == i) & (y == c)).sum()) for c in np.unique(y)}
            for i in range(len(uniq))
        ]
        rec = {
            "seed": s,
            "ross_pre_bits": mi_ross_discrete_continuous(pre, y, k=3),
            "post_redraws": out,
            "hard_post_distinct_rows": len(uniq),
            "hard_post_label_counts_per_distinct_row": rows,
        }
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "redraw":
        redraws([int(a) for a in sys.argv[2:]])
    else:
        main([int(a) for a in sys.argv[1:]])
