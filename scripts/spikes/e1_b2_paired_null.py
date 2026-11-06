"""Paired label-permutation null and learning gate for the E1.3c B-2 run.

Supplementary to ``e1_b2_dimensional_mine.py`` (ADR-0019 E1.3c), which
reports B-2 against the +0.10 bit threshold with bootstrap CIs but no
null distribution. This script reruns the same probe capture (same world,
seeds, lock, steps, batch, lesion schedule) and adds, per seed:

* a learning gate: the Phase-1 training loss (first / last 20 steps,
  compared with ln 4 = chance for 4 classes) and MI(pre) itself;
* a paired permutation null for the 1-D Kraskov and binning B-2: one
  label permutation per draw, applied to both the pre- and post-lesion
  probe (same probe, same seed), so B-2 under H0 keeps the pairing.

The statistic is the mean B-2 over seeds, as in the ADR. Null draws are
cheap (no retraining): only the labels are permuted. MINE is not
included in the null (one MINE fit per draw is too costly); its values
are read from the main runner.

Empirical p = (k + 1) / (n_null + 1), one-sided (B-2 >= observed).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e1_b2_dimensional_mine import (  # noqa: E402
    LesionSpec,
    _full_fused,
    build_loop,
    m2_snr_schedule,
    me3_scalar,
    mi_binning_bits,
)


def capture(world, seed, lock_after, steps_train, steps_lesion, batch_size):
    """Same calls as ``capture_probe``, plus the Phase-1 loss log."""
    loop = build_loop(world, seed, lock_after)
    losses: list[float] = []
    loop.pretrain(steps_train, batch_size=batch_size, seed=seed, loss_log=losses)
    probe = loop.world.sample(batch_size=batch_size, seed=seed - 99_999)
    pre = _full_fused(loop, probe)
    labels = probe.label.detach().cpu()
    lesion = LesionSpec(
        modality="audio",
        mode="M2",
        timing="T1",
        schedule=lambda s: m2_snr_schedule(s, snr_init=0.0, snr_floor=-20.0, k=steps_lesion),
    )
    loop.lesion_phase(lesion, steps=steps_lesion, batch_size=batch_size, seed=seed)
    post = _full_fused(loop, probe)
    return pre.numpy(), post.numpy(), labels.numpy(), losses


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="studyforrest")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lock-after", type=int, default=100)
    ap.add_argument("--steps-train", type=int, default=200)
    ap.add_argument("--steps-lesion", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--null-seed", type=int, default=20261106)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from nerve_wml.methodology import mi_kraskov_ksg_continuous

    def kras(x, y):
        return mi_kraskov_ksg_continuous(x.reshape(-1, 1), y.reshape(-1, 1)) / math.log(2)

    rng = np.random.default_rng(args.null_seed)
    seeds_out = []
    null_k = np.zeros((args.n_null, args.seeds))
    null_b = np.zeros((args.n_null, args.seeds))
    for seed in range(args.seeds):
        pre, post, labels, losses = capture(
            args.world, seed, args.lock_after, args.steps_train, args.steps_lesion, args.batch_size
        )
        x_pre, x_post = me3_scalar(pre), me3_scalar(post)
        y = labels.astype(float)
        n_bins = max(2, round(math.sqrt(len(y))))
        k_pre, k_post = kras(x_pre, y), kras(x_post, y)
        b_pre = mi_binning_bits(x_pre, labels, n_bins)
        b_post = mi_binning_bits(x_post, labels, n_bins)
        for i in range(args.n_null):
            perm = rng.permutation(len(y))
            yp, lp = y[perm], labels[perm]
            null_k[i, seed] = kras(x_post, yp) - kras(x_pre, yp)
            null_b[i, seed] = mi_binning_bits(x_post, lp, n_bins) - mi_binning_bits(
                x_pre, lp, n_bins
            )
        counts = np.bincount(labels.astype(int), minlength=4).tolist()
        rec = {
            "seed": seed,
            "label_counts": counts,
            "loss_first20_mean": float(np.mean(losses[:20])),
            "loss_last20_mean": float(np.mean(losses[-20:])),
            "ln4": math.log(4),
            "x_pre_std": float(x_pre.std()),
            "x_post_std": float(x_post.std()),
            "kraskov_pre": k_pre,
            "kraskov_post": k_post,
            "kraskov_delta_bits": k_post - k_pre,
            "binning_pre": b_pre,
            "binning_post": b_post,
            "binning_delta_bits": b_post - b_pre,
        }
        seeds_out.append(rec)
        print(json.dumps(rec), flush=True)

    def summarise(obs_vals, null_mat):
        obs = float(np.mean(obs_vals))
        null_mean = null_mat.mean(axis=1)
        k = int(np.sum(null_mean >= obs))
        return {
            "observed_mean": obs,
            "null_median": float(np.median(null_mean)),
            "null_p99": float(np.percentile(null_mean, 99)),
            "null_p01": float(np.percentile(null_mean, 1)),
            "p_one_sided": (k + 1) / (args.n_null + 1),
            "threshold": 0.10,
            "exceeds_threshold": bool(obs >= 0.10),
            "null_p99_below_threshold": bool(np.percentile(null_mean, 99) < 0.10),
        }

    out = {
        "world": args.world,
        "seeds": args.seeds,
        "lock_after": args.lock_after,
        "steps_train": args.steps_train,
        "steps_lesion": args.steps_lesion,
        "batch_size": args.batch_size,
        "n_null": args.n_null,
        "null_seed": args.null_seed,
        "null": "paired label permutation, same permutation on pre and post probe",
        "kraskov_1d": summarise([s["kraskov_delta_bits"] for s in seeds_out], null_k),
        "binning_1d": summarise([s["binning_delta_bits"] for s in seeds_out], null_b),
        "per_seed": seeds_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_seed"}, indent=2))


if __name__ == "__main__":
    main()
