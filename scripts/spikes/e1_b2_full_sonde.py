"""B-2 re-estimation with the FULL (B, K*d) Me3 sonde (gaussian world).

Context
-------
Until this change `lesion_phase` stored the Me3 probe as a 1-D mean-pool
(`fused.flatten(1).mean(-1)`), so MINE was pinned at 0.000 on every seed
(reports/sprint17_kraskov_vs_mine.csv) and the "Kraskov~=MINE~=0" accord
was an estimator artefact, not a corroborated null.

This spike rebuilds the loop EXACTLY as bouben_sens.cli._lesion_impl does,
runs lesion_phase(..., probe_pool="full") to retain the full (B, 896)
representation, pools pre/post codes + labels across a seed's grid cells,
and re-estimates B-2 = MI(post;y) - MI(pre;y) with three estimators:

  * Kraskov-KSG (multi-D, joint)  -> me3_delta
  * MINE / Donsker-Varadhan        -> me3_delta_mine
  * histogram binning (per-dim sum) -> _delta_binning

Bootstrap percentile CIs (resampling pooled rows) accompany each estimate.

CAVEAT: BOUBA_SENS_STUDYFORREST_DATA is unset and no Studyforrest cache
exists on disk, so this runs on the `gaussian` world (same fallback as the
retracted Sprint 14a/17 run). NOT a Studyforrest result.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from bouba_sens.metrics.mi_migration import me3_delta, me3_delta_mine, me3_mi

MODALITIES = ("audio", "vision", "tactile", "gravity", "force")
TIMINGS = ("T2",)
SNRS = (0.0, 0.1, 0.3)  # snr_floor sweep


def build_loop(seed: int):
    from track_p.multiplexer import GammaThetaMultiplexer

    from bouba_sens.encoders import (
        AudioEncoder,
        ForceEncoder,
        GravityEncoder,
        TactileEncoder,
        VisionEncoder,
    )
    from bouba_sens.head import IntegrationHead
    from bouba_sens.loop import AdaptationLoop
    from bouba_sens.nerve import CrossModalNerve
    from bouba_sens.sensory import SensoryWML
    from bouba_sens.world.gaussian import GaussianWorld

    world_obj = GaussianWorld(seed=seed)
    mux = GammaThetaMultiplexer(seed=0, constellation_lock_after=100)
    sensories = {
        "audio": SensoryWML(0, "audio", AudioEncoder(), mux, seed=1),
        "vision": SensoryWML(1, "vision", VisionEncoder(), mux, seed=2),
        "tactile": SensoryWML(2, "tactile", TactileEncoder(), mux, seed=3),
        "gravity": SensoryWML(3, "gravity", GravityEncoder(), mux, seed=4),
        "force": SensoryWML(4, "force", ForceEncoder(), mux, seed=5),
    }
    nerve = CrossModalNerve(mux, seed=0, codebook_lock_after=100)
    head = IntegrationHead(n_classes=4)
    return AdaptationLoop(world_obj, mux, sensories, nerve, head)


def run_cell(seed: int, modality: str, snr_floor: float, steps: int):
    from bouba_sens.lesion import LesionSpec, m2_snr_schedule

    loop = build_loop(seed)

    def schedule(step: int) -> float:
        return m2_snr_schedule(step, snr_init=1.0, snr_floor=snr_floor, k=40)

    spec = LesionSpec(modality=modality, mode="M2", timing="T2", schedule=schedule)  # type: ignore[arg-type]
    return loop.lesion_phase(spec, steps=steps, batch_size=16, seed=seed, probe_pool="full")


def _delta_binning(pre: np.ndarray, post: np.ndarray, labels: np.ndarray, bins: int = 8) -> float:
    """Histogram-plugin MI(codes;y) delta, summed over dims (upper bound proxy)."""

    def mi_one(x_col: np.ndarray, y: np.ndarray) -> float:
        xb = np.digitize(x_col, np.histogram_bin_edges(x_col, bins=bins)[1:-1])
        joint = np.zeros((bins + 1, int(y.max()) + 1))
        for xi, yi in zip(xb, y.astype(int)):
            joint[xi, yi] += 1
        p = joint / joint.sum()
        px = p.sum(1, keepdims=True)
        py = p.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = np.nansum(p * np.log2(p / (px * py + 1e-12) + 1e-12))
        return float(max(mi, 0.0))

    def mi_all(x: np.ndarray, y: np.ndarray) -> float:
        return sum(mi_one(x[:, j], y) for j in range(x.shape[1]))

    return mi_all(post, labels) - mi_all(pre, labels)


def bootstrap_ci(fn, pre, post, labels, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    n = pre.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(pre[idx], post[idx], labels[idx]))
        except Exception:
            continue
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.array(vals)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("reports/b2_full_sonde.csv"))
    args = ap.parse_args()

    rows = []
    for seed in args.seeds:
        pres, posts, labels = [], [], []
        for mod in MODALITIES:
            for snr in SNRS:
                rep = run_cell(seed, mod, snr, args.steps)
                pres.append(rep.pre_lesion_codes.detach().cpu().numpy().astype(np.float64))
                posts.append(rep.post_lesion_codes.detach().cpu().numpy().astype(np.float64))
                labels.append(rep.probe_labels.detach().cpu().numpy())
        pre = np.concatenate(pres, 0)
        post = np.concatenate(posts, 0)
        lab = np.concatenate(labels, 0)
        n, d = pre.shape

        # Point estimates
        krask = me3_delta(torch.from_numpy(pre), torch.from_numpy(post), torch.from_numpy(lab))
        binn = _delta_binning(pre, post, lab)
        try:
            mine = me3_delta_mine(pre, post, lab, n_epochs=300, batch_size=min(256, n))
        except Exception as exc:  # MINE can diverge; report tel quel
            mine = float("nan")
            print(f"  [seed {seed}] MINE failed: {exc}", flush=True)

        # Bootstrap CIs (Kraskov + binning; MINE too costly to bootstrap fully)
        k_m, k_lo, k_hi = bootstrap_ci(
            lambda a, b, c: me3_delta(torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(c)),
            pre, post, lab, n_boot=args.n_boot, seed=seed,
        )
        b_m, b_lo, b_hi = bootstrap_ci(_delta_binning, pre, post, lab, n_boot=args.n_boot, seed=seed)

        row = {
            "seed": seed, "n": n, "d": d,
            "kraskov_delta": krask, "kraskov_ci_lo": k_lo, "kraskov_ci_hi": k_hi,
            "mine_delta": mine,
            "binning_delta": binn, "binning_ci_lo": b_lo, "binning_ci_hi": b_hi,
        }
        rows.append(row)
        print(
            f"seed={seed} N={n} d={d} | Kraskov={krask:+.4f} "
            f"[{k_lo:+.4f},{k_hi:+.4f}] | MINE={mine:+.4f} | "
            f"binning={binn:+.4f} [{b_lo:+.4f},{b_hi:+.4f}]",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
