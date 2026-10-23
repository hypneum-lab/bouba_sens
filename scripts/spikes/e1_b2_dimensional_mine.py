"""E1/E2 — B-2 re-run with MINE at dimensionally-justified N.

Backlog E1+E2 (EXPERIMENT-BACKLOG.md, from REVIEW.md):

  E2: replace the heuristic "probe batch >= 128" with N >= f(D), where
      D is the *effective dimensionality* of the Me3 probe representation
      and f is the chosen estimator's known sample-complexity.
  E1: 1. compute the actual probe dimensionality D of the Me3 representation;
      2. fix N via E2 (N >~ f(D));
      3. re-estimate B-2 MI with MINE (Belghazi 2018) at that N, cross-check
         vs Kraskov k-NN and adaptive binning; report bootstrap CIs.

Key methodological point this script makes concrete
---------------------------------------------------
`AdaptationLoop.lesion_phase` stores the probe as `fused.flatten(1).mean(-1)`,
i.e. a **1-D scalar per sample** (loop.py:189,246). The full fused
representation is `(B, K, d_hidden)` = `(B, 8192)`. The retraction/sprint17
ran MINE on the 1-D mean-pooled code -> MINE returned exactly 0.0 every seed
(reports/sprint17_kraskov_vs_mine.csv). This script captures the FULL
`(B, d_fused)` probe so D, N, and the estimator comparison are meaningful.

NOTE on world: the headline B-2 null is on the 4.5-modal real Studyforrest
bridge. Provision its cache with
`scripts/fetch_studyforrest_sample.py --source openneuro` (OpenNeuro
ds000113 via S3) then export BOUBA_SENS_STUDYFORREST_DATA and run with
`--world studyforrest` (loads StudyforrestRealWorld, mode real5). Without the
cache, --world defaults to gaussian. No number is fabricated; whatever the
estimators return is reported verbatim. See ADR-0019 §E1.3c for the
real-bridge table and the audio/vision-proxy provenance caveat.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from bouba_sens.lesion import LesionSpec, m2_snr_schedule
from bouba_sens.sensory import MODALITIES, SensoryWML


def build_loop(world_name: str, seed: int, lock_after: int):
    from track_p.multiplexer import GammaThetaMultiplexer

    from bouba_sens.cli import _build_world
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

    world_obj = _build_world(world_name, seed)
    mux = GammaThetaMultiplexer(seed=seed, constellation_lock_after=lock_after)
    sensories = {
        "audio": SensoryWML(0, "audio", AudioEncoder(), mux, seed=seed + 1),
        "vision": SensoryWML(1, "vision", VisionEncoder(), mux, seed=seed + 2),
        "tactile": SensoryWML(2, "tactile", TactileEncoder(), mux, seed=seed + 3),
        "gravity": SensoryWML(3, "gravity", GravityEncoder(), mux, seed=seed + 4),
        "force": SensoryWML(4, "force", ForceEncoder(), mux, seed=seed + 5),
    }
    nerve = CrossModalNerve(mux, seed=seed, transducer_gating="hard")
    head = IntegrationHead(n_classes=4)
    return AdaptationLoop(world_obj, mux, sensories, nerve, head)


@torch.no_grad()
def _full_fused(loop, probe):
    carriers = {m: loop.sensories[m].step(getattr(probe, m)) for m in MODALITIES}
    fused = loop.nerve.fuse(carriers)  # (B, K, d_hidden)
    return fused.flatten(1).detach().cpu()  # (B, d_fused)


def capture_probe(
    world_name: str,
    seed: int,
    lock_after: int,
    steps_train: int,
    steps_lesion: int,
    batch_size: int,
):
    """Run P1 pretrain + P2 lesion, return full (B, d_fused) pre/post + labels.

    Mirrors AdaptationLoop.lesion_phase exactly, except the probe is kept
    at full dimensionality instead of being mean-pooled to (B,).
    """
    loop = build_loop(world_name, seed, lock_after)
    loop.pretrain(steps_train, batch_size=batch_size, seed=seed)

    probe = loop.world.sample(batch_size=batch_size, seed=seed - 99_999)
    pre_full = _full_fused(loop, probe)
    labels = probe.label.detach().cpu()

    lesion = LesionSpec(
        modality="audio",
        mode="M2",
        timing="T1",
        schedule=lambda s: m2_snr_schedule(s, snr_init=0.0, snr_floor=-20.0, k=steps_lesion),
    )
    loop.lesion_phase(lesion, steps=steps_lesion, batch_size=batch_size, seed=seed)
    post_full = _full_fused(loop, probe)
    return pre_full.numpy(), post_full.numpy(), labels.numpy()


# --------------------------------------------------------------------------
# Effective / intrinsic dimensionality estimators
# --------------------------------------------------------------------------
def participation_ratio(x: np.ndarray) -> float:
    """Linear effective dim: (sum eig)^2 / sum(eig^2) of the covariance."""
    xc = x - x.mean(0, keepdims=True)
    cov = (xc.T @ xc) / max(1, xc.shape[0] - 1)
    ev = np.linalg.eigvalsh(cov)
    ev = ev[ev > 0]
    if ev.size == 0:
        return 0.0
    return float(ev.sum() ** 2 / (ev**2).sum())


def twonn_intrinsic_dim(x: np.ndarray, seed: int = 0) -> float:
    """Two-NN intrinsic-dimension estimator (Facco et al. 2017)."""
    # pairwise distances
    d2 = np.sum(x * x, 1)[:, None] + np.sum(x * x, 1)[None, :] - 2 * x @ x.T
    np.fill_diagonal(d2, np.inf)
    d2 = np.clip(d2, 0, None)
    dist = np.sqrt(d2)
    dist.sort(axis=1)
    r1, r2 = dist[:, 0], dist[:, 1]
    mask = (r1 > 1e-12) & (r2 > r1)
    mu = r2[mask] / r1[mask]
    mu = mu[np.isfinite(mu) & (mu > 1)]
    if mu.size < 5:
        return float("nan")
    # MLE: d = N / sum(log mu)
    return float(mu.size / np.log(mu).sum())


# --------------------------------------------------------------------------
# B-2 estimators on the 1-D Me3 scalar code (mean-pool), as in the repo
# --------------------------------------------------------------------------
def me3_scalar(full: np.ndarray) -> np.ndarray:
    """Reproduce the repo's mean-pool: (B, d_fused) -> (B,)."""
    return full.mean(axis=1)


def mi_binning_bits(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Adaptive (equal-frequency) binning plug-in MI in bits.

    x: continuous 1-D code; y: integer labels. Bins x by quantiles.
    """
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    if qs.size < 2:
        return 0.0
    xb = np.clip(np.digitize(x, qs[1:-1]), 0, qs.size - 2)
    yb = y.astype(int)
    joint = np.zeros((xb.max() + 1, yb.max() + 1))
    for xi, yi in zip(xb, yb, strict=False):
        joint[xi, yi] += 1
    joint /= joint.sum()
    px = joint.sum(1, keepdims=True)
    py = joint.sum(0, keepdims=True)
    nz = joint > 0
    mi_nats = float(np.sum(joint[nz] * np.log(joint[nz] / (px @ py)[nz])))
    return mi_nats / math.log(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="gaussian")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lock-after", type=int, default=100)
    ap.add_argument("--steps-train", type=int, default=200)
    ap.add_argument("--steps-lesion", type=int, default=100)
    ap.add_argument(
        "--batch-size", type=int, default=512, help="probe batch N (E2 sets this from f(D))"
    )
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--out", type=Path, default=Path("reports/e1_b2_dimensional_mine.json"))
    args = ap.parse_args()

    from nerve_wml.methodology import mi_kraskov_ksg_continuous

    from bouba_sens.metrics.mi_migration import me3_mi_mine

    rng = np.random.default_rng(0)
    per_seed = []
    d_pr_list, d_2nn_list = [], []

    for seed in range(args.seeds):
        pre, post, labels = capture_probe(
            args.world,
            seed,
            args.lock_after,
            args.steps_train,
            args.steps_lesion,
            args.batch_size,
        )
        n_samples, d_fused = pre.shape

        # --- E1.1 dimensionality of the Me3 probe representation ---
        d_pr = participation_ratio(post)
        d_2nn = twonn_intrinsic_dim(post, seed=seed)
        d_pr_list.append(d_pr)
        if math.isfinite(d_2nn):
            d_2nn_list.append(d_2nn)

        # --- B-2 = MI(post) - MI(pre) with three estimators ---
        x_pre, x_post = me3_scalar(pre), me3_scalar(post)
        y = labels.astype(float)

        # Kraskov k-NN on the 1-D scalar (repo's me3_mi path uses sklearn;
        # here we use nerve_wml KSG directly for an apples-to-apples MI).
        k_pre = mi_kraskov_ksg_continuous(x_pre.reshape(-1, 1), y.reshape(-1, 1)) / math.log(2)
        k_post = mi_kraskov_ksg_continuous(x_post.reshape(-1, 1), y.reshape(-1, 1)) / math.log(2)
        kras_delta = k_post - k_pre

        # Adaptive binning plug-in (sqrt(N) bins, Rice-like).
        n_bins = max(2, round(math.sqrt(n_samples)))
        b_pre = mi_binning_bits(x_pre, labels, n_bins)
        b_post = mi_binning_bits(x_post, labels, n_bins)
        bin_delta = b_post - b_pre

        # MINE on the 1-D scalar (matches the repo / sprint17 probe shape).
        m_pre = me3_mi_mine(
            x_pre.reshape(-1, 1), labels.reshape(-1, 1), batch_size=min(256, n_samples)
        )
        m_post = me3_mi_mine(
            x_post.reshape(-1, 1), labels.reshape(-1, 1), batch_size=min(256, n_samples)
        )
        mine_delta = m_post - m_pre

        # MINE on the FULL (N, d_fused) representation — the dimensionally
        # honest probe the backlog asks for (not mean-pooled).
        mf_pre = me3_mi_mine(pre, labels.reshape(-1, 1), batch_size=min(256, n_samples))
        mf_post = me3_mi_mine(post, labels.reshape(-1, 1), batch_size=min(256, n_samples))
        mine_full_delta = mf_post - mf_pre

        per_seed.append(
            {
                "seed": seed,
                "N": n_samples,
                "d_fused": d_fused,
                "D_participation_ratio": d_pr,
                "D_twoNN": d_2nn,
                "kraskov_delta_bits": kras_delta,
                "binning_delta_bits": bin_delta,
                "mine_1d_delta_bits": mine_delta,
                "mine_full_delta_bits": mine_full_delta,
                "kraskov_pre": k_pre,
                "kraskov_post": k_post,
                "mine_full_pre": mf_pre,
                "mine_full_post": mf_post,
            }
        )
        print(
            f"seed={seed} N={n_samples} d_fused={d_fused} "
            f"D_PR={d_pr:.2f} D_2NN={d_2nn:.2f} | "
            f"Kraskov={kras_delta:+.4f} bin={bin_delta:+.4f} "
            f"MINE1d={mine_delta:+.4f} MINEfull={mine_full_delta:+.4f}",
            flush=True,
        )

    def ci(vals):
        a = np.array([v for v in vals if v is not None and math.isfinite(v)], float)
        if a.size == 0:
            return {"mean": None, "ci95": [None, None], "n": 0}
        boot = np.array([rng.choice(a, a.size, replace=True).mean() for _ in range(args.bootstrap)])
        return {
            "mean": float(a.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n": int(a.size),
        }

    summary = {
        "world": args.world,
        "lock_after": args.lock_after,
        "seeds": args.seeds,
        "D_participation_ratio_mean": float(np.mean(d_pr_list)),
        "D_twoNN_mean": (float(np.mean(d_2nn_list)) if d_2nn_list else None),
        "kraskov_delta": ci([s["kraskov_delta_bits"] for s in per_seed]),
        "binning_delta": ci([s["binning_delta_bits"] for s in per_seed]),
        "mine_1d_delta": ci([s["mine_1d_delta_bits"] for s in per_seed]),
        "mine_full_delta": ci([s["mine_full_delta_bits"] for s in per_seed]),
        "per_seed": per_seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_seed"}, indent=2))


if __name__ == "__main__":
    main()
