"""Known-signal calibration of the B-2 MI estimators (amendment v0.7, section 4.3).

Pre-filing, required, descriptive (not a pass/fail gate). Codes of the
confirmatory shape (N = 512, d = 896, 4 equiprobable classes) are built
as a 12-D Gaussian class mixture embedded in R^896:

    u = a * m_y + e,  e ~ N(0, I_12)        (m_y: fixed seeded unit vectors)
    z = Q u + 0.1 * n, n ~ N(0, I_896)      (Q: fixed random 896 x 12 orthonormal)

Because Q is orthonormal and the ambient noise is isotropic, the part of
z orthogonal to span(Q) is independent of y, so the true MI is that of
Q^T z = a * m_y + N(0, (1 + 0.1^2) I_12). It is computed by Monte Carlo
(10^5 draws, fixed common random numbers) and the scale ``a`` is solved
by bisection for each target MI. Baselines MI_pre in {0.25, 0.50, 1.00}
bit; at each, 20 independent (pre, post) replicate pairs per true
dMI in {-0.20, -0.10, -0.05, 0, +0.05, +0.10, +0.20} bit. dMI = 0 rows
are the known-null calibration, +-0.05 bit rows the injected effect at
the equivalence bound.

Reported per estimator (Ross 2014 primary; full-probe KSG and MINE
secondary): per cell, mean and SD of the estimated MI_pre, MI_post and
dMI, and bias = mean estimate - truth; per baseline, the least-squares
slope and intercept of estimated on true dMI over all replicates.

Classes are exactly balanced (128 rows each). Run on the filing host:

    uv run python scripts/calibrate_mi_known_signal.py --out reports/v07_mi_calibration.json

``--estimators ross,ksg`` skips MINE (much slower); the table recorded on
the OSF form must include all three.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

N_ROWS = 512
D_AMBIENT = 896
D_SIGNAL = 12
N_CLASSES = 4
NOISE_SD = 0.1
MC_DRAWS = 100_000
BASELINES_BITS: tuple[float, ...] = (0.25, 0.50, 1.00)
DELTAS_BITS: tuple[float, ...] = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20)
REPLICATES = 20
DESIGN_SEED = 20261023
LN2 = math.log(2.0)

Estimator = Callable[[NDArray[np.float64], NDArray[np.int64]], float]


@dataclass(frozen=True)
class Design:
    """Fixed parts of the generator: class directions and embedding."""

    means: NDArray[np.float64]  # (N_CLASSES, D_SIGNAL), unit rows
    embed: NDArray[np.float64]  # (D_AMBIENT, D_SIGNAL), orthonormal columns
    mc_noise: NDArray[np.float64]  # (MC_DRAWS, D_SIGNAL) common random numbers
    mc_labels: NDArray[np.int64]  # (MC_DRAWS,)


def make_design(seed: int = DESIGN_SEED, mc_draws: int = MC_DRAWS) -> Design:
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(N_CLASSES, D_SIGNAL))
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    q, _ = np.linalg.qr(rng.normal(size=(D_AMBIENT, D_SIGNAL)))
    mc_noise = rng.normal(size=(mc_draws, D_SIGNAL))
    mc_labels = rng.integers(0, N_CLASSES, size=mc_draws)
    return Design(means=means, embed=q, mc_noise=mc_noise, mc_labels=mc_labels)


def true_mi_bits(a: float, design: Design) -> float:
    """Monte-Carlo I(Y; Z) in bits for scale ``a`` (equiprobable classes)."""
    var = 1.0 + NOISE_SD**2
    centres = a * design.means  # (C, D)
    x = centres[design.mc_labels] + math.sqrt(var) * design.mc_noise
    # log N(x; c_j, var I) up to a constant common to every j
    sq = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(-1)  # (M, C)
    logk = -0.5 * sq / var
    log_own = logk[np.arange(len(x)), design.mc_labels]
    top = logk.max(axis=1)
    log_mix = top + np.log(np.exp(logk - top[:, None]).mean(axis=1))
    return float(np.mean(log_own - log_mix) / LN2)


def solve_scale(target_bits: float, design: Design, tol: float = 1e-6) -> float:
    """Bisection for ``a`` with true_mi_bits(a) = target (common random numbers)."""
    if target_bits <= 0.0:
        return 0.0
    lo, hi = 0.0, 1.0
    while true_mi_bits(hi, design) < target_bits:
        hi *= 2.0
        if hi > 1e3:
            raise ValueError(f"target {target_bits} bit not reachable")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if true_mi_bits(mid, design) < target_bits:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sample_codes(
    a: float, design: Design, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """One (N_ROWS, D_AMBIENT) code matrix with exactly balanced labels."""
    y = np.repeat(np.arange(N_CLASSES), N_ROWS // N_CLASSES).astype(np.int64)
    u = a * design.means[y] + rng.normal(size=(N_ROWS, D_SIGNAL))
    z = u @ design.embed.T + NOISE_SD * rng.normal(size=(N_ROWS, D_AMBIENT))
    return z, y


def _ross(z: NDArray[np.float64], y: NDArray[np.int64]) -> float:
    from bouba_sens.metrics.mi_ross import mi_ross_discrete_continuous

    return mi_ross_discrete_continuous(z, y, k=3)


def _ksg(z: NDArray[np.float64], y: NDArray[np.int64]) -> float:
    import torch

    from bouba_sens.metrics.mi_migration import me3_mi

    return me3_mi(torch.from_numpy(z), torch.from_numpy(y))


def _mine(z: NDArray[np.float64], y: NDArray[np.int64]) -> float:
    from bouba_sens.metrics.mi_migration import me3_mi_mine

    return float(me3_mi_mine(z, y.reshape(-1, 1), batch_size=256))


ESTIMATORS: dict[str, Estimator] = {"ross": _ross, "ksg": _ksg, "mine": _mine}


def _mean_sd(v: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(v, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def ls_line(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Least-squares slope and intercept of y on x."""
    slope, intercept = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    return float(slope), float(intercept)


def calibrate(
    estimators: dict[str, Estimator],
    *,
    baselines: Sequence[float] = BASELINES_BITS,
    deltas: Sequence[float] = DELTAS_BITS,
    replicates: int = REPLICATES,
    design: Design | None = None,
    seed: int = DESIGN_SEED,
) -> dict[str, object]:
    design = make_design(seed) if design is None else design
    scales: dict[float, float] = {}
    for base in baselines:
        for target in (base, *(base + d for d in deltas)):
            if target not in scales:
                scales[target] = solve_scale(target, design)
    truth = {t: true_mi_bits(a, design) for t, a in scales.items()}

    cells = []
    lines: dict[str, dict[str, dict[str, float]]] = {name: {} for name in estimators}
    for bi, base in enumerate(baselines):
        pts: dict[str, tuple[list[float], list[float]]] = {n: ([], []) for n in estimators}
        for di, delta in enumerate(deltas):
            post_t = base + delta
            est: dict[str, dict[str, list[float]]] = {
                n: {"pre": [], "post": [], "delta": []} for n in estimators
            }
            for r in range(replicates):
                rng = np.random.default_rng([seed, bi, di, r])
                z_pre, y_pre = sample_codes(scales[base], design, rng)
                z_post, y_post = sample_codes(scales[post_t], design, rng)
                for name, fn in estimators.items():
                    pre, post = fn(z_pre, y_pre), fn(z_post, y_post)
                    est[name]["pre"].append(pre)
                    est[name]["post"].append(post)
                    est[name]["delta"].append(post - pre)
                    pts[name][0].append(truth[post_t] - truth[base])
                    pts[name][1].append(post - pre)
            true_delta = truth[post_t] - truth[base]
            for name in estimators:
                row: dict[str, object] = {
                    "estimator": name,
                    "baseline_bits": base,
                    "delta_bits": delta,
                    "true_pre_bits": truth[base],
                    "true_post_bits": truth[post_t],
                    "true_delta_bits": true_delta,
                    "scale_pre": scales[base],
                    "scale_post": scales[post_t],
                    "label": "known-null"
                    if delta == 0.0
                    else ("injected-at-bound" if abs(delta) == 0.05 else "injected"),
                }
                for key, true_v in (
                    ("pre", truth[base]),
                    ("post", truth[post_t]),
                    ("delta", true_delta),
                ):
                    m, sd = _mean_sd(est[name][key])
                    row[f"{key}_mean"], row[f"{key}_sd"], row[f"{key}_bias"] = m, sd, m - true_v
                cells.append(row)
        for name in estimators:
            slope, intercept = ls_line(*pts[name])
            lines[name][f"{base:.2f}"] = {"slope": slope, "intercept": intercept}
    return {
        "design": {
            "n_rows": N_ROWS,
            "d_ambient": D_AMBIENT,
            "d_signal": D_SIGNAL,
            "n_classes": N_CLASSES,
            "noise_sd": NOISE_SD,
            "mc_draws": int(design.mc_noise.shape[0]),
            "replicates": replicates,
            "seed": seed,
        },
        "cells": cells,
        "slope_intercept": lines,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--estimators", default="ross,ksg,mine")
    ap.add_argument("--replicates", type=int, default=REPLICATES)
    ap.add_argument("--out", type=Path, default=Path("reports/v07_mi_calibration.json"))
    args = ap.parse_args()
    names = [n.strip() for n in args.estimators.split(",") if n.strip()]
    unknown = sorted(set(names) - set(ESTIMATORS))
    if unknown:
        raise SystemExit(f"unknown estimators: {unknown}")
    result = calibrate({n: ESTIMATORS[n] for n in names}, replicates=args.replicates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    for name, per_base in result["slope_intercept"].items():  # type: ignore[union-attr]
        for base, line in per_base.items():
            print(
                f"{name} MI_pre={base}: slope={line['slope']:.3f} intercept={line['intercept']:+.4f}"
            )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
