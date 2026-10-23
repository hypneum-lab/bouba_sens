#!/usr/bin/env python3
"""Power analysis for the 2026-10-23 draft pre-registrations.

Supports the three DRAFT OSF amendments (not yet filed):

* `docs/osf/amendment-v0.7-tost-equivalence-2026-10-23.md` (B-1/B-2 TOST)
* `docs/osf/amendment-v0.8-subspace-shift-2026-10-23.md` (Page trend)
* `docs/osf/amendment-v0.9-decorrelation-window-2026-10-23.md`

Reads only data already committed to the repo:

* B-1: `reports/v0.5_amedi_curve_multiseed_10seed_final.json`
  (10 seeds x 5 LOCK_AFTER, per-seed grid-median Me7, Q3/Q3+).
* B-2: per-seed Me3 delta tables of ADR-0019
  (`docs/adr/0019-b2-dimensional-mine-criterion.md`), copied below
  verbatim because the per-seed JSONs are not tracked.

No training, no new data. The unit of analysis is the seed.

Sections 3-4 have no pilot data (the metrics do not exist yet); they
report power as a function of a standardised effect size, plus a
planning SD borrowed from the B-1 Me7 spread for accuracy deltas.

Usage:
    python scripts/power_prereg_2026_10.py [--sims 100000]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ALPHA = 0.05
B1_BOUND = 0.05  # OSF Q6JYN B-1 threshold (frozen)
B2_BOUND = 0.10  # OSF Q6JYN B-2 threshold, bits (frozen)

# ADR-0019 per-seed B-2 = MI_post - MI_pre (bits), 5 seeds each.
B2_TABLES: dict[str, list[float]] = {
    "gaussian_kraskov_1d (E1.3)": [-0.073, 0.033, 0.000, -0.082, -0.034],
    "gaussian_mine_full (E1.3)": [-0.016, -0.007, 0.011, -0.171, -0.008],
    "gaussian_kraskov_full (E1.3b)": [-0.077, 0.000, 0.000, -0.063, 0.080],
    "gaussian_mine_full (E1.3b)": [-0.003, 0.012, -0.008, 0.120, 0.032],
    "studyforrest_kraskov_1d (E1.3c)": [0.0, -0.0049, 0.0, 0.0, 0.0],
    "ccfilm_kraskov_1d (E1.3d)": [-0.0800, 0.0000, -0.0274, 0.0069, -0.0579],
    "ccfilm_mine_full (E1.3d)": [-0.019, -0.024, 0.013, 0.013, 0.0042],
}


def tost_power_mc(mu: float, sd: float, n: int, bound: float, sims: int, rng) -> float:
    """Monte-Carlo power of a one-sample TOST (bounds +-bound, alpha)."""
    x = rng.normal(mu, sd, size=(sims, n))
    m = x.mean(1)
    se = x.std(1, ddof=1) / math.sqrt(n)
    tc = stats.t.ppf(1 - ALPHA, n - 1)
    return float(np.mean(((m + bound) / se > tc) & ((bound - m) / se > tc)))


def iut_power_mc(mu: np.ndarray, cov: np.ndarray, n: int, bound: float, sims: int, rng) -> float:
    """Power of the intersection-union test: TOST must pass in every condition."""
    x = rng.multivariate_normal(mu, cov, size=(sims, n))  # (sims, n, k)
    m = x.mean(1)
    se = x.std(1, ddof=1) / math.sqrt(n)
    tc = stats.t.ppf(1 - ALPHA, n - 1)
    ok = ((m + bound) / se > tc) & ((bound - m) / se > tc)
    return float(np.mean(ok.all(axis=1)))


def n_normal_approx(mu: float, sd: float, bound: float, power: float) -> int:
    """Large-sample normal-approximation N for TOST."""
    za = stats.norm.ppf(1 - ALPHA)
    if abs(mu) < 1e-12:
        zb = stats.norm.ppf(1 - (1 - power) / 2)
        return math.ceil(((za + zb) * sd / bound) ** 2)
    zb = stats.norm.ppf(power)
    return math.ceil(((za + zb) * sd / (bound - abs(mu))) ** 2)


def sd_ucl(sd: float, n: int, level: float = 0.80) -> float:
    """One-sided upper confidence limit of an SD (conservative planning value)."""
    df = n - 1
    return sd * math.sqrt(df / stats.chi2.ppf(1 - level, df))


def page_power_mc(levels: int, n: int, effect: float, sd_seed: float, sims: int, rng) -> float:
    """Power of a one-sided Page L test (blocks = seeds).

    Linear trend whose total range across the ordered levels is `effect`
    within-seed SDs; seeds add a random intercept of SD `sd_seed`. A
    predicted decreasing trend is tested by reversing the column order,
    so the same power applies.
    """
    x = np.linspace(0.0, effect, levels)
    hits = 0
    for _ in range(sims):
        data = x + rng.normal(0, 1, (n, levels)) + rng.normal(0, sd_seed, (n, 1))
        hits += stats.page_trend_test(data, method="asymptotic").pvalue < ALPHA
    return hits / sims


def t_one_sided_power_mc(mu: float, sd: float, n: int, sims: int, rng) -> float:
    x = rng.normal(mu, sd, size=(sims, n))
    t = x.mean(1) / (x.std(1, ddof=1) / math.sqrt(n))
    return float(np.mean(t > stats.t.ppf(1 - ALPHA, n - 1)))


def smallest_n(fn, target: float, n_min: int = 3, n_max: int = 200) -> int | None:
    for n in range(n_min, n_max + 1):
        if fn(n) >= target:
            return n
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--sims", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rep = json.loads(
        (args.repo / "reports/v0.5_amedi_curve_multiseed_10seed_final.json").read_text()
    )
    locks = [50, 75, 100, 125, 150]
    seeds = rep["all_seeds"]
    y = np.array([[rep["matrix"][str(s)][str(la)] for la in locks] for s in seeds])

    print(f"== B-1 (Me7), 10 seeds x 5 LOCK_AFTER, bound +-{B1_BOUND:.3f}")
    for j, la in enumerate(locks):
        print(f"  LOCK={la:>3}: mean={y[:, j].mean():+.4f} sd={y[:, j].std(ddof=1):.4f}")
    seed_mean = y.mean(1)
    mu1, sd1 = float(seed_mean.mean()), float(seed_mean.std(ddof=1))
    print(f"  per-seed mean over LOCK: mean={mu1:+.4f} sd={sd1:.4f}")
    # Observed-data TOST on the existing 10 seeds (descriptive only).
    se = sd1 / math.sqrt(len(seeds))
    p_lo = 1 - stats.t.cdf((mu1 + B1_BOUND) / se, len(seeds) - 1)
    p_hi = 1 - stats.t.cdf((B1_BOUND - mu1) / se, len(seeds) - 1)
    print(f"  descriptive TOST on existing 10 seeds: p_lower={p_lo:.2e} p_upper={p_hi:.2e}")

    sd_max_lock = float(y.std(0, ddof=1).max())
    cov = np.cov(y, rowvar=False)
    for label, mu, sd in [
        ("primary, mu=0", 0.0, sd1),
        ("primary, mu=observed", mu1, sd1),
        ("primary, mu=0, sd=max per-LOCK sd", 0.0, sd_max_lock),
        ("primary, mu=observed, sd=UCL80", mu1, sd_ucl(sd1, len(seeds))),
    ]:
        for target in (0.80, 0.90):
            n = smallest_n(
                lambda k, mu=mu, sd=sd: tost_power_mc(mu, sd, k, B1_BOUND, args.sims // 5, rng),
                target,
            )
            print(
                f"  [{label}] sd={sd:.4f} power>={target:.0%}: N_mc={n} "
                f"N_normal={n_normal_approx(mu, sd, B1_BOUND, target)}"
            )
    for label, mu in [("IUT 5 LOCKs, mu=0", np.zeros(5)), ("IUT 5 LOCKs, mu=observed", y.mean(0))]:
        for target in (0.80, 0.90):
            n = smallest_n(
                lambda k, mu=mu: iut_power_mc(mu, cov, k, B1_BOUND, args.sims // 20, rng), target, 5
            )
            print(f"  [{label}] power>={target:.0%}: N_mc={n}")
    for n in (10, 20, 30):
        print(
            f"  power at N={n}: primary(mu=obs)={tost_power_mc(mu1, sd1, n, B1_BOUND, args.sims, rng):.3f} "
            f"IUT(mu=obs)={iut_power_mc(y.mean(0), cov, n, B1_BOUND, args.sims // 10, rng):.3f}"
        )

    print(f"\n== B-2 (Me3 delta, bits), bound +-{B2_BOUND:.2f}")
    sds = {}
    for k, v in B2_TABLES.items():
        a = np.array(v)
        sds[k] = float(a.std(ddof=1))
        print(f"  {k:34s} mean={a.mean():+.4f} sd={sds[k]:.4f}")
    worst = max(sds, key=sds.get)
    sd2 = sds[worst]
    print(f"  planning sd = max over tables = {sd2:.4f} ({worst})")
    sd2u = sd_ucl(sd2, 5)
    print(f"  planning sd UCL80 (df=4) = {sd2u:.4f}")
    for sd_label, sd in (("sd", sd2), ("sd_UCL80", sd2u)):
        for mu in (0.0, -0.04):
            for target in (0.80, 0.90):
                n = smallest_n(
                    lambda k, mu=mu, sd=sd: tost_power_mc(mu, sd, k, B2_BOUND, args.sims // 5, rng),
                    target,
                )
                print(
                    f"  [{sd_label}, mu={mu:+.2f}] power>={target:.0%}: N_mc={n} "
                    f"N_normal={n_normal_approx(mu, sd, B2_BOUND, target)}"
                )
    for n in (10, 20):
        print(
            f"  power at N={n}: mu=0,sd={tost_power_mc(0.0, sd2, n, B2_BOUND, args.sims, rng):.3f} "
            f"mu=-0.04,sd_UCL80={tost_power_mc(-0.04, sd2u, n, B2_BOUND, args.sims, rng):.3f}"
        )

    print("\n== Page trend test power (one-sided, alpha=0.05), no pilot data")
    page_sims = max(args.sims // 50, 500)
    for levels, label in ((7, "v0.8 LOCK_AFTER grid"), (5, "v0.9 onset grid")):
        for n in (10, 20, 30):
            row = []
            for eff in (0.5, 1.0, 1.5):
                row.append(
                    f"range={eff}sd:{page_power_mc(levels, n, eff, 1.0, page_sims, rng):.2f}"
                )
            print(f"  {label} k={levels} N={n}: " + " ".join(row))

    print("\n== v0.9 decorrelation deficit D(onset=0), SESOI 0.05 accuracy")
    sd_d = sd_max_lock
    sd_du = sd_ucl(sd_max_lock, len(seeds))
    for sd_label, sd in (("sd=max per-LOCK Me7 sd", sd_d), ("sd_UCL80", sd_du)):
        for n in (10, 20):
            print(
                f"  [{sd_label}={sd:.4f}] N={n}: "
                f"one-sided t (mu=0.05) power={t_one_sided_power_mc(0.05, sd, n, args.sims, rng):.3f} "
                f"TOST+-0.05 (mu=0) power={tost_power_mc(0.0, sd, n, B1_BOUND, args.sims, rng):.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
