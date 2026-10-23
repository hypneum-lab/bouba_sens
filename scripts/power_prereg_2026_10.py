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

Two kinds of output:

* "component-test power only": power of one test in isolation (the
  original tables of the drafts);
* "complete decision rule": Monte-Carlo probability of each
  pre-registered verdict, simulating the full rule (conjunctions,
  magnitude guards, fixed-sequence gates, correlated outcomes,
  quantisation, rank exclusions and the protocol-failure rule).

`page_perm_pvalue` is the reference implementation of the
pre-registered Page test (within-block permutation, mid-ranks). The
complete-rule simulations use its permutation critical value
(`page_perm_crit`, no-tie calibration) rather than the asymptotic test.

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


# --------------------------------------------------------------------------
# Complete decision rules
# --------------------------------------------------------------------------

# Interval-based verdict classes (shared by v0.7 and the v0.9 magnitude axis).
EQUIV, BEYOND_POS, BEYOND_NEG, POS_UNRES, NEG_UNRES, INCONCL = range(6)
CLASS_NAMES = ("equiv", "beyond+", "beyond-", "pos-unres", "neg-unres", "inconcl")


def ci90(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided 90 % t CI of the mean along the last axis."""
    n = x.shape[-1]
    m = x.mean(-1)
    se = x.std(-1, ddof=1) / math.sqrt(n)
    tc = stats.t.ppf(1 - ALPHA, n - 1)
    return m - tc * se, m + tc * se


def classify(lo: np.ndarray, hi: np.ndarray, bound: float) -> np.ndarray:
    """Exhaustive partition of the 90 % CI [lo, hi] against {-b, 0, +b}."""
    out = np.full(lo.shape, INCONCL)
    out[(lo > -bound) & (hi < bound)] = EQUIV
    out[lo >= bound] = BEYOND_POS
    out[hi <= -bound] = BEYOND_NEG
    out[(lo > 0) & (lo < bound) & (hi >= bound)] = POS_UNRES
    out[(hi < 0) & (hi > -bound) & (lo <= -bound)] = NEG_UNRES
    return out


def class_probs(codes: np.ndarray) -> str:
    return " ".join(f"{n}={np.mean(codes == i):.3f}" for i, n in enumerate(CLASS_NAMES))


def page_l(data: np.ndarray) -> np.ndarray:
    """Page's L for increasing order; blocks on axis -2, levels on axis -1."""
    ranks = stats.rankdata(data, axis=-1)  # mid-ranks for ties
    weights = np.arange(1, data.shape[-1] + 1)
    return (ranks * weights).sum(axis=(-2, -1))


def page_asym_p(data: np.ndarray) -> np.ndarray:
    """One-sided (increasing) asymptotic Page p-value, vectorised over sims."""
    n, m = data.shape[-2], data.shape[-1]
    mean = n * m * (m + 1) ** 2 / 4
    var = n * m**2 * (m + 1) * (m**2 - 1) / 144
    return stats.norm.sf((page_l(data) - mean) / math.sqrt(var))


_PAGE_CRIT: dict[tuple[int, int], float] = {}


def page_perm_crit(n: int, m: int, draws: int = 200_000) -> float:
    """Permutation critical value of Page's L (no ties), one-sided alpha.

    Smallest c with P0(L >= c) <= alpha under within-block exchangeability,
    estimated from `draws` random rank permutations (fixed generator), so
    that power simulations use the registered permutation calibration
    rather than the asymptotic normal approximation.
    """
    key = (n, m)
    if key not in _PAGE_CRIT:
        g = np.random.default_rng(20261023 + 1000 * n + m)
        weights = np.arange(1, m + 1)
        lvals = np.zeros(draws)
        for start in range(0, draws, 20_000):
            k = min(20_000, draws - start)
            ranks = g.random((k, n, m)).argsort(-1).argsort(-1) + 1
            lvals[start : start + k] = (ranks * weights).sum(axis=(-2, -1))
        lv, counts = np.unique(lvals, return_counts=True)
        tail = np.cumsum(counts[::-1])[::-1] / draws  # P(L >= lv)
        _PAGE_CRIT[key] = float(lv[np.argmax(tail <= ALPHA)])
    return _PAGE_CRIT[key]


def page_perm_reject(data: np.ndarray) -> np.ndarray:
    """Vectorised one-sided (increasing) Page rejection at the permutation critical value."""
    return page_l(data) >= page_perm_crit(data.shape[-2], data.shape[-1])


def page_perm_pvalue(data: np.ndarray, n_perm: int, rng) -> float:
    """Pre-registered Page test: within-block permutation of mid-ranks.

    p = (1 + #{L_perm >= L_obs}) / (1 + n_perm), one-sided increasing.
    For a decreasing hypothesis pass `data[:, ::-1]`.
    """
    ranks = stats.rankdata(data, axis=-1)
    n, m = ranks.shape
    weights = np.arange(1, m + 1)
    l_obs = float((ranks * weights).sum())
    keys = rng.random((n_perm, n, m))
    perm = np.take_along_axis(np.broadcast_to(ranks, keys.shape), keys.argsort(-1), -1)
    l_perm = (perm * weights).sum(axis=(-2, -1))
    return float((1 + np.sum(l_perm >= l_obs - 1e-9)) / (1 + n_perm))


def v07_b1_rule(mu, cov, n, sims, rng, quantum=0.00625):
    """E-1 verdict classes and the fixed-sequence E-1 -> E-1b claim."""
    x = rng.multivariate_normal(mu, cov, size=(sims, n))  # (sims, n, 5)
    x = np.round(x / quantum) * quantum  # grid medians are quantised
    lo, hi = ci90(x.mean(-1))
    e1 = classify(lo, hi, B1_BOUND)
    lo_l, hi_l = ci90(np.moveaxis(x, -1, -2))  # (sims, 5)
    e1b = (e1 == EQUIV) & ((lo_l > -B1_BOUND) & (hi_l < B1_BOUND)).all(-1)
    return e1, e1b


def v08_rule(n, effect, guard, q_loss, sims, rng, levels=7, sd_seed=1.0):
    """v0.8 complete rule: Page + magnitude guard, rank-excluded blocks.

    Each seed block is independently rank-excluded with probability
    `q_loss` (no replacement). The run is a protocol failure if fewer
    than 80 % of the N blocks are complete. Units: within-seed SD.
    Returns P(Supported), P(Page alone among valid runs), P(protocol fail).
    """
    trend = np.linspace(0.0, effect, levels)
    y = trend + rng.normal(0, 1, (sims, n, levels)) + rng.normal(0, sd_seed, (sims, n, 1))
    keep = rng.random((sims, n)) >= q_loss
    n_keep = keep.sum(1)
    fail = n_keep < math.ceil(0.8 * n)
    supported = np.zeros(sims, bool)
    page_ok = np.zeros(sims, bool)
    for k in np.unique(n_keep[~fail]):
        idx = np.where((n_keep == k) & ~fail)[0]
        yk = np.stack([y[i][keep[i]] for i in idx])  # (len, k, levels)
        rej = page_perm_reject(yk)
        guard_ok = np.median(yk[..., -1] - yk[..., 0], axis=-1) > guard
        page_ok[idx] = rej
        supported[idx] = rej & guard_ok
    return supported.mean(), page_ok.mean(), fail.mean()


def v09_rule(n, mu0, profile, sd, rho, sims, rng, quantum=1 / 512):
    """v0.9 complete rule on the three axes, correlated onsets.

    D(s, o) = mu_o + u_s + e_so, var(u) = rho sd^2, var(e) = (1-rho) sd^2,
    quantised to 1/512 (difference of two accuracies on a 512 probe).
    """
    if profile == "linear":
        mu = mu0 * np.array([1.0, 0.75, 0.5, 0.25, 0.0])
    else:  # "early": only onset 0 affected
        mu = mu0 * np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    d = (
        mu
        + rng.normal(0, sd * math.sqrt(rho), (sims, n, 1))
        + rng.normal(0, sd * math.sqrt(1 - rho), (sims, n, 5))
    )
    d = np.round(d / quantum) * quantum
    d0 = d[..., 0]
    t = d0.mean(-1) / (d0.std(-1, ddof=1) / math.sqrt(n))
    harm = t > stats.t.ppf(1 - ALPHA, n - 1)  # D-1
    timing = page_perm_reject(d[..., ::-1])  # D-2, decreasing, permutation-calibrated
    lo, hi = ci90(d0)
    mag = classify(lo, hi, B1_BOUND)
    return {
        "D-1": harm.mean(),
        "D-2": timing.mean(),
        "headline D-1&D-2": (harm & timing).mean(),
        "  ...& mag>=SESOI": (harm & timing & (mag == BEYOND_POS)).mean(),
        "old D-1&mean>=.05": (harm & (d0.mean(-1) >= B1_BOUND)).mean(),
        "mag>=SESOI": (mag == BEYOND_POS).mean(),
        "mag negligible": (mag == EQUIV).mean(),
        "mag unresolved": ((mag != BEYOND_POS) & (mag != EQUIV)).mean(),
    }


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

    print(
        f"== B-1 (Me7), 10 seeds x 5 LOCK_AFTER, bound +-{B1_BOUND:.3f} [component-test power only]"
    )
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

    print(f"\n== B-2 (Me3 delta, bits), bound +-{B2_BOUND:.2f} [component-test power only]")
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

    print(
        "\n== Page trend test power (one-sided, alpha=0.05), no pilot data"
        " [component-test power only]"
    )
    page_sims = max(args.sims // 50, 500)
    for levels, label in ((7, "v0.8 LOCK_AFTER grid"), (5, "v0.9 onset grid")):
        for n in (10, 20, 30):
            row = []
            for eff in (0.5, 1.0, 1.5):
                row.append(
                    f"range={eff}sd:{page_power_mc(levels, n, eff, 1.0, page_sims, rng):.2f}"
                )
            print(f"  {label} k={levels} N={n}: " + " ".join(row))

    print(
        "\n== v0.9 decorrelation deficit D(onset=0), SESOI 0.05 accuracy"
        " [component-test power only]"
    )
    sd_d = sd_max_lock
    sd_du = sd_ucl(sd_max_lock, len(seeds))
    for sd_label, sd in (("sd=max per-LOCK Me7 sd", sd_d), ("sd_UCL80", sd_du)):
        for n in (10, 20):
            print(
                f"  [{sd_label}={sd:.4f}] N={n}: "
                f"one-sided t (mu=0.05) power={t_one_sided_power_mc(0.05, sd, n, args.sims, rng):.3f} "
                f"TOST+-0.05 (mu=0) power={tost_power_mc(0.0, sd, n, B1_BOUND, args.sims, rng):.3f}"
            )

    # ---------------------------------------------------------------
    # Complete decision rules
    # ---------------------------------------------------------------
    rs = max(args.sims // 5, 2000)  # sims per scenario
    print("\n== COMPLETE RULE v0.7 E-1 (+ fixed-sequence E-1b), quantised Me7")
    print("   classes from the 90% CI vs +-0.05: " + ", ".join(CLASS_NAMES))
    infl = (sd_ucl(sd1, len(seeds)) / sd1) ** 2
    for label, mu, cv in [
        ("mu=observed per-LOCK, cov=observed", y.mean(0), cov),
        ("mu=0, cov=observed", np.zeros(5), cov),
        ("mu=observed per-LOCK, cov x UCL80 inflation", y.mean(0), cov * infl),
    ]:
        for n in (20, 30):
            e1, e1b = v07_b1_rule(mu, cv, n, rs, rng)
            print(f"  [{label}] N={n}: {class_probs(e1)} | E-1b claim={e1b.mean():.3f}")

    print("\n== COMPLETE RULE v0.7 E-2 (normal model, planning SDs from other configs)")
    for sd in (sd2, sd2u):
        for mu in (0.0, -0.04, 0.04):
            for n in (20, 25, 30):
                x = rng.normal(mu, sd, (rs, n))
                lo, hi = ci90(x)
                codes = classify(lo, hi, B2_BOUND)
                print(f"  [sd={sd:.4f} mu={mu:+.2f}] N={n}: {class_probs(codes)}")

    print("\n== v0.8 Page: range giving 80% component power (k=7, sd_seed=1)")
    for n in (20, 30):
        for eff in (0.6, 0.65, 0.7, 0.75):
            pw = v08_rule(n, eff, -np.inf, 0.0, rs, rng)[1]
            print(f"  N={n} range={eff:.2f}sd: Page power={pw:.3f}")

    print("\n== COMPLETE RULE v0.8: Supported = Page p<.05 AND median endpoint diff > guard")
    print("   guard in within-seed SD units; q = per-block rank-exclusion probability;")
    print("   protocol failure if <80% of blocks complete (no replacement)")
    for n in (20, 30):
        for q in (0.0, 0.05, 0.10):
            for eff in (0.75, 1.0, 1.5):
                row = []
                for g in (0.0, 0.25, 0.5, 1.0):
                    sup, _, fail = v08_rule(n, eff, g, q, rs, rng)
                    row.append(f"g={g}:{sup:.2f}")
                print(
                    f"  N={n} q={q:.2f} range={eff}sd: " + " ".join(row) + f" | P(fail)={fail:.3f}"
                )

    print("\n== COMPLETE RULE v0.9 (three axes; D quantised to 1/512)")
    sd9 = sd_ucl(sd_max_lock, len(seeds))  # planning SD UCL80 = 0.0605
    for n in (20, 30):
        for sd in (sd9,):
            for rho in (0.0, 0.5):
                for profile in ("linear", "early"):
                    for mu0 in (0.0, 0.05, 0.075, 0.10):
                        r = v09_rule(n, mu0, profile, sd, rho, rs, rng)
                        txt = " ".join(f"{k.strip()}={v:.3f}" for k, v in r.items())
                        print(f"  N={n} sd={sd:.4f} rho={rho} {profile:6s} mu0={mu0:.3f}: {txt}")

    print("\n== v0.9 N selection: early profile, mu0=0.05, rho=0 (conservative), sd=0.0605")
    print("   permutation-calibrated Page; select N with estimate - 2 MC SE >= 0.80")
    for n in (30, 31, 32, 33, 34):
        sims = 100_000
        pw = v09_rule(n, 0.05, "early", sd9, 0.0, sims, rng)["headline D-1&D-2"]
        se = math.sqrt(pw * (1 - pw) / sims)
        print(f"  N={n}: P(A+ and C+)={pw:.4f} (MC SE {se:.4f}, lower={pw - 2 * se:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
