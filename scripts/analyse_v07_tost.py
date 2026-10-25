"""Confirmatory TOST analysis for amendment v0.7 (sections 3, 6 and 9).

Input JSON (one value per confirmatory seed; collected from the grid
aggregates, ``invariants.b1.median_me7`` per (seed, LOCK_AFTER), and from
the E1 runner, ``ross_full_delta_bits`` per seed)::

    {"e1": [{"seed": 1001, "locks": {"50": v, "75": v, "100": v,
             "125": v, "150": v}, "nolock": v}, ...],
     "e2_config": {... the frozen --preset v07-e2 configuration ...},
     "e2": [{"seed": 1001, "primary_readout": "gumbel_k8", "gumbel_draws": 8,
             "ross_full_delta_bits": v, "ross_full_delta_bits_hard": v,
             "switch_pre": false, "switch_post": false,
             "pre_top1_share": v, "pre_between_class_share": v,
             "post_top1_share": v, "post_between_class_share": v,
             "probe_label_counts": {"0": 128, ...}}, ...],
     "substitutions": [{"replaced": 1007, "by": 1156, "reason": "crash"}]}

``analyse_confirmatory`` is the only confirmatory entry point (used by
``main``). It refuses any input that is not exactly the frozen roster
1001-1100 and 1106-1155 (N = 150) after documented reserve substitutions (1156-1159, used in
order, at most 4, each with a reason), and any B-2 record that is not
from the Gumbel primary readout or lacks the hard-readout value (PI
decision 2026-10-25, after the B-2 pilot). E-2h (hard readout,
descriptive) is reported next to E-2 with the same statistics but no
label of its own; a difference between the two readouts is reported
and never changes the E-2 label. ``tost`` stays a generic,
reusable function.

Decision (frozen; PI decisions 2026-10-25, after the pilot): for E-1,
E-1b, E-1c and E-2 the robust TOST, the 90 % distribution-free
Hodges-Lehmann CI inside +-bound (``hl_tost``); the mean-based t-TOST
(two one-sided one-sample t-tests at alpha = 0.05) of every step is
reported as a secondary analysis without a label. Bounds 0.025
(B-1) and 0.05 bit (B-2). Each primary test gets exactly one of the six
section-9 labels from its 90 % CI. B-1 fixed sequence E-1 -> E-1b (IUT
over the five locks) -> E-1c (no-lock). Reported, no decision: 95 % CI,
a 10 000-resample percentile bootstrap TOST and a Wilcoxon signed-rank
TOST.

    uv run python scripts/analyse_v07_tost.py --input v07_values.json --out v07_tost.json
"""

from __future__ import annotations

import argparse
import functools
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

ALPHA = 0.05
B1_BOUND = 0.025
B2_BOUND = 0.05
LOCKS: tuple[str, ...] = ("50", "75", "100", "125", "150")
# N = 150 (PI decision 2026-10-25, 13:32; section 8): 1001-1100 and
# 1106-1155, skipping the B-2 pilot seeds 1101-1105; reserve 1156-1159.
ROSTER: tuple[int, ...] = tuple(range(1001, 1101)) + tuple(range(1106, 1156))
RESERVE: tuple[int, ...] = (1156, 1157, 1158, 1159)
SWITCH_TOP1 = 0.25  # bouba_sens.readout (v0.7 section 4.2)
SWITCH_BETWEEN = 0.10
N_BOOT = 10_000
BOOT_SEED = 20261023


def classify(lo: float, hi: float, bound: float) -> str:
    """Section-9 label of a 90 % CI [lo, hi] against {-bound, 0, +bound}."""
    if -bound < lo and hi < bound:
        return "equivalent"
    if lo >= bound:
        return "beyond_bound_registered"
    if hi <= -bound:
        return "beyond_bound_reverse"
    if 0 < lo < bound <= hi:
        return "effect_detected_size_unresolved"
    if lo <= -bound < hi < 0:
        return "reverse_effect_detected_size_unresolved"
    return "inconclusive"


def _t_ci(x: np.ndarray, level: float) -> tuple[float, float]:
    n = x.size
    se = x.std(ddof=1) / math.sqrt(n)
    q = stats.t.ppf(0.5 + level / 2, n - 1)
    m = float(x.mean())
    return m - q * se, m + q * se


def tost(values: Sequence[float], bound: float, *, n_boot: int = N_BOOT) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 3 or not np.all(np.isfinite(x)):
        raise ValueError("need >= 3 finite per-seed values")
    n = x.size
    se = x.std(ddof=1) / math.sqrt(n)
    p_lower = float(stats.t.sf((x.mean() + bound) / se, n - 1))
    p_upper = float(stats.t.cdf((x.mean() - bound) / se, n - 1))
    lo, hi = _t_ci(x, 1 - 2 * ALPHA)
    rng = np.random.default_rng(BOOT_SEED)
    boot = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    b_lo, b_hi = (float(v) for v in np.percentile(boot, [100 * ALPHA, 100 * (1 - ALPHA)]))
    w_lower = float(stats.wilcoxon(x + bound, alternative="greater").pvalue)
    w_upper = float(stats.wilcoxon(x - bound, alternative="less").pvalue)
    return {
        "n": int(n),
        "bound": bound,
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "ci90": [lo, hi],
        "ci95": list(_t_ci(x, 0.95)),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "label": classify(lo, hi, bound),
        "bootstrap": {"ci90": [b_lo, b_hi], "label": classify(b_lo, b_hi, bound)},
        "wilcoxon": {
            "p_lower": w_lower,
            "p_upper": w_upper,
            "equivalent": bool(max(w_lower, w_upper) < ALPHA),
        },
    }


@functools.lru_cache(maxsize=64)
def signed_rank_cdf(n: int) -> np.ndarray:
    """Exact null CDF of the Wilcoxon signed-rank statistic T+ for n untied values.

    Entry t is P(T+ <= t), t = 0 ... n(n + 1) / 2 (dynamic programming over
    the 2^n sign patterns).
    """
    counts = np.zeros(n * (n + 1) // 2 + 1)
    counts[0] = 1.0
    for i in range(1, n + 1):
        shifted = np.zeros_like(counts)
        shifted[i:] = counts[:-i]
        counts = counts + shifted
    return np.cumsum(counts) / counts.sum()


def hl_ci(values: Sequence[float], level: float) -> tuple[float, float, float]:
    """Hodges-Lehmann estimate and its distribution-free CI (Walsh averages).

    The CI is [W_(c), W_(M - c + 1)] of the M = n(n + 1) / 2 sorted Walsh
    averages (x_i + x_j) / 2, i <= j, with c the largest integer such that
    P(T+ <= c - 1) <= (1 - level) / 2 under the exact signed-rank null;
    its coverage is at least ``level`` for a continuous distribution
    symmetric about its centre (Hollander, Wolfe & Chicken, ch. 3).
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    i, j = np.triu_indices(n)
    walsh = np.sort((x[i] + x[j]) / 2)
    m = walsh.size
    cdf = signed_rank_cdf(n)
    a = (1 - level) / 2
    c = int(np.searchsorted(cdf, a, side="right"))  # P(T+ <= c - 1) <= a
    if c < 1:
        raise ValueError(f"n = {n} is too small for a {level:.0%} signed-rank CI")
    return float(np.median(walsh)), float(walsh[c - 1]), float(walsh[m - c])


def signed_rank_p(values: Sequence[float], theta: float, alternative: str) -> float:
    """Exact signed-rank p-value for H0: centre = theta, dual to ``hl_ci``.

    Frozen tie convention (Codex review r2): a Walsh average equal to
    theta never counts as beyond it. ``greater``: T = #{Walsh > theta},
    p = P(T+ >= T); ``less``: T = #{Walsh >= theta}, p = P(T+ <= T),
    with T+ the untied signed-rank law (``signed_rank_cdf``). Then, for
    any data, tied or not, p <= alpha exactly when the 1 - 2 alpha
    ``hl_ci`` lies strictly beyond theta on that side (lower bound >
    theta, resp. upper bound < theta). Under ties the untied law is an
    approximation (stated in v0.7 section 6); the CI decides.
    """
    x = np.asarray(values, dtype=float)
    i, j = np.triu_indices(x.size)
    w = (x[i] + x[j]) / 2
    cdf = signed_rank_cdf(x.size)
    if alternative == "greater":
        k = int((w > theta).sum())  # P(T+ >= k)
        return 1.0 if k <= 0 else float(1.0 - cdf[k - 1])
    if alternative == "less":
        k = int((w >= theta).sum())  # P(T+ <= k)
        return float(cdf[k])
    raise ValueError(f"unknown alternative {alternative!r}")


def hl_tost(values: Sequence[float], bound: float) -> dict[str, Any]:
    """Robust TOST: section-9 label from the 90 % Hodges-Lehmann CI (PI 2026-10-25)."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 3 or not np.all(np.isfinite(x)):
        raise ValueError("need >= 3 finite per-seed values")
    est, lo, hi = hl_ci(x, 1 - 2 * ALPHA)
    _, lo95, hi95 = hl_ci(x, 0.95)
    i, j = np.triu_indices(x.size)
    walsh = (x[i] + x[j]) / 2
    return {
        "n": int(x.size),
        "bound": bound,
        "estimator": "Hodges-Lehmann pseudo-median",
        "hl": est,
        "median": float(np.median(x)),
        "ci90": [lo, hi],
        "ci95": [lo95, hi95],
        "p_lower": signed_rank_p(x, -bound, "greater"),
        "p_upper": signed_rank_p(x, bound, "less"),
        "walsh_ties": int(walsh.size - np.unique(walsh).size),
        "label": classify(lo, hi, bound),
    }


def switch_analysis(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Registered secondary binary outcome (PI 2026-10-25): no label, no decision.

    Proportion of seeds with a post-lesion switch and its exact
    (Clopper-Pearson) two-sided 95 % CI; paired pre/post comparison by the
    exact McNemar test (two-sided binomial on the discordant seeds).
    """
    post = [bool(r["switch_post"]) for r in rows]
    pre = [bool(r["switch_pre"]) for r in rows]
    n, k = len(post), sum(post)
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    gained = sum(b and not a for a, b in zip(pre, post, strict=True))
    lost = sum(a and not b for a, b in zip(pre, post, strict=True))
    p_mc = float(stats.binomtest(gained, gained + lost, 0.5).pvalue) if gained + lost else 1.0
    return {
        "n": n,
        "switch_post": k,
        "proportion": k / n,
        "ci95_exact": [float(ci.low), float(ci.high)],
        "switch_pre": sum(pre),
        "mcnemar": {"gained": gained, "lost": lost, "p_two_sided_exact": p_mc},
    }


def _unique_seeds(rows: Sequence[dict[str, Any]]) -> None:
    seeds = [r["seed"] for r in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds in input")


def analyse_e1(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """E-1 (mean over the five locks), then E-1b (IUT), then E-1c (no-lock).

    PI decision 2026-10-25 (13:32): each step uses the robust TOST
    (``hl_tost``, 90 % Hodges-Lehmann CI); the mean-based t-TOST of each
    step is reported alongside without a label ("... mean (secondary)").
    """
    _unique_seeds(rows)
    for r in rows:
        if set(r["locks"]) != set(LOCKS):
            raise ValueError(f"seed {r['seed']}: locks must be exactly {LOCKS}")
    per_lock = {k: [float(r["locks"][k]) for r in rows] for k in LOCKS}
    e1_vals = [float(np.mean([r["locks"][k] for k in LOCKS])) for r in rows]
    out: dict[str, Any] = {
        "E-1": hl_tost(e1_vals, B1_BOUND),
        "E-1 mean (secondary)": _unlabelled(tost(e1_vals, B1_BOUND)),
    }
    out["E-1b mean (secondary)"] = {k: _unlabelled(tost(v, B1_BOUND)) for k, v in per_lock.items()}
    out["E-1c mean (secondary)"] = _unlabelled(tost([float(r["nolock"]) for r in rows], B1_BOUND))
    if out["E-1"]["label"] != "equivalent":
        out["E-1b"] = {"tested": False}
        out["E-1c"] = {"tested": False}
        return out
    by_lock = {k: hl_tost(v, B1_BOUND) for k, v in per_lock.items()}
    iut = all(r["label"] == "equivalent" for r in by_lock.values())
    out["E-1b"] = {"tested": True, "equivalent": iut, "per_lock": by_lock}
    if not iut:
        out["E-1c"] = {"tested": False}
        return out
    out["E-1c"] = {"tested": True, **hl_tost([float(r["nolock"]) for r in rows], B1_BOUND)}
    return out


def analyse_e2(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """E-2 primary: robust Hodges-Lehmann TOST (PI decision 2026-10-25)."""
    _unique_seeds(rows)
    return hl_tost([float(r["ross_full_delta_bits"]) for r in rows], B2_BOUND)


def resolve_roster(substitutions: Sequence[dict[str, Any]]) -> list[int]:
    """The 150 analysed seeds after documented reserve substitutions (section 8)."""
    if len(substitutions) > len(RESERVE):
        raise ValueError("more than 4 substitutions: the run halts (section 8)")
    seeds = list(ROSTER)
    replaced: set[int] = set()
    for i, sub in enumerate(substitutions):
        old, new = int(sub["replaced"]), int(sub["by"])
        if not str(sub.get("reason", "")).strip():
            raise ValueError(f"substitution of {old} has no reason")
        if new != RESERVE[i]:
            raise ValueError(
                f"reserve seeds must be used in order: expected {RESERVE[i]}, got {new}"
            )
        if old not in ROSTER or old in replaced:
            raise ValueError(f"substitution replaces {old}, not an unreplaced roster seed")
        replaced.add(old)
        seeds[seeds.index(old)] = new
    return seeds


def _check_roster(rows: Sequence[dict[str, Any]], expected: Sequence[int], what: str) -> None:
    _unique_seeds(rows)
    got = sorted(int(r["seed"]) for r in rows)
    if got != sorted(expected):
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        raise ValueError(
            f"{what}: seeds do not match the frozen roster after substitutions "
            f"(missing {missing[:5]}{'...' if len(missing) > 5 else ''}, "
            f"unexpected {extra[:5]}{'...' if len(extra) > 5 else ''}; "
            "an undocumented reserve seed needs a substitution entry)"
        )


def _check_e2_provenance(rows: Sequence[dict[str, Any]]) -> None:
    for r in rows:
        if r.get("primary_readout") != "gumbel_k8":
            raise ValueError(f"seed {r['seed']}: primary_readout must be 'gumbel_k8' (section 4.2)")
        for key in ("switch_pre", "switch_post"):
            if not isinstance(r.get(key), bool):
                raise ValueError(f"seed {r['seed']}: {key} must be a JSON boolean")
        _check_switch_rule(r)
        if r.get("probe_label_counts") != {"0": 128, "1": 128, "2": 128, "3": 128}:
            raise ValueError(f"seed {r['seed']}: probe_label_counts is not the balanced probe")
        if r.get("gumbel_draws") != 8:
            raise ValueError(f"seed {r['seed']}: the primary readout averages K = 8 Gumbel draws")
        for key in (
            "ross_full_delta_bits",
            "ross_full_delta_bits_hard",
            "switch_pre",
            "switch_post",
        ):
            if key not in r:
                raise ValueError(f"seed {r['seed']}: missing {key} (hard readout E-2h)")


def _check_switch_rule(r: dict[str, Any]) -> None:
    """Recompute the frozen switch rule from the recorded structure statistics."""
    for phase in ("pre", "post"):
        try:
            top1 = float(r[f"{phase}_top1_share"])
            between = float(r[f"{phase}_between_class_share"])
        except KeyError as exc:
            raise ValueError(f"seed {r['seed']}: missing {exc.args[0]}") from exc
        if not (0.0 <= top1 <= 1.0 and 0.0 <= between <= 1.0):  # also rejects NaN
            raise ValueError(f"seed {r['seed']}: {phase} structure shares must be in [0, 1]")
        if (top1 >= SWITCH_TOP1 and between >= SWITCH_BETWEEN) != r[f"switch_{phase}"]:
            raise ValueError(f"seed {r['seed']}: switch_{phase} disagrees with the frozen rule")


def _unlabelled(res: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in res.items() if k != "label"}
    out["bootstrap"] = {k: v for k, v in res["bootstrap"].items() if k != "label"}
    out["wilcoxon"] = {k: v for k, v in res["wilcoxon"].items() if k != "equivalent"}
    return out


def analyse_e2_with_secondary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """E-2 (Gumbel K-draw readout, robust TOST, decides) and reported analyses.

    E-2 mean: the earlier mean-based t-TOST on the same values (reported,
    no label). E-2h: hard readout, robust TOST (reported, no label).
    Switch: the binary outcome (``switch_analysis``).
    """
    e2 = analyse_e2(rows)
    mean_based = tost([float(r["ross_full_delta_bits"]) for r in rows], B2_BOUND)
    h = hl_tost([float(r["ross_full_delta_bits_hard"]) for r in rows], B2_BOUND)
    differs = classify(*h["ci90"], B2_BOUND) != e2["label"]
    return {
        "E-2": e2,
        "E-2 mean (secondary)": _unlabelled(mean_based),
        "E-2h": {k: v for k, v in h.items() if k != "label"},
        "switch": switch_analysis(rows),
        "readout_disagreement": {
            "differs": differs,
            "note": (
                "descriptive: the hard-readout 90 % CI falls in a different section-9 "
                "category than E-2; the E-2 label is unchanged"
                if differs
                else "descriptive: both readouts fall in the same section-9 category"
            ),
        },
    }


# The frozen B-2 configuration (v0.7 section 4.2; ``--preset v07-e2`` of the
# E1 runner, summary fields of its output JSON).
E2_CONFIG: dict[str, Any] = {
    "world": "studyforrest",
    "lock_after": 100,
    "steps_train": 200,
    "steps_lesion": 100,
    "snr_init_db": 0.0,
    "snr_floor_db": -20.0,
    "train_batch": 16,
    "lesion_batch": 16,
    "probe_n": 512,
    "k_steps": 50,
    "probe_demod": "both",
    "probe_scheme": "balanced",
    "gumbel_draws": 8,
    "preset": "v07-e2",
    "primary_readout": "gumbel_k8",
}


def _check_e2_config(cfg: Any) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("e2_config (the E1 runner summary) is required")
    bad = {k: cfg.get(k) for k, v in E2_CONFIG.items() if cfg.get(k) != v}
    if bad:
        raise ValueError(f"e2_config differs from the frozen configuration: {bad}")


def analyse_confirmatory(data: dict[str, Any]) -> dict[str, Any]:
    """Validated confirmatory analysis: frozen roster, Gumbel-readout provenance."""
    subs = list(data.get("substitutions", []))
    expected = resolve_roster(subs)
    result: dict[str, Any] = {"roster": {"n": len(expected), "substitutions": subs}}
    if "e1" in data:
        _check_roster(data["e1"], expected, "e1")
        result["B-1"] = analyse_e1(data["e1"])
    if "e2" in data:
        _check_roster(data["e2"], expected, "e2")
        _check_e2_config(data.get("e2_config"))
        _check_e2_provenance(data["e2"])
        result["B-2"] = analyse_e2_with_secondary(data["e2"])
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.input.read_text())
    result = analyse_confirmatory(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
