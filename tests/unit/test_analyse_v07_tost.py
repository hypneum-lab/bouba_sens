"""Confirmatory TOST analysis of amendment v0.7 (sections 6 and 9)."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import analyse_v07_tost as an  # type: ignore[import-not-found]  # noqa: E402

LABELS = {
    "equivalent",
    "beyond_bound_registered",
    "beyond_bound_reverse",
    "effect_detected_size_unresolved",
    "reverse_effect_detected_size_unresolved",
    "inconclusive",
}


def test_classify_matches_section_9_table() -> None:
    d = 0.05
    assert an.classify(-0.04, 0.04, d) == "equivalent"
    assert an.classify(0.01, 0.04, d) == "equivalent"  # non-zero but inside
    assert an.classify(0.05, 0.09, d) == "beyond_bound_registered"
    assert an.classify(-0.09, -0.05, d) == "beyond_bound_reverse"
    assert an.classify(0.01, 0.06, d) == "effect_detected_size_unresolved"
    assert an.classify(-0.06, -0.01, d) == "reverse_effect_detected_size_unresolved"
    assert an.classify(-0.06, 0.01, d) == "inconclusive"
    assert an.classify(-0.01, 0.05, d) == "inconclusive"


def test_classify_is_exhaustive_and_exclusive_on_a_grid() -> None:
    d = 0.05
    grid = np.round(np.arange(-0.12, 0.1201, 0.01), 10)
    for lo, hi in itertools.combinations_with_replacement(grid, 2):
        label = an.classify(float(lo), float(hi), d)
        assert label in LABELS
        hits = [
            -d < lo and hi < d,
            lo >= d,
            hi <= -d,
            0 < lo < d <= hi,
            lo <= -d < hi < 0,
            lo <= 0 <= hi and (lo <= -d or hi >= d),
        ]
        assert sum(hits) == 1, (lo, hi, hits)


def test_tost_pvalues_match_scipy_and_ci90() -> None:
    rng = np.random.default_rng(3)
    for mu in (-0.06, -0.02, 0.0, 0.03):
        x = rng.normal(mu, 0.05, size=30)
        r = an.tost(x, 0.05)
        p_lo = stats.ttest_1samp(x, -0.05, alternative="greater").pvalue
        p_hi = stats.ttest_1samp(x, 0.05, alternative="less").pvalue
        assert r["p_lower"] == pytest.approx(p_lo, rel=1e-12)
        assert r["p_upper"] == pytest.approx(p_hi, rel=1e-12)
        lo, hi = r["ci90"]
        assert (max(p_lo, p_hi) < 0.05) == (lo > -0.05 and hi < 0.05)
        assert (r["label"] == "equivalent") == (max(p_lo, p_hi) < 0.05)
        lo95, hi95 = r["ci95"]
        assert lo95 < lo < hi < hi95


def test_wilcoxon_tost_and_bootstrap_are_reported() -> None:
    x = np.random.default_rng(4).normal(0.0, 0.02, size=40)
    r = an.tost(x, 0.05)
    assert r["wilcoxon"]["equivalent"] is True
    assert r["bootstrap"]["label"] == "equivalent"
    again = an.tost(x, 0.05)
    assert again["bootstrap"]["ci90"] == r["bootstrap"]["ci90"]  # fixed seed


def _e1_table(lock_means: dict[int, float], nolock: float, n: int, sd: float) -> list[dict]:
    rng = np.random.default_rng(5)
    rows = []
    for s in range(n):
        rows.append(
            {
                "seed": 1001 + s,
                "locks": {str(k): float(m + rng.normal(0, sd)) for k, m in lock_means.items()},
                "nolock": float(nolock + rng.normal(0, sd)),
            }
        )
    return rows


def test_e1_fixed_sequence_all_pass() -> None:
    rows = _e1_table({50: 0, 75: 0, 100: 0, 125: 0, 150: 0}, 0.0, 95, 0.01)
    out = an.analyse_e1(rows)
    assert out["E-1"]["label"] == "equivalent"
    assert out["E-1b"]["tested"] and out["E-1b"]["equivalent"]
    assert out["E-1c"]["tested"] and out["E-1c"]["label"] == "equivalent"


def test_e1_fixed_sequence_stops_after_failed_iut() -> None:
    rows = _e1_table({50: 0.04, 75: 0, 100: 0, 125: 0, 150: -0.04}, 0.0, 95, 0.01)
    out = an.analyse_e1(rows)
    assert out["E-1"]["label"] == "equivalent"  # aggregate cancels
    assert out["E-1b"]["tested"] and not out["E-1b"]["equivalent"]
    assert out["E-1c"] == {"tested": False}


def test_e1_requires_all_five_locks() -> None:
    rows = _e1_table({50: 0, 75: 0, 100: 0, 125: 0}, 0.0, 10, 0.01)
    with pytest.raises(ValueError, match="locks"):
        an.analyse_e1(rows)


def test_e2_uses_half_threshold_bound() -> None:
    vals = [{"seed": 1001 + i, "ross_full_delta_bits": v} for i, v in enumerate([0.01] * 5)]
    vals += [{"seed": 2000, "ross_full_delta_bits": -0.01}]
    out = an.analyse_e2(vals)
    assert out["bound"] == 0.05
    assert out["n"] == 6


def test_duplicate_seeds_rejected() -> None:
    vals = [{"seed": 1, "ross_full_delta_bits": 0.0}] * 2
    with pytest.raises(ValueError, match="duplicate"):
        an.analyse_e2(vals)


# --- confirmatory entry point (Codex review of b6fda3f) -------------------

ROSTER = list(range(1001, 1096))


def _e2_rows(seeds, gumbel=0.0, hard=0.0, sd=0.01, readout="gumbel_k8", seed=0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    return [
        {
            "seed": s,
            "primary_readout": readout,
            "gumbel_draws": 8,
            "switch_pre": False,
            "switch_post": i % 10 == 0,
            "ross_full_delta_bits": float(gumbel + rng.normal(0, sd)),
            "ross_full_delta_bits_hard": float(hard + rng.normal(0, sd)),
        }
        for i, s in enumerate(seeds)
    ]


def _e1_rows(seeds):  # type: ignore[no-untyped-def]
    return [
        {"seed": s, "locks": {k: 0.0 + 0.001 * (i % 3) for k in an.LOCKS}, "nolock": 0.0}
        for i, s in enumerate(seeds)
    ]


def test_confirmatory_accepts_full_roster() -> None:
    out = an.analyse_confirmatory({"e1": _e1_rows(ROSTER), "e2": _e2_rows(ROSTER)})
    assert out["B-2"]["E-2"]["n"] == 95
    assert out["B-1"]["E-1"]["n"] == 95
    assert out["roster"]["substitutions"] == []


def test_confirmatory_rejects_partial_roster() -> None:
    with pytest.raises(ValueError, match="roster"):
        an.analyse_confirmatory({"e2": _e2_rows(ROSTER[:4])})
    with pytest.raises(ValueError, match="roster"):
        an.analyse_confirmatory({"e2": _e2_rows(ROSTER[:-1])})


def test_confirmatory_rejects_pilot_and_unlisted_seeds() -> None:
    seeds = [*ROSTER[:-1], 1101]  # a B-2 pilot seed
    with pytest.raises(ValueError, match="roster"):
        an.analyse_confirmatory({"e2": _e2_rows(seeds)})


def test_confirmatory_rejects_hard_primary_or_missing_readout() -> None:
    with pytest.raises(ValueError, match="gumbel_k8"):
        an.analyse_confirmatory({"e2": _e2_rows(ROSTER, readout="hard")})
    with pytest.raises(ValueError, match="gumbel_k8"):
        an.analyse_confirmatory({"e2": _e2_rows(ROSTER, readout="gumbel")})
    rows = _e2_rows(ROSTER)
    del rows[3]["ross_full_delta_bits_hard"]
    with pytest.raises(ValueError, match="hard"):
        an.analyse_confirmatory({"e2": rows})


def test_reserve_substitutions_are_documented_and_in_order() -> None:
    seeds = [s for s in ROSTER if s not in (1007, 1050)] + [1096, 1097]
    subs = [
        {"replaced": 1007, "by": 1096, "reason": "crash"},
        {"replaced": 1050, "by": 1097, "reason": "NaN metric"},
    ]
    out = an.analyse_confirmatory({"e2": _e2_rows(seeds), "substitutions": subs})
    assert out["B-2"]["E-2"]["n"] == 95
    assert out["roster"]["substitutions"] == subs
    with pytest.raises(ValueError, match="substitution"):  # undocumented
        an.analyse_confirmatory({"e2": _e2_rows(seeds)})
    bad = [dict(subs[0], by=1097), dict(subs[1], by=1096)]  # reserve out of order
    with pytest.raises(ValueError, match="order"):
        an.analyse_confirmatory({"e2": _e2_rows(seeds), "substitutions": bad})
    noreason = [dict(subs[0], reason=""), subs[1]]
    with pytest.raises(ValueError, match="reason"):
        an.analyse_confirmatory({"e2": _e2_rows(seeds), "substitutions": noreason})


def test_e2h_reported_alongside_without_label_and_disagreement_flagged() -> None:
    rows = _e2_rows(ROSTER, gumbel=0.0, hard=0.08, sd=0.01)
    out = an.analyse_confirmatory({"e2": rows})["B-2"]
    assert out["E-2"]["label"] == "equivalent"
    e2h = out["E-2h"]
    assert "label" not in e2h and e2h["estimator"].startswith("Hodges-Lehmann")
    assert e2h["n"] == 95 and e2h["bound"] == 0.05
    assert e2h["ci90"][0] > 0.05
    assert out["readout_disagreement"]["differs"] is True
    same = an.analyse_confirmatory({"e2": _e2_rows(ROSTER)})["B-2"]
    assert same["readout_disagreement"]["differs"] is False
    assert same["E-2"]["label"] == "equivalent"


# --- PI decisions 2026-10-25: robust E-2, switch outcome -----------------


def test_signed_rank_cdf_matches_brute_force() -> None:
    import itertools

    n = 8
    sums = [
        sum(i + 1 for i, b in enumerate(bits) if b) for bits in itertools.product((0, 1), repeat=n)
    ]
    ref = np.cumsum(np.bincount(sums, minlength=n * (n + 1) // 2 + 1)) / 2**n
    np.testing.assert_allclose(an.signed_rank_cdf(n), ref)


def test_hl_ci_matches_its_definition_and_covers() -> None:
    x = np.array([0.3, -0.1, 0.05, 0.2, 0.9, -0.4, 0.15, 0.0, 0.6, 0.25])
    est, lo, hi = an.hl_ci(x, 0.90)
    i, j = np.triu_indices(10)
    w = np.sort((x[i] + x[j]) / 2)
    assert est == pytest.approx(np.median(w))
    cdf = an.signed_rank_cdf(10)
    c = max(c for c in range(1, 56) if cdf[c - 1] <= 0.05)
    assert (lo, hi) == (w[c - 1], w[55 - c])
    rng = np.random.default_rng(3)  # coverage at a symmetric, heavy-tailed law
    hits = 0
    for _ in range(400):
        y = rng.standard_t(2, size=30) + 1.0
        _, a, b = an.hl_ci(y, 0.90)
        hits += a <= 1.0 <= b
    assert hits / 400 >= 0.86


def test_hl_ci_refuses_too_few_values() -> None:
    with pytest.raises(ValueError, match="too small"):
        an.hl_ci([0.1, 0.2, 0.3, 0.4], 0.90)


def test_hl_tost_is_robust_to_one_outlier() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 0.01, 95)
    x[0] = 0.65
    assert an.hl_tost(x, 0.05)["label"] == "equivalent"
    assert an.tost(x, 0.05)["ci90"][1] > an.hl_tost(x, 0.05)["ci90"][1]


def test_e2_primary_is_robust_and_mean_based_is_secondary() -> None:
    out = an.analyse_confirmatory({"e2": _e2_rows(ROSTER)})["B-2"]
    assert out["E-2"]["estimator"].startswith("Hodges-Lehmann")
    assert "label" in out["E-2"] and "label" not in out["E-2 mean (secondary)"]
    sw = out["switch"]
    assert (sw["n"], sw["switch_post"], sw["switch_pre"]) == (95, 10, 0)
    assert sw["ci95_exact"][0] < 10 / 95 < sw["ci95_exact"][1]
    assert sw["mcnemar"]["gained"] == 10 and sw["mcnemar"]["lost"] == 0
    assert sw["mcnemar"]["p_two_sided_exact"] == pytest.approx(2 * 0.5**10)


def test_confirmatory_requires_k8_and_the_switch_fields() -> None:
    rows = _e2_rows(ROSTER)
    rows[5]["gumbel_draws"] = 1
    with pytest.raises(ValueError, match="K = 8"):
        an.analyse_confirmatory({"e2": rows})
    rows = _e2_rows(ROSTER)
    del rows[5]["switch_post"]
    with pytest.raises(ValueError, match="switch_post"):
        an.analyse_confirmatory({"e2": rows})
