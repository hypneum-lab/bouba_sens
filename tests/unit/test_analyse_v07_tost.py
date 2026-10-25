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
    vals = [{"seed": 1001 + i, "ross_full_delta_bits": v} for i, v in enumerate([0.01] * 3)]
    vals += [{"seed": 2000, "ross_full_delta_bits": -0.01}]
    out = an.analyse_e2(vals)
    assert out["bound"] == 0.05
    assert out["n"] == 4


def test_duplicate_seeds_rejected() -> None:
    vals = [{"seed": 1, "ross_full_delta_bits": 0.0}] * 2
    with pytest.raises(ValueError, match="duplicate"):
        an.analyse_e2(vals)


# --- confirmatory entry point (Codex review of b6fda3f) -------------------

ROSTER = list(range(1001, 1096))


def _e2_rows(seeds, gumbel=0.0, hard=0.0, sd=0.01, readout="gumbel", seed=0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    return [
        {
            "seed": s,
            "primary_readout": readout,
            "ross_full_delta_bits": float(gumbel + rng.normal(0, sd)),
            "ross_full_delta_bits_hard": float(hard + rng.normal(0, sd)),
        }
        for s in seeds
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
    with pytest.raises(ValueError, match="gumbel"):
        an.analyse_confirmatory({"e2": _e2_rows(ROSTER, readout="hard")})
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
    assert "label" not in e2h and "label" not in e2h["bootstrap"]
    assert e2h["n"] == 95 and e2h["bound"] == 0.05
    assert e2h["ci90"][0] > 0.05
    assert out["readout_disagreement"]["differs"] is True
    same = an.analyse_confirmatory({"e2": _e2_rows(ROSTER)})["B-2"]
    assert same["readout_disagreement"]["differs"] is False
    assert same["E-2"]["label"] == "equivalent"
