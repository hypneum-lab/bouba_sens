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
