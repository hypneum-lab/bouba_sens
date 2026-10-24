"""v0.9 planning SD (PI decision 2026-10-24): chi-square UCL80, df = 4."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
from scipy import stats

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "power_prereg_2026_10.py"


def _load():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("power_prereg_2026_10", _PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


power = _load()


def test_v09_planning_multiplier_is_1_558() -> None:
    mult = power.sd_ucl(1.0, 5)
    assert mult == pytest.approx(math.sqrt(4 / stats.chi2.ppf(0.20, 4)), rel=1e-12)
    assert round(mult, 3) == 1.558
    assert power.sd_ucl(0.03, 5) == pytest.approx(0.03 * mult, rel=1e-12)


def test_df_4_is_the_largest_multiplier_over_the_satterthwaite_range() -> None:
    mults = [math.sqrt(df / stats.chi2.ppf(0.20, df)) for df in range(4, 17)]
    assert max(mults) == mults[0]


def test_v08_protocol_failure_threshold_is_70_percent() -> None:
    import numpy as np

    assert power.V08_COMPLETE_MIN == 0.70
    # 14 of 20 complete blocks pass at 70 % (13 fail); with q = 0.3 the
    # failure rate under 70 % must be well below the 80 % rule.
    _, _, f70 = power.v08_rule(20, 1.0, 0.0, 0.3, 4000, np.random.default_rng(0))
    _, _, f80 = power.v08_rule(20, 1.0, 0.0, 0.3, 4000, np.random.default_rng(0), threshold=0.8)
    exact70 = stats.binom.cdf(13, 20, 0.7)
    exact80 = stats.binom.cdf(15, 20, 0.7)
    assert f70 == pytest.approx(exact70, abs=0.03)
    assert f80 == pytest.approx(exact80, abs=0.03)
    assert f70 < f80


SAFEGUARD_TOLERANCE = 0.01  # stated in v0.8 section 7


def test_selection_counterexample_is_controlled_by_the_safeguard() -> None:
    """Codex round-2 counterexample: N = 40, 7 iid N(0,1) levels, no trend,
    guard 0, outcome-dependent exclusion keeping blocks whose endpoint
    difference exceeds its null 14.9th percentile (~85 % kept)."""
    import numpy as np

    sims = 4000
    out = power.v08_rule_selection(
        40, 0.0, 0.0, 0.149, sims, np.random.default_rng(1), exclusion="outcome"
    )
    se = math.sqrt(0.05 * 0.95 / sims)
    assert out["complete_only"] > 0.15  # the problem is reproduced
    assert out["safeguarded"] <= 0.05 + SAFEGUARD_TOLERANCE + 3 * se
    assert 0.80 < out["kept_fraction"] < 0.90
    ind = power.v08_rule_selection(
        40, 0.0, 0.0, 0.149, sims, np.random.default_rng(2), exclusion="independent"
    )
    assert ind["complete_only"] <= 0.05 + 3 * se
    assert ind["safeguarded"] <= ind["complete_only"] + 1e-12
    lev = power.v08_rule_selection(
        40, 0.0, 0.0, 0.149, sims, np.random.default_rng(3), exclusion="level"
    )
    assert lev["safeguarded"] <= 0.05 + 3 * se
    assert 0.80 < lev["kept_fraction"] < 0.90  # same per-seed incompleteness q
