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
