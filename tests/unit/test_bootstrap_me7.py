from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from scripts.bootstrap_me7 import bootstrap_me7_median_ci  # type: ignore[import-not-found]


def test_tight_ci_on_well_separated_sample() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=0.1, scale=0.01, size=75)
    ci = bootstrap_me7_median_ci(sample, n_boot=2000, seed=0)
    assert 0.08 <= ci["ci_low"] <= ci["median"] <= ci["ci_high"] <= 0.12


def test_ci_straddles_zero_on_near_zero_sample() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=0.001, scale=0.02, size=75)
    ci = bootstrap_me7_median_ci(sample, n_boot=2000, seed=0)
    assert ci["ci_low"] <= 0.0 <= ci["ci_high"]


def test_determinism_on_seed() -> None:
    sample = np.linspace(-0.01, 0.02, 75)
    a = bootstrap_me7_median_ci(sample, n_boot=500, seed=7)
    b = bootstrap_me7_median_ci(sample, n_boot=500, seed=7)
    assert a == b
