"""Fast checks (R = 200) for scripts/me6_null_calibration.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from me6_null_calibration import (  # type: ignore[import-not-found]  # noqa: E402
    B3_ME6_THRESHOLD,
    DEFAULT_CELLS,
    calibrate,
    main,
    me6_max_abs_off_diag,
    n_min_for,
    read_aggregate,
    simulate_null,
)

SMALL = {"replicates": 200, "fine_replicates": 50, "cells": DEFAULT_CELLS, "seed": 7}


def test_me6_matches_reference_definition() -> None:
    """max |M[i,j] - M[j,i]| over off-diagonal pairs, diagonal ignored."""
    perf = np.full((5, 5), 0.25)
    perf[0, 3] = 0.75  # audio queried, gravity lesioned
    perf[3, 0] = 0.25
    perf[2, 2] = 0.0  # diagonal must not count
    assert me6_max_abs_off_diag(perf) == 0.5
    assert me6_max_abs_off_diag(np.full((5, 5), 0.6)) == 0.0
    batch = np.stack([perf, np.full((5, 5), 0.6)])
    np.testing.assert_allclose(me6_max_abs_off_diag(batch), [0.5, 0.0])


def test_simulate_null_shapes_and_discreteness() -> None:
    per_trio, med = simulate_null(64, 0.25, replicates=200, cells=30, seed=1)
    assert per_trio.shape == (200 * 30,)
    assert med.shape == (200,)
    assert per_trio.min() >= 0.0 and per_trio.max() <= 1.0
    # accuracies are multiples of 1/64 so Me6 per trio is too, medians of 1/128
    np.testing.assert_allclose(per_trio * 64, np.round(per_trio * 64), atol=1e-9)
    np.testing.assert_allclose(med * 128, np.round(med * 128), atol=1e-9)


def test_null_p99_shrinks_with_n_and_threshold_always_passes() -> None:
    result = calibrate(n_grid=[64, 1024], fine_n_grid=[], p_grid=[0.25, 0.5], observed=[], **SMALL)
    rows = {(r["n"], r["p"]): r for r in result["null"]}
    for p in (0.25, 0.5):
        small, large = rows[(64, p)], rows[(1024, p)]
        for key in ("me6_per_trio", "me6_median_over_cells"):
            s, lg = small[key], large[key]
            assert 0.0 <= s["median"] <= s["p95"] <= s["p99"] <= 1.0
            assert lg["p99"] < s["p99"]
        # At n = 64 the 0.02 threshold is cleared by essentially every null grid.
        assert small["me6_median_over_cells"]["frac_above_threshold"] == 1.0
        assert small["me6_per_trio"]["frac_above_threshold"] > 0.99
    # Sanity band for the verdict statistic under H0 at production n.
    assert 0.10 <= rows[(64, 0.5)]["me6_median_over_cells"]["median"] <= 0.22
    assert 0.08 <= rows[(64, 0.25)]["me6_median_over_cells"]["median"] <= 0.20


def test_n_min_is_monotone_in_observed_value() -> None:
    result = calibrate(
        n_grid=[64, 256, 1024],
        fine_n_grid=[128, 512],
        p_grid=[0.25],
        observed=[("low", 0.14, "t"), ("high", 0.45, "t")],
        **SMALL,
    )
    table = {(r["n"], r["p"]): r for r in result["null"]}
    low = n_min_for(0.14, 0.25, table, key="me6_median_over_cells")
    high = n_min_for(0.45, 0.25, table, key="me6_median_over_cells")
    assert high is not None and high == 64  # far above any null at production n
    assert low is None or low >= high
    by_name = {(r["name"], r["p"]): r for r in result["n_min"]}
    assert by_name[("high", 0.25)]["n_min_median_statistic"] == 64
    # the per-trio reference is at least as demanding as the median statistic
    for row in result["n_min"]:
        a, b = row["n_min_median_statistic"], row["n_min_per_trio"]
        if a is not None and b is not None:
            assert b >= a


def test_deterministic_given_seed() -> None:
    kwargs = {"n_grid": [64], "fine_n_grid": [], "p_grid": [0.3], "observed": []}
    a = calibrate(**kwargs, **SMALL)["null"]
    b = calibrate(**kwargs, **SMALL)["null"]
    assert a == b


def test_read_aggregate_extracts_me1_and_k(tmp_path: Path) -> None:
    body = {
        "cells": {
            "t1_audio_floor": {"me1": {"mean": 0.20, "ci_low": 0.1, "ci_high": 0.3}},
            "t2_force_plus10": {"me1": {"mean": 0.30, "ci_low": 0.2, "ci_high": 0.4}},
        },
        "invariants": {"b3": {"median_me6_max_abs": 0.109375, "cells_counted": 30}},
    }
    path = tmp_path / "agg.json"
    path.write_text(json.dumps(body))
    got = read_aggregate(path)
    assert got["found"] is True
    assert got["n_cells"] == 2
    assert (got["min"], got["mean"], got["max"]) == (0.20, 0.25, 0.30)
    assert got["b3_median_me6_max_abs"] == 0.109375
    assert got["b3_cells_counted"] == 30
    missing = read_aggregate(tmp_path / "nope.json")
    assert missing["found"] is False


def test_cli_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "cal.json"
    rc = main(
        [
            "--replicates",
            "200",
            "--fine-replicates",
            "50",
            "--n",
            "64",
            "--fine-n",
            "128",
            "--p",
            "0.25",
            "--observed",
            "gaussian=0.1484",
            "--no-aggregate",
            "--out",
            str(out),
            "--quiet",
        ]
    )
    assert rc == 0
    body = json.loads(out.read_text())
    assert body["schema"].startswith("bouba_sens.me6_null_calibration")
    assert body["params"]["threshold"] == B3_ME6_THRESHOLD
    assert body["params"]["cells"] == DEFAULT_CELLS
    assert body["real_me1"]["found"] is False
    assert {r["n"] for r in body["null"]} == {64, 128}
    assert body["observed"] == [{"name": "gaussian", "value": 0.1484, "source": "command line"}]
