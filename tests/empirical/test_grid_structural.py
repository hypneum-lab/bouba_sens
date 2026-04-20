"""Task 4.5 + 5.3 — structural checks on post-Studio aggregate artifacts.

Prefers `reports/v0.2_aggregate.json` (Sprint 5 / v0.2 run); falls back
to `reports/v0.1_aggregate.json` for the Sprint 4 snapshot. Sprint 5
adds `cells_counted >= 1` assertions on B-1/B-2/B-3 which the v0.1
artifact cannot satisfy by design (see ADR-0003).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

_REPORTS = Path(__file__).resolve().parents[2] / "reports"
_V02_PATH = _REPORTS / "v0.2_aggregate.json"
_V01_PATH = _REPORTS / "v0.1_aggregate.json"

if _V02_PATH.exists():
    _AGGREGATE_PATH = _V02_PATH
    _V02_RUN = True
elif _V01_PATH.exists():
    _AGGREGATE_PATH = _V01_PATH
    _V02_RUN = False
else:
    pytest.skip(
        f"no aggregate at {_V02_PATH} or {_V01_PATH}; run scripts/run_grid.sh + "
        "scripts/aggregate_grid.py first",
        allow_module_level=True,
    )


_AGGREGATE = json.loads(_AGGREGATE_PATH.read_text())


def test_grid_shape_30_cells() -> None:
    """5 modalities x 2 timings x 3 SNR = 30 unique cells after seed aggregation."""
    assert len(_AGGREGATE["cells"]) == 30


def test_grid_all_metric_summaries_finite() -> None:
    """Every cell-metric mean/ci_low/ci_high must be finite (no NaN, no Inf)."""
    for cell_name, cell in _AGGREGATE["cells"].items():
        for metric_name, summary in cell.items():
            for stat_key, stat_value in summary.items():
                assert math.isfinite(stat_value), (
                    f"{cell_name}.{metric_name}.{stat_key} not finite (got {stat_value})"
                )


def test_grid_invariants_packed_correctly() -> None:
    inv = _AGGREGATE["invariants"]
    for key in ("b1", "b2", "b3"):
        assert key in inv, f"invariant {key} missing"
        assert isinstance(inv[key]["passes"], bool)
        assert "threshold" in inv[key]
        assert "cells_counted" in inv[key]


def test_grid_thresholds_match_spec() -> None:
    thr = _AGGREGATE["thresholds"]
    assert thr["b1_me7"] == 0.05
    assert thr["b2_me3_delta"] == 0.10
    assert thr["b3_me6"] == 0.02


@pytest.mark.skipif(not _V02_RUN, reason="Sprint 5 coverage assertion (v0.2 only)")
def test_v02_invariants_have_data() -> None:
    """Sprint 5 / Task 5.3: every invariant must have at least one cell."""
    inv = _AGGREGATE["invariants"]
    for key in ("b1", "b2", "b3"):
        assert inv[key]["cells_counted"] >= 1, (
            f"invariant {key} has no cells in v0.2 aggregate — coverage regression from Sprint 5"
        )
