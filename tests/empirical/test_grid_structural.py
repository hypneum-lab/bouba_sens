"""Task 4.5 — structural checks on the post-Studio aggregate artifact.

These tests run ONLY when `reports/v0.1_aggregate.json` exists. Absent
the file (i.e. before Task 4.7 Studio grid run), the module skips at
collection. This keeps CI green on fresh clones while still asserting
the full Sprint 4 grid shape once results are in.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

_AGGREGATE_PATH = Path(__file__).resolve().parents[2] / "reports" / "v0.1_aggregate.json"

if not _AGGREGATE_PATH.exists():
    pytest.skip(
        f"aggregate missing ({_AGGREGATE_PATH}); run scripts/run_grid.sh + "
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
