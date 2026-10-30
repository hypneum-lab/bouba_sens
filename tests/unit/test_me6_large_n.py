"""ADR-0026 — scripts/me6_large_n.py: blocks per size, report, power curve."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from me6_large_n import (  # type: ignore[import-not-found]  # noqa: E402
    blocks,
    main,
    parse_blocks,
    parse_sizes,
    power,
    reading,
    report,
    successes_file_for,
)
from me6_paired_null import MODALITIES  # type: ignore[import-not-found]  # noqa: E402

SNRS = ("floor", "minus10", "plus10")


def _write_grid(root: Path, successes: np.ndarray, *, name: str) -> None:
    """`successes[k, q, m, i]` -> 30 trio cells under the run_grid.sh names."""
    k = 0
    for seed in range(5):
        for timing in ("T1", "T2"):
            for snr in SNRS:
                if k >= successes.shape[0]:
                    return
                for j, lesioned in enumerate(MODALITIES):
                    cell = root / f"seed{seed}_{timing}_{lesioned}_{snr}"
                    cell.mkdir(parents=True, exist_ok=True)
                    body = {
                        q: [int(v) for v in successes[k, i, j]] for i, q in enumerate(MODALITIES)
                    }
                    (cell / name).write_text(json.dumps(body))
                k += 1


def _null_grid(rng: np.random.Generator, n: int, k: int = 30) -> np.ndarray:
    return (rng.random((k, 5, 5, n)) < 0.25).astype(np.int8)


def test_parsers_and_names() -> None:
    assert parse_sizes("4096,64,256,64") == [64, 256, 4096]
    with pytest.raises(ValueError):
        parse_sizes("0")
    assert parse_blocks("64:1, 256:2,legacy:4") == {"64": 1, "256": 2, "legacy": 4}
    with pytest.raises(ValueError):
        parse_blocks("64:0")
    assert successes_file_for("legacy") == "per_query_successes.json"
    assert successes_file_for(1024) == "per_query_successes_n1024.json"


def test_reading_rule_order() -> None:
    assert reading(complete=False, p_value=1e-6).startswith("inconclusive")
    assert reading(complete=True, p_value=0.005) == "excess over the paired null"
    assert reading(complete=True, p_value=0.2) == "compatible with the paired null"
    assert reading(complete=True, p_value=0.03) == "inconclusive"


def test_blocks_per_size(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    for n in (16, 64):
        _write_grid(tmp_path, _null_grid(rng, n, k=6), name=successes_file_for(n))
    # Runs of 8 identical outcomes -> long block on the contiguous world.
    runs = np.repeat((rng.random((6, 5, 5, 8)) < 0.5).astype(np.int8), 8, axis=-1)
    _write_grid(tmp_path, runs, name=successes_file_for("legacy"))
    out = blocks(tmp_path, world="mock", sizes=[16, 64])
    assert set(out["blocks"]) == {"16", "64", "legacy"}
    assert out["blocks"]["legacy"] >= 8
    assert out["blocks"]["64"] == 1
    assert blocks(tmp_path, world="gaussian", sizes=[16, 64])["blocks"]["legacy"] == 1


def test_report_null_and_planted_asymmetry(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    _write_grid(tmp_path, _null_grid(rng, 256), name=successes_file_for(256))
    planted = _null_grid(rng, 256)
    planted[:, 0, 1, :] = (rng.random((30, 256)) < 0.6).astype(np.int8)  # audio | vision
    _write_grid(tmp_path, planted, name=successes_file_for("legacy"))
    out = report(
        tmp_path,
        world="gaussian",
        blocks_by_size={"256": 1, "legacy": 1, "4096": 1},
        replicates=400,
        seed=3,
    )
    null_row, planted_row = out["rows"]["256"], out["rows"]["legacy"]
    assert null_row["complete"] and null_row["trios"] == 30 and null_row["n_items"] == 256
    assert null_row["p_value"] > 0.05
    assert null_row["reading"] == "compatible with the paired null"
    # At n = 256 and p ~ 0.25 the frozen rule fires under the null (ADR-0021).
    assert null_row["rule_0p02_rate_under_null"] == 1.0
    assert null_row["rule_0p02_observed_passes"] is True
    assert planted_row["reading"] == "excess over the paired null"
    assert out["rows"]["4096"]["reading"] == "not run"
    with pytest.raises(ValueError):
        report(tmp_path, world="gaussian", blocks_by_size={"256": 2}, replicates=10)


def test_power_curve_shape() -> None:
    out = power(
        sizes=[64, 4096],
        p=0.25,
        deltas=[0.02, 0.15],
        null_replicates=400,
        alt_replicates=200,
        seed=0,
    )
    small, large = out["rows"]
    assert small["rule_0p02_rate_under_null"] == 1.0
    assert large["rule_0p02_rate_under_null"] < 0.2
    assert large["null_p99"] < small["null_p99"]
    pw = {
        (r["n"], a["delta"]): a["power_paired_test"] for r in out["rows"] for a in r["alternatives"]
    }
    assert pw[(4096, 0.15)] == 1.0
    assert pw[(64, 0.02)] < 0.2
    assert pw[(4096, 0.15)] >= pw[(64, 0.15)]


def test_main_power_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "power.json"
    rc = main(
        [
            "power", "--sizes", "64", "--deltas", "0.1", "--null-replicates", "100",
            "--alt-replicates", "50", "--out", str(out),
        ]
    )  # fmt: skip
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["kind"] == "power" and data["rows"][0]["n"] == 64
