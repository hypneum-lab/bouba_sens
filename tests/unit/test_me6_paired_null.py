"""ADR-0022 — scripts/me6_paired_null.py on synthetic per-item outcomes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import aggregate_grid  # type: ignore[import-not-found]  # noqa: E402
from me6_paired_null import (  # type: ignore[import-not-found]  # noqa: E402
    B3_ME6_THRESHOLD,
    CELL_RE,
    MODALITIES,
    PAIR_NAMES,
    Trio,
    analyse,
    discover_trios,
    main,
    me6_from_successes,
    pair_differences,
    paired_null,
)

SNRS = ("floor", "minus10", "plus10")


def _symmetric_successes(rng: np.random.Generator, *, cells: int, n: int, p: float) -> np.ndarray:
    """H0 data: every entry iid Bernoulli(p), shape (cells, 5, 5, n)."""
    return (rng.random((cells, 5, 5, n)) < p).astype(np.int8)


def _asymmetric_successes(rng: np.random.Generator, *, cells: int, n: int) -> np.ndarray:
    """audio-queried / vision-lesioned always right, the mirror entry always wrong."""
    s = _symmetric_successes(rng, cells=cells, n=n, p=0.3)
    s[:, 0, 1, :] = 1
    s[:, 1, 0, :] = 0
    return s


def _write_grid(root: Path, successes: np.ndarray, *, with_me1: bool = False) -> list[str]:
    """Lay `successes[k]` out as run_grid.sh cell directories; returns trio labels."""
    labels: list[str] = []
    k = 0
    for seed in range(5):
        for timing in ("T1", "T2"):
            for snr in SNRS:
                if k >= successes.shape[0]:
                    return labels
                for col_j, lesioned in enumerate(MODALITIES):
                    cell = root / f"seed{seed}_{timing}_{lesioned}_{snr}"
                    cell.mkdir(parents=True)
                    body = {
                        query: [int(v) for v in successes[k, row_i, col_j]]
                        for row_i, query in enumerate(MODALITIES)
                    }
                    (cell / "per_query_successes.json").write_text(json.dumps(body))
                    if with_me1:
                        me1 = {q: float(np.mean(v)) for q, v in body.items()}
                        (cell / "per_query_me1.json").write_text(json.dumps(me1))
                        (cell / "eval_report.json").write_text(json.dumps({"me1": me1["audio"]}))
                labels.append(f"seed{seed}_{timing}_{snr}")
                k += 1
    return labels


def test_cell_regex_and_modalities_match_aggregate_grid() -> None:
    assert CELL_RE.pattern == aggregate_grid._CELL_RE.pattern
    assert MODALITIES == aggregate_grid._MODALITIES
    assert len(PAIR_NAMES) == 10


def test_me6_from_successes_matches_definition() -> None:
    s = np.zeros((5, 5, 8), dtype=np.int8)
    s[0, 3, :6] = 1  # P[0,3] = 0.75 ; P[3,0] = 0
    s[2, 2, :] = 1  # diagonal must not count
    assert me6_from_successes(s) == pytest.approx(0.75)
    d = pair_differences(s)
    assert d.shape == (10, 8)
    assert set(np.unique(d)) <= {-1, 0, 1}
    assert me6_from_successes(np.ones((5, 5, 8), dtype=np.int8)) == 0.0


def test_observed_me6_matches_aggregate_grid(tmp_path: Path) -> None:
    """Same matrix convention and value as `aggregate_grid._compute_me6`."""
    rng = np.random.default_rng(11)
    successes = _asymmetric_successes(rng, cells=3, n=64)
    _write_grid(tmp_path, successes, with_me1=True)
    records = list(aggregate_grid._iter_cell_records(tmp_path))
    reference = sorted(aggregate_grid._compute_me6(records))
    trios, incomplete = discover_trios(tmp_path)
    assert not incomplete
    assert len(trios) == 3
    ours = sorted(float(me6_from_successes(t.successes)) for t in trios)
    np.testing.assert_allclose(ours, reference, atol=1e-6)
    # rows = query, cols = lesioned, as in aggregate_grid
    assert trios[0].successes[0, 1].mean() == 1.0 and trios[0].successes[1, 0].mean() == 0.0


def test_discovery_skips_incomplete_trios_and_ignores_other_dirs(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    _write_grid(tmp_path, _symmetric_successes(rng, cells=2, n=16, p=0.5))
    (tmp_path / "seed0_T1_force_floor" / "per_query_successes.json").unlink()
    (tmp_path / "phase1_seed0").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    trios, incomplete = discover_trios(tmp_path)
    assert [t.label for t in trios] == ["seed0_T1_minus10"]
    assert incomplete == [
        {
            "cell": "seed0_T1_floor",
            "present": ["audio", "gravity", "tactile", "vision"],
            "missing_lesioned": ["force"],
        }
    ]


def test_discovery_rejects_inconsistent_item_counts(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    _write_grid(tmp_path, _symmetric_successes(rng, cells=1, n=8, p=0.5))
    path = tmp_path / "seed0_T1_audio_floor" / "per_query_successes.json"
    body = json.loads(path.read_text())
    body["vision"] = body["vision"][:-1]
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="item counts differ"):
        discover_trios(tmp_path)


def test_paired_null_is_deterministic_and_keeps_discordant_structure() -> None:
    rng = np.random.default_rng(2)
    d = pair_differences(_symmetric_successes(rng, cells=4, n=32, p=0.4))
    a_cells, a_med = paired_null(d, replicates=300, seed=25102026)
    b_cells, b_med = paired_null(d, replicates=300, seed=25102026)
    assert a_cells.shape == (300, 4) and a_med.shape == (300,)
    np.testing.assert_array_equal(a_cells, b_cells)
    np.testing.assert_array_equal(a_med, b_med)
    # Me6 under the null is bounded by the discordant fraction of the worst pair.
    bound = np.abs(d).sum(axis=-1).max(axis=-1) / d.shape[-1]
    assert (a_cells <= bound[None, :] + 1e-12).all()
    # A pair without discordant items contributes exactly zero.
    d0 = np.zeros((2, 10, 16), dtype=np.int8)
    cells0, med0 = paired_null(d0, replicates=50, seed=1)
    assert cells0.max() == 0.0 and med0.max() == 0.0


def test_symmetric_data_gives_uniform_like_p_values() -> None:
    """Under H0 the tail probability of the observed median is not extreme on average."""
    p_values = []
    for data_seed in range(12):
        rng = np.random.default_rng(100 + data_seed)
        s = _symmetric_successes(rng, cells=6, n=64, p=0.3)
        trios = [Trio(str(k), "T1", "floor", s[k]) for k in range(6)]
        out = analyse(trios, replicates=1000, seed=25102026, with_binomial=False)
        p_values.append(out["median"]["frac_ge_observed"])
    assert 0.3 <= float(np.mean(p_values)) <= 0.75, p_values
    assert min(p_values) > 0.0  # never "impossible" under its own null


def test_constructed_asymmetry_gives_small_p_value() -> None:
    rng = np.random.default_rng(7)
    s = _asymmetric_successes(rng, cells=6, n=64)
    trios = [Trio(str(k), "T2", "plus10", s[k]) for k in range(6)]
    out = analyse(trios, replicates=2000, seed=25102026, with_binomial=True)
    med = out["median"]
    assert med["observed"] == 1.0
    assert med["count_ge_observed"] == 0
    assert med["p_value"] == pytest.approx(1 / 2001)
    assert med["null"]["p99"] < med["observed"]
    for cell in out["cells"]:
        assert cell["observed_argmax_pair"] == "audio|vision"
        assert cell["observed_signed_pairs"]["audio|vision"] == 1.0
        assert cell["frac_ge_observed"] == 0.0
    ref = out["binomial_reference"]
    assert ref["cells"] == 6 and ref["n"] == 64
    assert ref["median_tail"]["frac_ge_observed"] == 0.0


def test_cli_writes_json(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    root = tmp_path / "grid"
    labels = _write_grid(root, _symmetric_successes(rng, cells=2, n=64, p=0.25))
    out = tmp_path / "paired.json"
    rc = main(
        [
            "--root",
            str(root),
            "--replicates",
            "200",
            "--seed",
            "5",
            "--out",
            str(out),
            "--quiet",
        ]
    )
    assert rc == 0
    body = json.loads(out.read_text())
    assert body["schema"].startswith("bouba_sens.me6_paired_null")
    assert body["params"]["threshold"] == B3_ME6_THRESHOLD
    assert body["params"]["cells_counted"] == 2
    assert body["params"]["complete"] is False
    assert body["params"]["n_items"] == 64
    assert [c["cell"] for c in body["cells"]] == labels
    assert set(body["median"]) >= {"observed", "null", "frac_ge_observed", "p_value"}
    assert "binomial_reference" in body
    assert main(["--root", str(tmp_path / "empty"), "--out", str(out)]) == 2
