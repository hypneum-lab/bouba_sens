"""Me6 large-n diagnostic amendment — scripts/me6_large_n_diagnostic.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from me6_large_n_diagnostic import (  # type: ignore[import-not-found]  # noqa: E402
    _phi_null,
    block_sums,
    d1,
    d2,
    d3,
    exact_sign_p,
    pair_flip_medians,
    parse_blocks,
    phi,
    reprobe_seed,
    transpose_null_medians,
)
from me6_paired_null import (  # type: ignore[import-not-found]  # noqa: E402
    MODALITIES,
    pair_differences,
)

SNRS = ("floor", "minus10", "plus10")


def _write_grid(root: Path, successes: np.ndarray, *, size: int) -> None:
    """`successes[k, q, m, i]` -> 30 trio cells under the run_grid.sh names."""
    k = 0
    for seed in range(400, 405):
        for timing in ("T1", "T2"):
            for snr in SNRS:
                for j, lesioned in enumerate(MODALITIES):
                    cell = root / f"seed{seed}_{timing}_{lesioned}_{snr}"
                    cell.mkdir(parents=True, exist_ok=True)
                    body = {q: successes[k, i, j].tolist() for i, q in enumerate(MODALITIES)}
                    (cell / f"per_query_successes_n{size}.json").write_text(json.dumps(body))
                k += 1


def test_block_sums() -> None:
    d = np.array([[1, -1, 1, 1, 0]], dtype=np.int8)
    assert block_sums(d, 1).tolist() == [[1, -1, 1, 1, 0]]
    assert block_sums(d, 2).tolist() == [[0, 2, 0]]
    with pytest.raises(ValueError):
        block_sums(d, 0)


def test_phi_is_one_in_expectation_under_iid_flip() -> None:
    rng = np.random.default_rng(0)
    units = rng.integers(-1, 2, size=(4, 10, 40)).astype(np.int8)
    null = _phi_null(units, replicates=6000, seed=1, size=40)
    assert abs(null.mean() - 1.0) < 0.04
    # A fixed effect (all differences of one sign) inflates phi far above 1.
    assert phi(np.ones((4, 10, 40), dtype=np.int8)) == pytest.approx(40.0)


def test_joint_flip_is_a_per_item_transpose() -> None:
    rng = np.random.default_rng(2)
    s = rng.integers(0, 2, size=(1, 5, 5, 8)).astype(np.int8)
    d = pair_differences(s)
    t = s.copy()
    t[..., 3] = s[..., 3].swapaxes(-1, -2)
    flipped = d.copy()
    flipped[..., 3] *= -1
    assert np.array_equal(pair_differences(t), flipped)


def test_transpose_flip_reduces_to_pair_flip_for_disjoint_pairs() -> None:
    # Every item is discordant in exactly one pair: the shared sign then acts
    # on one pair only, so the joint and the independent flips coincide in law.
    n_trios, n_items = 6, 200
    rng = np.random.default_rng(3)
    units = np.zeros((n_trios, 10, n_items), dtype=np.int8)
    which = rng.integers(0, 10, size=(n_trios, n_items))
    for k in range(n_trios):
        units[k, which[k], np.arange(n_items)] = rng.choice([-1, 1], size=n_items)
    _, joint = transpose_null_medians(units, n_items=n_items, replicates=4000, seed=4)
    _, indep = pair_flip_medians(units, n_items=n_items, replicates=4000, seed=5)
    assert abs(joint.mean() - indep.mean()) < 0.004
    assert np.quantile(joint, 0.9) == pytest.approx(np.quantile(indep, 0.9), abs=0.01)


def test_exact_sign_p_enumerates_64_patterns() -> None:
    assert exact_sign_p(np.full(6, 0.1)) == pytest.approx(2 / 64)
    assert exact_sign_p(np.zeros(6)) == 1.0
    p = exact_sign_p(np.array([0.3, -0.1, 0.2, 0.05, -0.4, 0.15]))
    assert (p * 64) == pytest.approx(round(p * 64))


def test_reprobe_seed_namespace() -> None:
    assert reprobe_seed(400, 0) == 5_106_000
    assert reprobe_seed(404, 5) == 5_106_045
    with pytest.raises(ValueError):
        reprobe_seed(400, 6)


def test_parse_blocks() -> None:
    assert parse_blocks("64:1,4096:16") == {64: 1, 4096: 16}


def test_end_to_end_iid_versus_fixed_effect(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    n = 256
    iid = (rng.random((30, 5, 5, n)) < 0.25).astype(np.int8)
    _write_grid(tmp_path / "iid", iid, size=n)
    fixed = iid.copy()
    # A real per-trio asymmetry of random sign on audio | vision (cell effect C).
    for k in range(30):
        a, b = (0, 1) if k % 2 else (1, 0)
        fixed[k, a, b] = (rng.random(n) < 0.45).astype(np.int8)
    _write_grid(tmp_path / "fixed", fixed, size=n)

    r_iid = d1(tmp_path / "iid", blocks={n: 1}, replicates=400, split_size=n)
    r_fix = d1(tmp_path / "fixed", blocks={n: 1}, replicates=400, split_size=n)
    assert r_iid["split_half"]["r_all"]["p_value"] > 0.01
    assert r_fix["split_half"]["r_all"]["p_value"] < 0.01
    assert r_fix["phi"][str(n)]["observed"] > r_iid["phi"][str(n)]["observed"]

    assert d2(tmp_path / "iid", blocks={n: 1}, replicates=400)["rows"][str(n)]["p_value"] > 0.01
    assert d2(tmp_path / "fixed", blocks={n: 1}, replicates=400)["rows"][str(n)]["p_value"] < 0.01

    # Random-sign effect: D3 sees no consistent direction on audio | vision.
    out = d3(tmp_path / "fixed", size=n, replicates=400)
    assert out["per_pair"]["audio|vision"]["p_value"] > 0.01
    assert out["within_seed"]["400"]["patterns"] == 64
