"""scripts/b3_lesion_interaction.py — interaction residual, exact sign flip, verdict."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from b3_lesion_interaction import (  # type: ignore[import-not-found]  # noqa: E402
    exact_sign_flip,
    interaction_residual,
    report,
    verdict,
)
from me6_paired_null import MODALITIES  # type: ignore[import-not-found]  # noqa: E402

SNRS = ("floor", "minus10", "plus10")


def _write(root: Path, successes: np.ndarray, seeds: list[int]) -> None:
    k = 0
    for seed in seeds:
        for timing in ("T1", "T2"):
            for snr in SNRS:
                for j, lesioned in enumerate(MODALITIES):
                    cell = root / f"seed{seed}_{timing}_{lesioned}_{snr}"
                    cell.mkdir(parents=True)
                    body = {
                        q: [int(v) for v in successes[k, i, j]] for i, q in enumerate(MODALITIES)
                    }
                    (cell / "per_query_successes_v2.json").write_text(json.dumps(body))
                k += 1
        d = root / f"phase1_seed{seed}"
        d.mkdir()
        mods = {m: {"accuracy": 0.6, "baseline": 0.25, "n_codes_test": 3} for m in MODALITIES}
        (d / "probe_v2_meta.json").write_text(json.dumps({"modalities": mods}))


def test_additive_matrix_has_zero_interaction_residual() -> None:
    a = np.array([0.7, 0.6, 0.5, 0.3, 0.2])
    b = np.array([0.0, -0.05, 0.02, 0.01, 0.03])
    p = a[:, None] + b[None, :]
    assert np.allclose(interaction_residual(p), 0.0, atol=1e-12)


def test_interaction_is_detected_by_the_residual() -> None:
    p = np.full((5, 5), 0.5)
    p[0, 1] += 0.2  # lesioning vision helps audio only
    r = interaction_residual(p)
    assert np.abs(r).max() > 0.1


def test_exact_sign_flip_consistent_vs_random() -> None:
    rng = np.random.default_rng(0)
    consistent = np.tile(rng.normal(size=20), (10, 1)) + 0.01 * rng.normal(size=(10, 20))
    out = exact_sign_flip(consistent)
    assert out["patterns"] == 1024
    assert out["p_value"] == pytest.approx(2 / 1024)  # identity and global flip tie
    noise = rng.normal(size=(10, 20))
    assert exact_sign_flip(noise)["p_value"] > 0.05


def test_verdict_rule() -> None:
    assert verdict(learns=False, control_passes=True, complete=True, p_value=0.0).startswith(
        "inconclusive"
    )
    assert verdict(learns=True, control_passes=False, complete=True, p_value=0.0).startswith(
        "inconclusive"
    )
    assert verdict(learns=True, control_passes=True, complete=False, p_value=0.0).startswith(
        "inconclusive"
    )
    assert verdict(learns=True, control_passes=True, complete=True, p_value=0.002) == (
        "lesion x query interaction present"
    )
    assert verdict(learns=True, control_passes=True, complete=True, p_value=0.2) == (
        "compatible with no interaction"
    )
    assert verdict(learns=True, control_passes=True, complete=True, p_value=0.02).startswith(
        "inconclusive"
    )


@pytest.mark.parametrize("effect", [0.0, 0.5])
def test_report_end_to_end(tmp_path: Path, effect: float) -> None:
    rng = np.random.default_rng(1)
    seeds = list(range(10))
    n = 64
    probs = np.full((5, 5), 0.5)
    probs[0, 1] += effect * 0.8  # interaction: audio query, vision lesion
    probs[2, 3] -= effect * 0.8
    s = (rng.random((60, 5, 5, n)) < probs[None, :, :, None]).astype(np.int8)
    _write(tmp_path, s, seeds)
    out = report(tmp_path, world="gaussian", block=1, learns=True, replicates=200)
    assert out["trios"] == 60 and out["complete"]
    if effect:
        assert out["verdict"] == "lesion x query interaction present"
        assert out["secondary_block_permutation"]["p_value"] < 0.05
    else:
        assert out["primary"]["p_value"] > 0.01
