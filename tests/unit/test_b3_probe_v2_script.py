"""ADR-0023 — scripts/b3_probe_v2.py: block rule, positive control, verdict."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from b3_probe_v2 import (  # type: ignore[import-not-found]  # noqa: E402
    acf_mean,
    block_from_acf,
    block_rule,
    positive_control,
    report,
    verdict,
)
from me6_paired_null import (  # type: ignore[import-not-found]  # noqa: E402
    MODALITIES,
    discover_trios,
)

SNRS = ("floor", "minus10", "plus10")


def _write_grid(root: Path, successes: np.ndarray, *, name: str) -> None:
    k = 0
    for seed in range(5):
        for timing in ("T1", "T2"):
            for snr in SNRS:
                if k >= successes.shape[0]:
                    return
                for col_j, lesioned in enumerate(MODALITIES):
                    cell = root / f"seed{seed}_{timing}_{lesioned}_{snr}"
                    cell.mkdir(parents=True, exist_ok=True)
                    body = {
                        q: [int(v) for v in successes[k, i, col_j]]
                        for i, q in enumerate(MODALITIES)
                    }
                    (cell / name).write_text(json.dumps(body))
                k += 1


def _write_phase1(root: Path, seed: int, *, vision_acc: float, baseline: float) -> None:
    d = root / f"phase1_seed{seed}"
    d.mkdir(parents=True)
    mods = {
        m: {
            "accuracy": vision_acc if m == "vision" else baseline,
            "baseline": baseline,
            "n_codes_test": 1,
        }
        for m in MODALITIES
    }
    (d / "probe_v2_meta.json").write_text(json.dumps({"modalities": mods}))
    (d / "init_probe_v2_meta.json").write_text(json.dumps({"modalities": mods}))


def test_block_from_acf_rule() -> None:
    assert block_from_acf({1: 0.05, 2: 0.0}, n_items=256) == (1, 1)
    assert block_from_acf({1: 0.4, 2: 0.2, 3: 0.09}, n_items=256) == (4, 3)
    assert block_from_acf({1: 0.4, 2: 0.08}, n_items=256) == (2, 2)
    assert block_from_acf({k: 0.3 for k in range(1, 65)}, n_items=256) == (64, None)
    acf = {k: 0.5 for k in range(1, 9)} | {9: 0.05}
    assert block_from_acf(acf, n_items=256) == (16, 9)


def test_acf_mean_iid_vs_ar1() -> None:
    rng = np.random.default_rng(0)
    iid = rng.normal(size=(200, 256))
    acf_iid, kept = acf_mean(iid, [1, 2])
    assert kept == 200 and abs(acf_iid[1]) < 0.02
    ar = np.zeros((200, 256))
    for t in range(1, 256):
        ar[:, t] = 0.8 * ar[:, t - 1] + rng.normal(size=200)
    acf_ar, _ = acf_mean(ar, [1, 4])
    assert acf_ar[1] > 0.7 and acf_ar[4] > 0.3
    assert acf_mean(np.zeros((3, 10)), [1]) == ({}, 0)


def test_block_rule_iid_world_is_one_and_ar_world_is_longer(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    s = (rng.random((6, 5, 5, 64)) < 0.5).astype(np.int8)
    _write_grid(tmp_path / "iid", s, name="per_query_successes_v2.json")
    out = block_rule(tmp_path / "iid", world="gaussian")
    assert out["block"] == 1 and out["trios"] == 6
    # Runs of 8 identical outcomes -> strong short-lag dependence.
    runs = np.repeat((rng.random((6, 5, 5, 8)) < 0.5).astype(np.int8), 8, axis=-1)
    _write_grid(tmp_path / "ar", runs, name="per_query_successes_v2.json")
    out = block_rule(tmp_path / "ar", world="mock")
    assert out["block"] >= 8


def test_positive_control_pass_and_fail(tmp_path: Path) -> None:
    for seed in range(5):
        _write_phase1(tmp_path / "ok", seed, vision_acc=0.6, baseline=0.3)
        _write_phase1(tmp_path / "ko", seed, vision_acc=0.35, baseline=0.3)
    ok = positive_control(tmp_path / "ok")
    assert ok["passes"] and ok["mean_margin"] == pytest.approx(0.3) and ok["n_seeds"] == 5
    assert "untrained_reference_mean_margin_per_modality" in ok
    assert not positive_control(tmp_path / "ko")["passes"]


def test_verdict_rule() -> None:
    assert verdict(control_passes=False, complete=True, p_value=1e-4).startswith("inconclusive")
    assert verdict(control_passes=True, complete=False, p_value=1e-4).startswith("inconclusive")
    assert verdict(control_passes=True, complete=True, p_value=0.009) == "asymmetry present"
    assert verdict(control_passes=True, complete=True, p_value=0.05) == "compatible with the null"
    assert verdict(control_passes=True, complete=True, p_value=0.03).startswith("inconclusive")


def test_report_end_to_end_on_asymmetric_grid(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    s = (rng.random((30, 5, 5, 64)) < 0.4).astype(np.int8)
    s[:, 0, 1, :] = 1
    s[:, 1, 0, :] = 0
    _write_grid(tmp_path, s, name="per_query_successes_v2.json")
    for seed in range(5):
        _write_phase1(tmp_path, seed, vision_acc=0.6, baseline=0.3)
    out = report(tmp_path, world="gaussian", block=1, replicates=500, seed=0)
    assert out["b3_v2"]["complete"]
    assert out["b3_v2"]["observed"] == pytest.approx(1.0)
    assert out["b3_v2"]["verdict"] == "asymmetry present"
    assert out["paired_null"]["params"]["successes_file"] == "per_query_successes_v2.json"


def test_discover_trios_reads_the_requested_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    s = (rng.random((2, 5, 5, 16)) < 0.5).astype(np.int8)
    _write_grid(tmp_path, s, name="per_query_successes_v2.json")
    assert discover_trios(tmp_path)[0] == []
    trios, _ = discover_trios(tmp_path, successes_file="per_query_successes_v2.json")
    assert len(trios) == 2
