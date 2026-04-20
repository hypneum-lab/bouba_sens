"""Task 7.3 Step 6 — CLI smoke test.

Builds tiny synthetic report.pkl files for 3 "worlds", invokes the
compare_mi_estimators CLI via CliRunner, confirms the JSON shape.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from scripts.compare_mi_estimators import app


def _write_report(dir_: Path, rng: np.random.Generator, label_strength: float) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    labels = rng.integers(0, 4, 128)
    pre = rng.normal(size=(128, 8))
    post = pre + rng.normal(scale=0.05, size=(128, 8))
    post[:, 0] += labels * label_strength
    payload = {
        "pre_lesion_codes": pre,
        "post_lesion_codes": post,
        "pre_lesion_labels": labels,
    }
    (dir_ / "report.pkl").write_bytes(pickle.dumps(payload))


def test_cli_emits_decay_ordering(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    runs_root = tmp_path / "runs"
    # Gaussian gets biggest label signal, Sinusoid gets smallest
    for world, strength in [("gaussian", 0.4), ("xor", 0.2), ("sinusoid", 0.1)]:
        src_name = "v02_grid" if world == "gaussian" else f"v02_{world}"
        _write_report(runs_root / src_name / "cell_0", rng, strength)
        _write_report(runs_root / src_name / "cell_1", rng, strength)
    out = tmp_path / "cmp.json"
    result = CliRunner().invoke(
        app,
        [
            "gaussian",
            "xor",
            "sinusoid",
            "--runs-root",
            str(runs_root),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert set(payload["per_world_median"]) == {"gaussian", "xor", "sinusoid"}
    assert set(payload["per_world_median"]["gaussian"]) == {"kraskov", "binning", "mine"}
    # At least kraskov or binning should see the decay in this controlled setup
    # (MINE is noisy at 300 epochs on 128 samples ; don't assert on it)
    orderings = payload["decay_ordering_holds"]
    assert orderings["kraskov"] or orderings["binning"], orderings
