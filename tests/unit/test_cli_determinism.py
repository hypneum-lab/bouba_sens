"""Two identical CLI runs must produce byte-identical artefacts (issue #35).

The autouse ``_deterministic_seed`` fixture seeds the *pytest* process, which
hid the bug: ``bouba-sens train`` built its encoders and head from the
unseeded global torch RNG, so each fresh process started from different
weights. Running the CLI in subprocesses escapes the fixture and checks what
a user actually gets from ``scripts/v01_smoke.sh``.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pipeline(root: Path) -> None:
    phase1 = root / "phase1"
    phase2 = root / "phase2_t2"
    for cmd in [
        ["train", "--steps", "3", "--batch-size", "4", "--seed", "0", "--out", str(phase1)],
        [
            "lesion",
            "--ckpt",
            str(phase1),
            "--modality",
            "audio",
            "--steps",
            "3",
            "--seed",
            "10000",
            "--out",
            str(phase2),
        ],
        [
            "eval",
            "--run",
            str(phase2),
            "--metrics",
            "Me1,Me2",
            "--out",
            str(phase2 / "eval_report.json"),
        ],
    ]:
        subprocess.run(
            ["uv", "run", "bouba-sens", *cmd],
            check=True,
            capture_output=True,
            text=True,
        )


def test_cli_pipeline_is_byte_identical_across_processes(tmp_path: Path) -> None:
    run_a, run_b = tmp_path / "a", tmp_path / "b"
    _pipeline(run_a)
    _pipeline(run_b)

    for rel in [
        "phase1/model.pt",
        "phase2_t2/model.pt",
        "phase2_t2/per_query_me1.json",
        "phase2_t2/eval_report.json",
    ]:
        assert _sha256(run_a / rel) == _sha256(run_b / rel), rel
