"""Task 3.8 acceptance tests for the CLI."""

from __future__ import annotations

import json
import pickle
import subprocess
from pathlib import Path


def test_cli_version_still_works() -> None:
    result = subprocess.run(
        ["uv", "run", "bouba-sens", "version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "bouba_sens" in result.stdout


def test_cli_sim_writes_parquet(tmp_path: Path) -> None:
    out = tmp_path / "world.parquet"
    result = subprocess.run(
        [
            "uv",
            "run",
            "bouba-sens",
            "sim",
            "--world",
            "gaussian",
            "--size",
            "16",
            "--seed",
            "0",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert out.exists()
    assert out.stat().st_size > 0
    assert "wrote 16 rows" in result.stdout


def test_cli_train_writes_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    subprocess.run(
        [
            "uv",
            "run",
            "bouba-sens",
            "train",
            "--steps",
            "5",
            "--batch-size",
            "4",
            "--seed",
            "0",
            "--out",
            str(run_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (run_dir / "checkpoint.pkl").exists()
    with (run_dir / "checkpoint.pkl").open("rb") as f:
        ckpt = pickle.load(f)
    # Checkpoint has all expected keys.
    assert ckpt.mux_state
    assert ckpt.nerve_state
    assert ckpt.head_state
    # metadata.json recorded.
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["phase"] == "pretrain"
    assert meta["steps"] == 5


def test_cli_eval_writes_report(tmp_path: Path) -> None:
    train_dir = tmp_path / "phase1"
    lesion_dir = tmp_path / "phase2"
    eval_out = tmp_path / "eval.json"
    # Quick end-to-end: train 5 -> lesion 5 -> eval.
    for cmd in [
        [
            "uv",
            "run",
            "bouba-sens",
            "train",
            "--steps",
            "5",
            "--batch-size",
            "4",
            "--seed",
            "0",
            "--out",
            str(train_dir),
        ],
        [
            "uv",
            "run",
            "bouba-sens",
            "lesion",
            "--ckpt",
            str(train_dir),
            "--modality",
            "audio",
            "--steps",
            "5",
            "--seed",
            "1",
            "--out",
            str(lesion_dir),
        ],
        [
            "uv",
            "run",
            "bouba-sens",
            "eval",
            "--run",
            str(lesion_dir),
            "--metrics",
            "Me1,Me2",
            "--out",
            str(eval_out),
        ],
    ]:
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    assert eval_out.exists()
    body = json.loads(eval_out.read_text())
    assert body["me1"] is not None
    assert body["me2"] is not None
