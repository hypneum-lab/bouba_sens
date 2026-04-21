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


def test_train_emits_model_pt_and_config_yaml(tmp_path: Path) -> None:
    """ADR-0007 O-pur Phase A — train must emit model.pt + config.yaml
    alongside the legacy checkpoint.pkl + metadata.json artefacts (purely
    additive; legacy files untouched)."""
    import torch

    run_dir = tmp_path / "train_opur"
    subprocess.run(
        [
            "uv",
            "run",
            "bouba-sens",
            "train",
            "--steps",
            "3",
            "--batch-size",
            "4",
            "--seed",
            "0",
            "--world",
            "gaussian",
            "--out",
            str(run_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    # New canonical artefacts (ADR-0007 Phase A).
    model_pt = run_dir / "model.pt"
    config_yaml = run_dir / "config.yaml"
    assert model_pt.exists(), "train must emit model.pt"
    assert config_yaml.exists(), "train must emit config.yaml"

    state = torch.load(model_pt, weights_only=True)
    for key in ("mux", "nerve", "head", "sensory"):
        assert key in state, f"model.pt missing key '{key}'"

    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.to_container(OmegaConf.load(config_yaml))
    except ImportError:
        import yaml

        cfg = yaml.safe_load(config_yaml.read_text())
    assert isinstance(cfg, dict)
    for key in ("world", "seed", "steps", "batch_size", "phase", "version"):
        assert key in cfg, f"config.yaml missing key '{key}'"
    assert cfg["phase"] == "intact"

    # Phase A is purely additive — legacy files must still exist untouched.
    assert (run_dir / "checkpoint.pkl").exists()
    assert (run_dir / "metadata.json").exists()


def test_lesion_emits_model_pt_and_config_yaml(tmp_path: Path) -> None:
    """ADR-0007 O-pur Phase A — lesion must emit model.pt + config.yaml
    with the post-lesion state and lesion-specific config fields."""
    import torch

    train_dir = tmp_path / "phase1"
    lesion_dir = tmp_path / "phase2"

    subprocess.run(
        [
            "uv",
            "run",
            "bouba-sens",
            "train",
            "--steps",
            "3",
            "--batch-size",
            "4",
            "--seed",
            "0",
            "--out",
            str(train_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "uv",
            "run",
            "bouba-sens",
            "lesion",
            "--ckpt",
            str(train_dir),
            "--modality",
            "audio",
            "--timing",
            "T2",
            "--steps",
            "3",
            "--seed",
            "1",
            "--snr-init",
            "20.0",
            "--snr-floor",
            "-20.0",
            "--k-steps",
            "100",
            "--out",
            str(lesion_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    model_pt = lesion_dir / "model.pt"
    config_yaml = lesion_dir / "config.yaml"
    assert model_pt.exists(), "lesion must emit model.pt"
    assert config_yaml.exists(), "lesion must emit config.yaml"

    state = torch.load(model_pt, weights_only=True)
    for key in ("mux", "nerve", "head", "sensory"):
        assert key in state, f"model.pt missing key '{key}'"

    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.to_container(OmegaConf.load(config_yaml))
    except ImportError:
        import yaml

        cfg = yaml.safe_load(config_yaml.read_text())
    assert isinstance(cfg, dict)
    for key in (
        "modality",
        "timing",
        "snr_init",
        "snr_floor",
        "k_steps",
        "seed",
        "steps",
        "phase",
        "t1_ckpt",
    ):
        assert key in cfg, f"config.yaml missing key '{key}'"
    assert cfg["phase"] == "lesion"
    assert cfg["modality"] == "audio"
    assert cfg["timing"] == "T2"
    # t1_ckpt reference is a relative path string (ADR-0007 Phase A).
    assert isinstance(cfg["t1_ckpt"], str)

    # Legacy artefacts still emitted untouched.
    assert (lesion_dir / "per_query_me1.json").exists()
    assert (lesion_dir / "report.pkl").exists()
    assert (lesion_dir / "metadata.json").exists()


def test_eval_run_emits_me1_probe_additive(tmp_path: Path) -> None:
    """ADR-0007 Phase B Option 4 — ``eval --run`` must emit BOTH the legacy
    ``me1`` (from ``me1_accuracy(report.pkl)`` — primary paper observable,
    fixed by ADR-0004 / ADR-0005) AND the new ``me1_probe`` (canonical
    probe pass ``query_accuracy("audio", seed+777)`` on the reloaded
    model — Phase B reproducibility auxiliary).

    The two observables are distinct by design (adaptation-curve tail
    vs. frozen probe). They coexist additively in ``eval_report.json``
    as of v0.4.0.
    """
    train_dir = tmp_path / "phase1"
    lesion_dir = tmp_path / "phase2"
    eval_out = tmp_path / "eval.json"

    subprocess.run(
        [
            "uv",
            "run",
            "bouba-sens",
            "train",
            "--steps",
            "3",
            "--batch-size",
            "4",
            "--seed",
            "0",
            "--out",
            str(train_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "uv",
            "run",
            "bouba-sens",
            "lesion",
            "--ckpt",
            str(train_dir),
            "--modality",
            "audio",
            "--timing",
            "T2",
            "--steps",
            "3",
            "--seed",
            "1",
            "--out",
            str(lesion_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "uv",
            "run",
            "bouba-sens",
            "eval",
            "--run",
            str(lesion_dir),
            "--metrics",
            "Me1,Me2,Me3",
            "--out",
            str(eval_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    body = json.loads(eval_out.read_text())

    # Legacy key preserved — me1 is still me1_accuracy(report) (primary
    # observable, fixed by ADR-0004 / ADR-0005).
    assert "me1" in body, "eval_report.json must keep legacy me1 field"
    assert body["me1"] is not None
    assert isinstance(body["me1"], float)

    # New additive key — me1_probe is query_accuracy on the reloaded
    # model; distinct observable from me1.
    assert "me1_probe" in body, "eval --run must emit me1_probe (Phase B Option 4)"
    assert body["me1_probe"] is not None
    assert isinstance(body["me1_probe"], float)
    assert 0.0 <= body["me1_probe"] <= 1.0

    # The two observables measure different things (adaptation-curve
    # tail vs. frozen probe); in the general case they differ.
    assert body["me1"] != body["me1_probe"], (
        "me1 (adaptation curve tail) and me1_probe (frozen probe) should "
        "be distinct floats in the general case"
    )

    # Other keys untouched — no regression on v0.3.0 eval_report shape.
    for key in ("me2", "me3_delta", "me6_max_abs", "me7", "me8", "me9"):
        assert key in body, f"eval --run regression: missing key '{key}'"


def test_eval_run_me1_probe_is_null_when_model_pt_absent(tmp_path: Path) -> None:
    """ADR-0007 Phase B Option 4 — when ``model.pt`` / ``config.yaml`` are
    missing (legacy run pre-Phase A), ``eval --run`` must degrade
    gracefully by emitting ``me1_probe: null`` rather than crashing.
    """
    # Build a lesion dir, then delete Phase A artefacts to simulate a
    # pre-Phase-A legacy run.
    train_dir = tmp_path / "phase1"
    lesion_dir = tmp_path / "phase2"
    eval_out = tmp_path / "eval.json"

    subprocess.run(
        [
            "uv",
            "run",
            "bouba-sens",
            "train",
            "--steps",
            "3",
            "--batch-size",
            "4",
            "--seed",
            "0",
            "--out",
            str(train_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "uv",
            "run",
            "bouba-sens",
            "lesion",
            "--ckpt",
            str(train_dir),
            "--modality",
            "audio",
            "--timing",
            "T2",
            "--steps",
            "3",
            "--seed",
            "1",
            "--out",
            str(lesion_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    # Simulate a legacy (pre-Phase A) run: delete model.pt + config.yaml.
    (lesion_dir / "model.pt").unlink()
    (lesion_dir / "config.yaml").unlink()

    subprocess.run(
        [
            "uv",
            "run",
            "bouba-sens",
            "eval",
            "--run",
            str(lesion_dir),
            "--out",
            str(eval_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    body = json.loads(eval_out.read_text())
    # Legacy me1 still present.
    assert body["me1"] is not None
    # me1_probe degrades to null, not a crash.
    assert "me1_probe" in body
    assert body["me1_probe"] is None
