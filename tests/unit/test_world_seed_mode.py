"""ADR-0027 — world seed of `lesion` / `eval` (`--world-seed-mode`)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch
import typer
from omegaconf import OmegaConf

from bouba_sens.cli import _build_world, config_world_seed, resolve_lesion_world_seed
from bouba_sens.world.base import world_sha256
from bouba_sens.world.ccfilm import CCFilmWorld
from bouba_sens.world.studyforrest import StudyforrestRealWorld, StudyforrestWorld

SYNTHETIC = ("gaussian", "xor", "sinusoid", "studyforrest")


def test_resolve_lesion_world_seed(tmp_path: Path) -> None:
    assert resolve_lesion_world_seed("legacy", world_seed=None, ckpt=None, timing="T1") == 0
    assert resolve_lesion_world_seed("legacy", world_seed=None, ckpt=tmp_path, timing="T2") == 0
    with pytest.raises(typer.BadParameter):
        resolve_lesion_world_seed("legacy", world_seed=3, ckpt=None, timing="T1")
    with pytest.raises(typer.BadParameter):
        resolve_lesion_world_seed("other", world_seed=None, ckpt=None, timing="T1")
    assert resolve_lesion_world_seed("cell", world_seed=3, ckpt=None, timing="T1") == 3
    with pytest.raises(typer.BadParameter):
        resolve_lesion_world_seed("cell", world_seed=None, ckpt=None, timing="T1")
    (tmp_path / "metadata.json").write_text(json.dumps({"phase": "pretrain", "seed": 7}))
    assert resolve_lesion_world_seed("cell", world_seed=None, ckpt=tmp_path, timing="T2") == 7
    assert resolve_lesion_world_seed("cell", world_seed=7, ckpt=tmp_path, timing="T2") == 7
    with pytest.raises(typer.BadParameter):
        resolve_lesion_world_seed("cell", world_seed=8, ckpt=tmp_path, timing="T2")


def test_config_world_seed() -> None:
    old = OmegaConf.create({"seed": 70001, "world": "gaussian"})
    assert config_world_seed(old) == ("legacy", 70001)
    assert config_world_seed(old, override="legacy") == ("legacy", 70001)
    with pytest.raises(typer.BadParameter):
        config_world_seed(old, override="cell")
    new = OmegaConf.create({"seed": 70001, "world_seed_mode": "cell", "world_seed": 7})
    assert config_world_seed(new) == ("cell", 7)
    assert config_world_seed(new, override="legacy") == ("legacy", 70001)


@pytest.mark.parametrize("name", SYNTHETIC)
def test_synthetic_worlds_depend_on_the_world_seed(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BOUBA_SENS_STUDYFORREST_DATA", raising=False)
    a, b = _build_world(name, 0), _build_world(name, 7)
    assert world_sha256(a) == world_sha256(_build_world(name, 0))
    assert world_sha256(a) != world_sha256(b)


def _fake_cache(root: Path, n: int, seed: int, *, five: bool = True) -> Path:
    g = torch.Generator().manual_seed(seed)
    root.mkdir(parents=True)
    tensors = {
        "audio": torch.randn(n, 128, generator=g),
        "vision": torch.randn(n, 16, 16, generator=g),
        "labels": torch.randint(0, 4, (n,), generator=g),
    }
    if five:
        tensors |= {
            "tactile": torch.randn(n, 32, generator=g),
            "gravity": torch.randn(n, 3, generator=g),
            "force": torch.randn(n, 6, generator=g),
        }
    for k, t in tensors.items():
        torch.save(t, root / f"{k}.pt")
    return root


def test_real_worlds_do_not_depend_on_the_world_seed(tmp_path: Path) -> None:
    """Real bridges load their content from the cache; the seed is only stored."""
    from bouba_sens.world.actionsense import ActionSenseWorld

    five = _fake_cache(tmp_path / "five", 300, 1)
    two = _fake_cache(tmp_path / "two", 300, 2, five=False)
    builders = {
        "studyforrest_real5": lambda s: StudyforrestRealWorld(seed=s, data_dir=five),
        "studyforrest_real2": lambda s: StudyforrestWorld(seed=s, data_dir=two),
        "ccfilm": lambda s: CCFilmWorld(seed=s, data_dir=five),
        "actionsense": lambda s: ActionSenseWorld(seed=s, data_dir=five),
    }
    for name, build in builders.items():
        assert world_sha256(build(0)) == world_sha256(build(7)) == world_sha256(build(400)), name


def _run(*args: str) -> None:
    subprocess.run(["uv", "run", "bouba-sens", *args], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("world", ["gaussian", "studyforrest"])
def test_cli_legacy_is_unchanged_and_cell_uses_the_phase1_world(
    world: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BOUBA_SENS_STUDYFORREST_DATA", raising=False)
    phase1 = tmp_path / "phase1"
    _run("train", "--steps", "3", "--batch-size", "4", "--seed", "7", "--world", world,
         "--no-probe-v2", "--out", str(phase1))  # fmt: skip
    common = ("--modality", "vision", "--steps", "3", "--seed", "70001", "--world", world,
              "--no-probe-v2")  # fmt: skip
    cells = {
        "default": ("--ckpt", str(phase1), "--timing", "T2"),
        "legacy": ("--ckpt", str(phase1), "--timing", "T2", "--world-seed-mode", "legacy"),
        "cell": ("--ckpt", str(phase1), "--timing", "T2", "--world-seed-mode", "cell"),
        "cell_t1": ("--timing", "T1", "--world-seed-mode", "cell", "--world-seed", "7"),
    }
    for name, extra in cells.items():
        _run("lesion", *common, *extra, "--out", str(tmp_path / name))
        _run("eval", "--run", str(tmp_path / name), "--metrics", "Me1,Me2,Me3",
             "--out", str(tmp_path / name / "eval_report.json"))  # fmt: skip

    # legacy: explicit flag == default, byte for byte, and nothing recorded.
    for f in ("per_query_successes.json", "per_query_me1.json", "metadata.json",
              "config.yaml", "eval_report.json"):  # fmt: skip
        assert (tmp_path / "default" / f).read_bytes() == (tmp_path / "legacy" / f).read_bytes(), f
    a = torch.load(tmp_path / "default" / "model.pt", weights_only=True)
    b = torch.load(tmp_path / "legacy" / "model.pt", weights_only=True)
    for part in ("mux", "nerve", "head"):
        for k in a[part]:
            assert torch.equal(a[part][k], b[part][k])
    assert "world_seed_mode" not in json.loads((tmp_path / "default" / "metadata.json").read_text())
    assert "world_sha256" not in json.loads((tmp_path / "default" / "eval_report.json").read_text())

    # cell: train, lesion and eval see the same world (seed 7).
    train_sha = world_sha256(_build_world(world, 7))
    assert train_sha != world_sha256(_build_world(world, 0))
    for name in ("cell", "cell_t1"):
        meta = json.loads((tmp_path / name / "metadata.json").read_text())
        cfg = OmegaConf.load(tmp_path / name / "config.yaml")
        rep = json.loads((tmp_path / name / "eval_report.json").read_text())
        assert meta["world_seed_mode"] == "cell" and meta["world_seed"] == 7
        assert cfg["world_seed_mode"] == "cell" and int(cfg["world_seed"]) == 7
        assert meta["world_sha256"] == cfg["world_sha256"] == rep["world_sha256"] == train_sha

    # eval --world-seed-mode legacy on a cell run reproduces the historical rebuild.
    _run("eval", "--run", str(tmp_path / "cell"), "--metrics", "Me1", "--world-seed-mode",
         "legacy", "--out", str(tmp_path / "cell_as_legacy.json"))  # fmt: skip
    assert "world_sha256" not in json.loads((tmp_path / "cell_as_legacy.json").read_text())
