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


# Frozen reference (ADR-0027 review): outputs of the commands below on
# a1fd6ff (main before ADR-0027), electron-server, torch 2.11.0, numpy
# 2.4.4. The default must reproduce the old code, not only the new one.
# Frozen: a summary of the lesion world (seed 0; float fields as sum
# and absolute sum to 1e-5 relative, labels by sha256 -- world_sha256
# itself hashes float bits and differs across CPUs, as CI showed), the
# sha256 of the discrete per-item / per-query files, and the metrics
# (dyadic values exactly, me3_delta to 1e-6). Not frozen: model.pt and report.pkl,
# whose float weights may differ in the last bits between CPU
# instruction sets (oneDNN / MKL kernels); they were byte-identical
# between a1fd6ff and this branch on electron-server.
LEGACY_REFERENCE = {
    "gaussian": {
        "world0": {
            "z": (98.493091, 6561.857587),
            "audio": (-35.935259, 13034.644301),
            "vision": (-92.660564, 18425.887378),
            "tactile": (-2.280372, 6558.096545),
            "gravity": (-25.260868, 636.623612),
            "force": (-35.181922, 1227.12281),
            "label": "301b51ff65051fc334737e027b8a3171cbc0e778c4799b7a55824f08022d1518",
        },
        "t1": {
            "successes": "37b92dab16ae57d845c9337f1220bc8bb2ed06489674efea3c5da5a7addb8a48",
            "me1_file": "4c1cb6b8e870ba3471f34efc48ded1546043a08601423f9bd5b673a5fe2633b3",
            "eval": (0.4375, 0.15625, 0.265625, -0.202701),
        },
        "t2": {
            "successes": "c88ac3a642568172cfb08f1aebf559da74b7acae8c5ee883e7704cc846f0daf1",
            "me1_file": "064fe5616c74d869b2b85e7927454a14bd437321e1002779c5fa6d709e3fcb27",
            "eval": (0.3125, 0.21875, 0.34375, 0.123757),
        },
        "paired": (0.15625, 0.21875, -0.0625),
    },
    "studyforrest": {
        "world0": {
            "z": (0.0, 0.0),
            "audio": (7864.605731, 21674.534936),
            "vision": (15790.668621, 44767.709536),
            "tactile": (0.0, 0.0),
            "gravity": (0.0, 0.0),
            "force": (0.0, 0.0),
            "label": "c11be3679e61d8132a4645bf53e251886329bff058ff4c7768fc3a4b24cc944a",
        },
        "t1": {
            "successes": "5e2c2c3ea723874832c632f921cf6b7b69d045781d420d59c6535ead5f51f3dd",
            "me1_file": "e9a3adea7dc658a9b558489a448330ac854f84a89fddc673c195f00681608559",
            "eval": (0.3125, 0.390625, 0.1875, -0.144191),
        },
        "t2": {
            "successes": "956520ddb4e508ba0fcd1c96daeab9c5cb179e8bd1a01f5ea74421670fc4484b",
            "me1_file": "7d62f26af6203ac6d40857f415a7114081f286424886c2d729fdd076d941b1f6",
            "eval": (0.1875, 0.3125, 0.484375, -0.069842),
        },
        "paired": (0.390625, 0.3125, 0.078125),
    },
}


def _world_summary(world: object) -> dict[str, object]:
    import hashlib

    s = world.sample(batch_size=256, seed=20_261_030)  # type: ignore[attr-defined]
    out: dict[str, object] = {}
    for name in ("z", "audio", "vision", "tactile", "gravity", "force"):
        x = getattr(s, name).double()
        out[name] = (float(x.sum()), float(x.abs().sum()))
    out["label"] = hashlib.sha256(s.label.contiguous().numpy().tobytes()).hexdigest()
    return out


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("world", ["gaussian", "studyforrest"])
def test_default_reproduces_the_pre_adr0027_reference(
    world: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BOUBA_SENS_STUDYFORREST_DATA", raising=False)
    monkeypatch.delenv("BOUBA_SENS_STUDYFORREST_MOCK_SAMPLES", raising=False)
    ref = LEGACY_REFERENCE[world]
    summary = _world_summary(_build_world(world, 0))
    want = ref["world0"]
    assert summary["label"] == want["label"]  # type: ignore[index]
    for name in ("z", "audio", "vision", "tactile", "gravity", "force"):
        assert summary[name] == pytest.approx(want[name], rel=1e-5, abs=1e-3), name  # type: ignore[index]
    phase1 = tmp_path / "phase1"
    _run("train", "--steps", "3", "--batch-size", "4", "--seed", "7", "--world", world,
         "--no-probe-v2", "--out", str(phase1))  # fmt: skip
    common = ("--modality", "vision", "--steps", "3", "--world", world, "--no-probe-v2")
    _run("lesion", *common, "--seed", "70001", "--ckpt", str(phase1), "--timing", "T2",
         "--out", str(tmp_path / "t2"))  # fmt: skip
    _run("lesion", *common, "--seed", "70002", "--timing", "T1", "--out", str(tmp_path / "t1"))
    for cell in ("t1", "t2"):
        d = tmp_path / cell
        _run("eval", "--run", str(d), "--metrics", "Me1,Me2,Me3",
             "--out", str(d / "eval_report.json"))  # fmt: skip
        assert _sha(d / "per_query_successes.json") == ref[cell]["successes"]  # type: ignore[index]
        assert _sha(d / "per_query_me1.json") == ref[cell]["me1_file"]  # type: ignore[index]
        rep = json.loads((d / "eval_report.json").read_text())
        me1, me1_probe, me2, me3 = ref[cell]["eval"]  # type: ignore[index]
        assert (rep["me1"], rep["me1_probe"], rep["me2"]) == (me1, me1_probe, me2)
        assert rep["me3_delta"] == pytest.approx(me3, abs=1e-6)
        assert "world_sha256" not in rep
    _run("eval", "--t1-ckpt", str(tmp_path / "t1"), "--t2-ckpt", str(tmp_path / "t2"),
         "--out", str(tmp_path / "paired.json"))  # fmt: skip
    paired = json.loads((tmp_path / "paired.json").read_text())
    assert (paired["me1_t1"], paired["me1_t2"], paired["me7"]) == ref["paired"]
    assert "world_seed_mode" not in paired


@pytest.mark.parametrize("world", ["gaussian", "studyforrest"])
def test_cell_mode_with_probe_v2_and_paired_eval(
    world: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BOUBA_SENS_STUDYFORREST_DATA", raising=False)
    phase1 = tmp_path / "phase1"
    _run("train", "--steps", "3", "--batch-size", "4", "--seed", "7", "--world", world,
         "--probe-v2-seed", "75099", "--out", str(phase1))  # fmt: skip
    common = ("--modality", "vision", "--steps", "3", "--world", world, "--probe-v2-seed",
              "75000", "--world-seed-mode", "cell")  # fmt: skip
    _run("lesion", *common, "--seed", "70001", "--ckpt", str(phase1), "--timing", "T2",
         "--out", str(tmp_path / "t2"))  # fmt: skip
    _run("lesion", *common, "--seed", "70002", "--timing", "T1", "--world-seed", "7",
         "--out", str(tmp_path / "t1"))  # fmt: skip
    train_sha = world_sha256(_build_world(world, 7))
    for cell in ("t1", "t2"):
        d = tmp_path / cell
        meta = json.loads((d / "metadata.json").read_text())
        assert meta["world_sha256"] == train_sha
        v2 = json.loads((d / "probe_v2_meta.json").read_text())
        assert v2["probe_seed"] == 75000
        assert (d / "per_query_successes_v2.json").exists()

    # Paired eval: mode propagated, fingerprints recorded; mixed modes refused.
    _run("eval", "--t1-ckpt", str(tmp_path / "t1"), "--t2-ckpt", str(tmp_path / "t2"),
         "--out", str(tmp_path / "paired.json"))  # fmt: skip
    paired = json.loads((tmp_path / "paired.json").read_text())
    assert paired["world_seed_mode"] == "cell"
    assert paired["world_sha256_t1"] == paired["world_sha256_t2"] == train_sha
    _run("lesion", "--modality", "vision", "--steps", "3", "--world", world, "--no-probe-v2",
         "--seed", "70003", "--timing", "T1", "--out", str(tmp_path / "t1_legacy"))  # fmt: skip
    mixed = subprocess.run(
        ["uv", "run", "bouba-sens", "eval", "--t1-ckpt", str(tmp_path / "t1_legacy"),
         "--t2-ckpt", str(tmp_path / "t2"), "--out", str(tmp_path / "mixed.json")],
        capture_output=True, text=True,
    )  # fmt: skip
    assert mixed.returncode != 0
    flat = "".join(ch for ch in mixed.stderr if ch.isalnum() or ch == "-")
    assert "world-seedmodelegacy" in flat  # rich may wrap the message
    assert not (tmp_path / "mixed.json").exists()
