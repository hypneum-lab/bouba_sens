"""ADR-0023 — B-3 probe v2: linear read-out on frozen per-modality representations."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from track_p.multiplexer import GammaThetaMultiplexer

from bouba_sens.encoders import (
    AudioEncoder,
    ForceEncoder,
    GravityEncoder,
    TactileEncoder,
    VisionEncoder,
)
from bouba_sens.head import IntegrationHead
from bouba_sens.linear_probe import fit_linear_probe
from bouba_sens.loop import PROBE_V2_TEST_SIZE, AdaptationLoop
from bouba_sens.nerve import CrossModalNerve
from bouba_sens.sensory import MODALITIES, Modality, SensoryWML
from bouba_sens.world.base import WorldSample
from bouba_sens.world.gaussian import GaussianWorld


def _build_loop(world: object, seed: int = 0) -> AdaptationLoop:
    mux = GammaThetaMultiplexer(seed=seed)
    sensories: dict[Modality, SensoryWML] = {
        "audio": SensoryWML(0, "audio", AudioEncoder(), mux, seed=seed + 1),
        "vision": SensoryWML(1, "vision", VisionEncoder(), mux, seed=seed + 2),
        "tactile": SensoryWML(2, "tactile", TactileEncoder(), mux, seed=seed + 3),
        "gravity": SensoryWML(3, "gravity", GravityEncoder(), mux, seed=seed + 4),
        "force": SensoryWML(4, "force", ForceEncoder(), mux, seed=seed + 5),
    }
    nerve = CrossModalNerve(mux, seed=seed)
    head = IntegrationHead(n_classes=4)
    return AdaptationLoop(world, mux, sensories, nerve, head)  # type: ignore[arg-type]


class _ShuffledLabelWorld:
    """Gaussian inputs with labels drawn independently of them (negative control)."""

    def __init__(self, seed: int = 0) -> None:
        self._inner = GaussianWorld(seed=seed)

    def sample(self, batch_size: int, seed: int) -> WorldSample:
        s = self._inner.sample(batch_size=batch_size, seed=seed)
        gen = torch.Generator().manual_seed(seed + 123_456)
        return replace(s, label=torch.randint(0, 4, (batch_size,), generator=gen))

    def modality_dims(self) -> dict[str, tuple[int, ...]]:
        return self._inner.modality_dims()


# --- linear_probe ----------------------------------------------------------


def _clusters(
    rng: np.random.Generator, n: int, *, informative: bool
) -> tuple[np.ndarray, np.ndarray]:
    y = rng.integers(0, 4, size=n)
    x = rng.normal(size=(n, 16))
    if informative:
        x[np.arange(n), y] += 3.0
    return x, y


def test_linear_probe_positive_control_on_synthetic_clusters() -> None:
    """Informative features: far above the majority baseline."""
    rng = np.random.default_rng(0)
    x_tr, y_tr = _clusters(rng, 512, informative=True)
    x_te, y_te = _clusters(rng, 256, informative=True)
    fit = fit_linear_probe(x_tr, y_tr, x_te, y_te)
    assert fit.accuracy > 0.85
    assert fit.accuracy - fit.baseline > 0.5
    assert fit.cv_accuracy > 0.85
    assert len(fit.successes) == 256 and set(fit.successes) <= {0, 1}
    assert fit.n_train == 512 and fit.n_test == 256


def test_linear_probe_negative_control_stays_near_chance() -> None:
    rng = np.random.default_rng(1)
    x_tr, y_tr = _clusters(rng, 512, informative=False)
    x_te, y_te = _clusters(rng, 256, informative=False)
    fit = fit_linear_probe(x_tr, y_tr, x_te, y_te)
    assert abs(fit.accuracy - 0.25) < 0.1


def test_linear_probe_is_deterministic_and_leaves_global_rng_alone() -> None:
    rng = np.random.default_rng(2)
    x_tr, y_tr = _clusters(rng, 200, informative=True)
    x_te, y_te = _clusters(rng, 64, informative=True)
    np_state = np.random.get_state()[1].copy()
    torch_state = torch.get_rng_state().clone()
    a = fit_linear_probe(x_tr, y_tr, x_te, y_te)
    b = fit_linear_probe(x_tr, y_tr, x_te, y_te)
    assert a == b
    assert np.array_equal(np.random.get_state()[1], np_state)
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_linear_probe_constant_features_predict_majority() -> None:
    y_tr = np.array([0] * 30 + [1] * 10)
    y_te = np.array([0, 1, 0, 0])
    fit = fit_linear_probe(np.zeros((40, 3)), y_tr, np.zeros((4, 3)), y_te)
    assert fit.successes == [1, 0, 1, 1]
    assert fit.accuracy == fit.baseline == 0.75


def test_linear_probe_single_class_pool() -> None:
    fit = fit_linear_probe(np.ones((10, 2)), np.zeros(10), np.ones((3, 2)), np.array([0, 1, 0]))
    assert fit.successes == [1, 0, 1]


# --- AdaptationLoop.probe_v2 -----------------------------------------------


def test_probe_v2_positive_control_on_gaussian_vision() -> None:
    """Vision (256-d, full rank in z) is linearly decodable from its hidden state."""
    loop = _build_loop(GaussianWorld(seed=0), seed=0)
    res = loop.probe_v2(seed=4242)
    vision = res["vision"].fit
    assert vision.n_test == PROBE_V2_TEST_SIZE
    assert vision.accuracy - vision.baseline >= 0.10


def test_probe_v2_negative_control_on_shuffled_labels() -> None:
    loop = _build_loop(_ShuffledLabelWorld(seed=0), seed=0)
    res = loop.probe_v2(seed=99)
    for m in MODALITIES:
        assert res[m].fit.accuracy < 0.40, (m, res[m].fit.accuracy)


def test_probe_v2_shape_and_shared_items() -> None:
    loop = _build_loop(GaussianWorld(seed=1), seed=1)
    res = loop.probe_v2(seed=7, test_size=64, train_size=128)
    for m in MODALITIES:
        s = res[m].fit.successes
        assert len(s) == 64 and set(s) <= {0, 1}
        assert res[m].n_train_dropped == 0
        assert 1 <= res[m].n_codes_test <= 64
    assert loop.query_successes_v2("audio", seed=7, test_size=64, train_size=128) == list(
        res["audio"].fit.successes
    )
    acc = loop.query_accuracy_v2("tactile", seed=7, test_size=64, train_size=128)
    assert acc == pytest.approx(float(np.mean(res["tactile"].fit.successes)))


def test_probe_v2_drops_training_items_shared_with_the_test_set() -> None:
    class _RepeatWorld(GaussianWorld):
        def sample(self, batch_size: int, seed: int) -> WorldSample:
            return super().sample(batch_size=batch_size, seed=0)  # same items every draw

    loop = _build_loop(_RepeatWorld(seed=0), seed=0)
    res = loop.probe_v2(seed=5, test_size=32, train_size=64)
    # The 32 test items are the first 32 pool items: they are removed.
    assert res["audio"].n_train_dropped == 32
    assert res["audio"].fit.n_train == 32


def test_probe_v2_does_not_touch_v1_path_rng_or_state(deterministic_torch: None) -> None:
    """Bit identity of the v1 probe when v2 runs in between."""
    loop = _build_loop(GaussianWorld(seed=3), seed=3)
    torch.manual_seed(11)
    ref = [loop.query_accuracy(m, seed=2024) for m in MODALITIES]
    ref_succ = [loop.query_successes(m, seed=2024) for m in MODALITIES]
    params = {k: v.clone() for k, v in loop.snapshot().head_state.items()}
    sensory = {
        m: {k: v.clone() for k, v in loop.sensories[m].state_dict().items()} for m in MODALITIES
    }

    torch.manual_seed(11)
    rng_before = torch.get_rng_state().clone()
    loop.probe_v2(seed=31, test_size=64, train_size=128)
    assert torch.equal(torch.get_rng_state(), rng_before)
    again = [loop.query_accuracy(m, seed=2024) for m in MODALITIES]
    again_succ = [loop.query_successes(m, seed=2024) for m in MODALITIES]
    assert again == ref
    assert again_succ == ref_succ
    for k, v in loop.snapshot().head_state.items():
        assert torch.equal(v, params[k])
    for m in MODALITIES:
        for k, v in loop.sensories[m].state_dict().items():
            assert torch.equal(v, sensory[m][k])


# --- CLI ---------------------------------------------------------------------


def _run(args: list[str]) -> None:
    subprocess.run(["uv", "run", "bouba-sens", *args], check=True, capture_output=True, text=True)


def test_cli_v1_outputs_bit_identical_with_and_without_probe_v2(tmp_path: Path) -> None:
    import torch as _torch

    outs = {}
    for flag in ("--probe-v2", "--no-probe-v2"):
        root = tmp_path / flag.strip("-")
        train_dir, lesion_dir = root / "phase1", root / "cell"
        _run(
            [
                "train",
                "--steps",
                "3",
                "--batch-size",
                "4",
                "--seed",
                "0",
                flag,
                "--out",
                str(train_dir),
            ]
        )
        _run(
            [
                "lesion", "--ckpt", str(train_dir), "--modality", "vision", "--timing", "T2",
                "--steps", "3", "--seed", "1", flag, "--probe-v2-seed", "5000",
                "--out", str(lesion_dir),
            ]
        )  # fmt: skip
        outs[flag] = (train_dir, lesion_dir)

    (t_on, l_on), (t_off, l_off) = outs["--probe-v2"], outs["--no-probe-v2"]
    for name in ("per_query_successes.json", "per_query_me1.json"):
        assert (l_on / name).read_bytes() == (l_off / name).read_bytes()
    for d_on, d_off in ((t_on, t_off), (l_on, l_off)):
        a = _torch.load(d_on / "model.pt", weights_only=True)
        b = _torch.load(d_off / "model.pt", weights_only=True)
        for part in ("mux", "nerve", "head"):
            for k in a[part]:
                assert _torch.equal(a[part][k], b[part][k]), (d_on, part, k)
    assert not (l_off / "per_query_successes_v2.json").exists()

    succ = json.loads((l_on / "per_query_successes_v2.json").read_text())
    assert set(succ) == set(MODALITIES)
    assert all(len(v) == PROBE_V2_TEST_SIZE for v in succ.values())
    meta = json.loads((l_on / "probe_v2_meta.json").read_text())
    assert meta["probe_seed"] == 5000
    for prefix in ("", "init_"):
        assert (t_on / f"{prefix}per_query_successes_v2.json").exists()
        assert (t_on / f"{prefix}probe_v2_meta.json").exists()
