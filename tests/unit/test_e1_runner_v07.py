"""v0.7 pre-filing tasks for the E1 runner (amendment v0.7 section 4.3).

Items 1-4 and 6: explicit seed list, three separate batch sizes, the
primary Ross full-probe output, the secondary full-probe KSG output and a
``--k-steps`` option. Unit tests required by the draft: the three batch
sizes are honoured; ``ksg_full_delta_bits`` equals ``me3_delta`` on the
captured matrices; seed lists reproduce.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from bouba_sens.metrics.mi_migration import me3_delta
from bouba_sens.metrics.mi_ross import mi_ross_discrete_continuous

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "spikes" / "e1_b2_dimensional_mine.py"


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("e1_b2_dimensional_mine", _PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


runner = _load_runner()

_SMALL = {"steps_train": 2, "steps_lesion": 3}


def test_parser_seed_list_batches_and_k_steps() -> None:
    args = runner.build_parser().parse_args(
        [
            "--seed-list",
            "1101,1102, 1105",
            "--train-batch",
            "16",
            "--lesion-batch",
            "16",
            "--probe-n",
            "512",
            "--k-steps",
            "50",
        ]
    )
    assert runner.resolve_seeds(args) == [1101, 1102, 1105]
    assert (args.train_batch, args.lesion_batch, args.probe_n, args.k_steps) == (16, 16, 512, 50)


def test_parser_legacy_defaults_unchanged() -> None:
    args = runner.build_parser().parse_args([])
    assert runner.resolve_seeds(args) == [0, 1, 2, 3, 4]
    assert args.batch_size == 512
    assert (args.train_batch, args.lesion_batch, args.probe_n, args.k_steps) == (None,) * 4


def test_seed_list_and_seed_count_are_exclusive() -> None:
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(["--seeds", "3", "--seed-list", "1,2"])


def test_three_batch_sizes_are_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int, int]] = []
    phase = {"name": "probe"}
    original = runner.build_loop

    def build(world_name: str, seed: int, lock_after: int) -> Any:
        loop = original(world_name, seed, lock_after)
        sample = loop.world.sample
        pretrain, lesion_phase = loop.pretrain, loop.lesion_phase

        def rec_sample(batch_size: int, seed: int) -> Any:
            calls.append((phase["name"], batch_size, seed))
            return sample(batch_size=batch_size, seed=seed)

        def rec_pretrain(*a: Any, **kw: Any) -> Any:
            phase["name"] = "train"
            try:
                return pretrain(*a, **kw)
            finally:
                phase["name"] = "probe"

        def rec_lesion(*a: Any, **kw: Any) -> Any:
            phase["name"] = "lesion"
            try:
                return lesion_phase(*a, **kw)
            finally:
                phase["name"] = "probe"

        loop.world.sample = rec_sample
        loop.pretrain, loop.lesion_phase = rec_pretrain, rec_lesion
        return loop

    monkeypatch.setattr(runner, "build_loop", build)
    pre, post, labels = runner.capture_probe(
        "gaussian", 3, 100, train_batch=4, lesion_batch=8, probe_n=32, **_SMALL
    )
    assert pre.shape[0] == post.shape[0] == labels.shape[0] == 32
    assert {b for p, b, _ in calls if p == "train"} == {4}
    assert ("probe", 32, 3 - 99_999) in calls
    lesion_sizes = {b for p, b, s in calls if p == "lesion" and s != 3 - 99_999}
    assert lesion_sizes == {8}


def test_legacy_batch_size_sets_all_three(monkeypatch: pytest.MonkeyPatch) -> None:
    pre, post, labels = runner.capture_probe("gaussian", 3, 100, batch_size=24, **_SMALL)
    assert pre.shape[0] == post.shape[0] == labels.shape[0] == 24


def test_k_steps_reaches_the_lesion_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[float] = []

    def schedule(step: int, *, snr_init: float, snr_floor: float, k: float) -> float:
        seen.append(k)
        return 0.0

    monkeypatch.setattr(runner, "m2_snr_schedule", schedule)
    runner.capture_probe("gaussian", 3, 100, batch_size=16, k_steps=2, **_SMALL)
    assert seen and set(seen) == {2}
    seen.clear()
    runner.capture_probe("gaussian", 3, 100, batch_size=16, **_SMALL)
    assert set(seen) == {_SMALL["steps_lesion"]}  # legacy: k = steps_lesion


def test_seed_list_path_pins_deterministic_kernels() -> None:
    prev = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        runner.configure_determinism(runner.build_parser().parse_args([]))
        assert not torch.are_deterministic_algorithms_enabled()
        args = runner.build_parser().parse_args(["--seed-list", "1101"])
        runner.configure_determinism(args)
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(prev)


def test_seed_lists_reproduce_independently_of_list_order(
    deterministic_torch: None,
) -> None:
    # Linux LAPACK lstsq can differ by ~1 ulp between calls;
    # the fixture forces deterministic kernels (see conftest).
    kw = {"train_batch": 4, "lesion_batch": 4, "probe_n": 32, "k_steps": 2, **_SMALL}
    alone = runner.capture_probe("gaussian", 7, 100, rng_seed=7, **kw)
    runner.capture_probe("gaussian", 5, 100, rng_seed=5, **kw)  # consume global RNG
    torch.rand(1000)
    after = runner.capture_probe("gaussian", 7, 100, rng_seed=7, **kw)
    for a, b in zip(alone, after, strict=True):
        np.testing.assert_array_equal(a, b)


def test_full_probe_outputs_match_reference_functions() -> None:
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(4), 32)
    pre = rng.normal(size=(128, 20))
    post = pre + 0.5 * labels[:, None] * rng.normal(size=(1, 20))
    out = runner.full_probe_b2(pre, post, labels)
    ross = mi_ross_discrete_continuous(post, labels, k=3) - mi_ross_discrete_continuous(
        pre, labels, k=3
    )
    ksg = me3_delta(torch.from_numpy(pre), torch.from_numpy(post), torch.from_numpy(labels))
    assert out["ross_full_delta_bits"] == ross
    assert out["ksg_full_delta_bits"] == ksg
    assert out["ross_full_post_bits"] - out["ross_full_pre_bits"] == ross
