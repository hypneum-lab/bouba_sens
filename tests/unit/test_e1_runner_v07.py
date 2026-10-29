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


# --- v0.7 PI decision 2026-10-24: both probe readouts --------------------


def test_parser_probe_demod_default_and_choices() -> None:
    assert runner.build_parser().parse_args([]).probe_demod == "soft"
    args = runner.build_parser().parse_args(["--probe-demod", "both"])
    assert args.probe_demod == "both"
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(["--probe-demod", "hard"])


def test_both_readouts_share_one_run_and_keep_legacy_gumbel(
    deterministic_torch: None,
) -> None:
    kw = {"train_batch": 4, "lesion_batch": 4, "probe_n": 32, "k_steps": 2, **_SMALL}
    legacy = runner.capture_probe("gaussian", 7, 100, rng_seed=7, **kw)
    labels, readouts = runner.capture_probe_readouts("gaussian", 7, 100, rng_seed=7, **kw)
    assert set(readouts) == {"hard", "gumbel"}
    np.testing.assert_array_equal(labels, legacy[2])
    np.testing.assert_array_equal(readouts["gumbel"][0], legacy[0])
    np.testing.assert_array_equal(readouts["gumbel"][1], legacy[1])
    hard_pre, hard_post = readouts["hard"]
    assert hard_pre.shape == hard_post.shape == legacy[0].shape
    assert not np.array_equal(hard_pre, readouts["gumbel"][0])


def test_hard_readout_is_deterministic_on_a_fixed_model() -> None:
    loop = runner.build_loop("gaussian", 3, 100)
    probe = loop.world.sample(batch_size=16, seed=1)
    a = runner._full_fused(loop, probe, hard=True)
    torch.rand(100)
    b = runner._full_fused(loop, probe, hard=True)
    assert torch.equal(a, b)


def test_b2_fields_gumbel_is_primary_and_hard_is_suffixed() -> None:
    # PI decision 2026-10-25 (after the pilot): Gumbel primary, hard descriptive.
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(4), 32)
    hard = (rng.normal(size=(128, 10)), rng.normal(size=(128, 10)))
    gumbel = (rng.normal(size=(128, 10)), rng.normal(size=(128, 10)))
    out = runner.b2_readout_fields({"hard": hard, "gumbel": gumbel}, labels)
    ref_h = runner.full_probe_b2(*hard, labels)
    ref_g = runner.full_probe_b2(*gumbel, labels)
    assert out["primary_readout"] == "gumbel"
    for key, val in ref_g.items():
        assert out[key] == val
        assert out[f"{key}_hard"] == ref_h[key]
    legacy = runner.b2_readout_fields({"gumbel": gumbel}, labels)
    assert legacy == {**ref_g, "primary_readout": "gumbel"}


# --- v0.7 PI decisions 2026-10-24: confirmatory presets ------------------

_V07 = {
    "lock_after": 100,
    "steps_train": 200,
    "steps_lesion": 100,
    "train_batch": 16,
    "lesion_batch": 16,
    "probe_n": 512,
    "k_steps": 50,
    "probe_demod": "both",
    "gumbel_draws": 8,
}


@pytest.mark.parametrize(
    ("preset", "world", "scheme"),
    [("v07-e2", "studyforrest", "balanced"), ("v07-e2b", "ccfilm", "contiguous")],
)
def test_preset_sets_the_frozen_configuration(preset: str, world: str, scheme: str) -> None:
    argv = ["--preset", preset, "--seed-list", "1101"]
    args = runner.build_parser().parse_args(argv)
    runner.apply_preset(args, argv)
    assert args.world == world
    assert args.probe_scheme == scheme
    for key, val in _V07.items():
        assert getattr(args, key) == val, key


def test_preset_accepts_matching_and_refuses_conflicting_options() -> None:
    for argv, bad_key in [
        (["--preset", "v07-e2", "--train-batch", "16"], None),
        (["--preset", "v07-e2", "--probe-n", "256"], "probe_n"),
        (["--preset", "v07-e2", "--world", "ccfilm"], "world"),
    ]:
        args = runner.build_parser().parse_args(argv)
        if bad_key is None:
            runner.apply_preset(args, argv)
        else:
            with pytest.raises(SystemExit, match=bad_key):
                runner.apply_preset(args, argv)


def test_no_preset_keeps_legacy_defaults() -> None:
    args = runner.build_parser().parse_args([])
    runner.apply_preset(args, [])
    assert args.preset is None
    assert args.world == "gaussian"
    assert (args.train_batch, args.lesion_batch, args.probe_n, args.k_steps) == (None,) * 4
    assert args.probe_demod == "soft"
    assert args.probe_scheme == "contiguous"


def test_preset_world_check_refuses_the_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOUBA_SENS_STUDYFORREST_DATA", raising=False)
    with pytest.raises(SystemExit, match="StudyforrestRealWorld"):
        runner.check_world("studyforrest", seed=1101)
    runner.check_world("gaussian", seed=0)  # no requirement outside the presets' worlds


# --- Codex review of b6fda3f ----------------------------------------------


def test_legacy_path_skips_amendment_estimators() -> None:
    rng = np.random.default_rng(0)
    labels = np.zeros(512, dtype=int)
    labels[0] = 1  # class counts 511 / 1: Ross would refuse
    pre, post = rng.normal(size=(512, 8)), rng.normal(size=(512, 8))
    legacy = runner.build_parser().parse_args([])
    assert runner.amendment_b2_fields(legacy, {"gumbel": (pre, post)}, labels) == {}
    both = runner.build_parser().parse_args(["--probe-demod", "both"])
    rows = np.repeat(np.arange(4), 32)
    out = runner.amendment_b2_fields(
        both, {"hard": (pre[:128], post[:128]), "gumbel": (pre[:128], post[:128])}, rows
    )
    assert out["primary_readout"] == "gumbel" and "ross_full_delta_bits_hard" in out


def test_preset_refuses_explicit_option_equal_to_the_parser_default() -> None:
    argv = ["--preset", "v07-e2", "--probe-demod", "soft"]
    args = runner.build_parser().parse_args(argv)
    with pytest.raises(SystemExit, match="probe_demod"):
        runner.apply_preset(args, argv)
    argv = ["--preset", "v07-e2", "--world", "gaussian"]
    with pytest.raises(SystemExit, match="world"):
        runner.apply_preset(runner.build_parser().parse_args(argv), argv)
    argv = ["--preset", "v07-e2", "--steps-lesion", "100"]  # equal to the preset: fine
    args = runner.build_parser().parse_args(argv)
    runner.apply_preset(args, argv)
    assert args.steps_lesion == 100


# --- v0.7 PI decision 2026-10-25: class-balanced E-2 probe ----------------


class _IndexedToyWorld:
    """Minimal world with run-structured labels, ``gather`` and ``dataset_labels``."""

    def __init__(self) -> None:
        from bouba_sens.world.base import WorldSample

        self._ws = WorldSample
        self.labels = torch.cat([torch.full((300,), c, dtype=torch.long) for c in (2, 0, 3, 1)])

    @property
    def dataset_labels(self) -> torch.Tensor:
        return self.labels

    def gather(self, idx: torch.Tensor):  # type: ignore[no-untyped-def]
        n = len(idx)
        return self._ws(
            z=torch.zeros(n, 1),
            audio=idx.float()[:, None],
            vision=torch.zeros(n, 1),
            tactile=torch.zeros(n, 1),
            gravity=torch.zeros(n, 1),
            force=torch.zeros(n, 1),
            label=self.labels[idx],
        )

    def sample(self, batch_size: int, seed: int):  # type: ignore[no-untyped-def]
        return self.gather(torch.arange(batch_size))


def test_balanced_probe_scheme_draws_the_class_balanced_rows() -> None:
    from bouba_sens.probe import class_balanced_probe_indices

    world = _IndexedToyWorld()
    probe = runner.draw_probe(world, 512, 1101 - 99_999, "balanced")
    ref = class_balanced_probe_indices(world.labels, seed=1101 - 99_999)
    assert torch.equal(probe.audio[:, 0].long(), ref)
    assert torch.bincount(probe.label, minlength=4).tolist() == [128] * 4
    legacy = runner.draw_probe(world, 512, 5, "contiguous")
    assert torch.equal(legacy.audio[:, 0].long(), torch.arange(512))
    with pytest.raises(ValueError, match="512"):
        runner.draw_probe(world, 256, 0, "balanced")
    with pytest.raises(ValueError, match="scheme"):
        runner.draw_probe(world, 512, 0, "other")


# --- v0.7 PI decision 2026-10-25: K = 8 Gumbel draws ---------------------


def test_k_draws_leave_training_and_legacy_readouts_unchanged(
    deterministic_torch: None,
) -> None:
    kw = {"train_batch": 4, "lesion_batch": 4, "probe_n": 32, "k_steps": 2, **_SMALL}
    base = runner._capture("gaussian", 7, 100, rng_seed=7, readouts=("gumbel", "hard"), **kw)
    with_k = runner._capture(
        "gaussian", 7, 100, rng_seed=7, readouts=("gumbel", "hard"), gumbel_k=3, **kw
    )
    for r in ("gumbel", "hard"):
        np.testing.assert_array_equal(base[1][r][0], with_k[1][r][0])
        np.testing.assert_array_equal(base[1][r][1], with_k[1][r][1])
    pre_k, post_k = with_k[1]["gumbel_k"]
    assert pre_k.shape == (3, *base[1]["gumbel"][0].shape)
    assert not np.array_equal(pre_k[0], pre_k[1])  # independent draws
    again = runner._capture(
        "gaussian", 7, 100, rng_seed=7, readouts=("gumbel", "hard"), gumbel_k=3, **kw
    )
    np.testing.assert_array_equal(again[1]["gumbel_k"][1], post_k)


def test_b2_fields_average_the_k_draws_and_add_the_switch() -> None:
    rng = np.random.default_rng(2)
    labels = np.repeat(np.arange(4), 32)
    draws = [(rng.normal(size=(128, 10)), rng.normal(size=(128, 10))) for _ in range(3)]
    single = (rng.normal(size=(128, 10)), rng.normal(size=(128, 10)))
    pre_k = np.stack([d[0] for d in draws])
    post_k = np.stack([d[1] for d in draws])
    out = runner.b2_readout_fields({"gumbel": single, "gumbel_k": (pre_k, post_k)}, labels)
    per = [runner.full_probe_b2(a, b, labels) for a, b in draws]
    assert out["primary_readout"] == "gumbel_k3" and out["gumbel_draws"] == 3
    for key in ("ross_full_delta_bits", "ross_full_pre_bits", "ksg_full_delta_bits"):
        assert out[key] == pytest.approx(np.mean([d[key] for d in per]))
    assert (
        out["ross_full_delta_bits_gumbel1"]
        == runner.full_probe_b2(*single, labels)["ross_full_delta_bits"]
    )
    assert out["switch_post"] is False and 0 <= out["post_top1_share"] <= 1


def test_v07_confirmatory_seeds_need_the_filing_reference() -> None:
    args = runner.build_parser().parse_args(["--seed-list", "4001,4002"])
    with pytest.raises(SystemExit, match="4001-4154"):
        runner.resolve_seeds(args)
    args = runner.build_parser().parse_args(["--seed-list", "4001", "--osf-filed", "doi"])
    assert runner.resolve_seeds(args) == [4001]
    args = runner.build_parser().parse_args(["--seed-list", "1101,1102"])
    assert runner.resolve_seeds(args) == [1101, 1102]
