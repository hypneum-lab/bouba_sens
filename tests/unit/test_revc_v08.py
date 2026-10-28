"""v0.8 rev. C (PI decision 2026-10-28): readouts, drift floor, gates, learning gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from bouba_sens import v08_driver as drv
from bouba_sens.builders import build_loop
from bouba_sens.learning_gate import balanced_accuracy, heldout_task_metrics, learning_gate
from bouba_sens.metrics.revc_shift import (
    ENCODER_DIMS,
    LOGIT_DIMS,
    block_failures,
    distance,
    drift_floor,
    evaluate_block_revc,
    normalise,
)
from bouba_sens.metrics.subspace_aggregate import LOCKS
from bouba_sens.sensory import MODALITIES
from bouba_sens.subspace_protocol import SNR_FLOORS, BlockCodes, capture_block
from bouba_sens.world.gaussian import GaussianWorld
from bouba_sens.world.studyforrest import StudyforrestRealWorld

pytestmark = pytest.mark.usefixtures("deterministic_torch")

RUNS = [(0, 400), (1, 1200), (2, 1500), (3, 700), (1, 800), (2, 600)]
_SMALL = {"modalities": ("audio", "vision"), "snr_floors": (-20.0, 10.0), "steps": 2,
          "batch_size": 4}  # fmt: skip


@pytest.fixture
def real_world(tmp_path: Path) -> StudyforrestRealWorld:
    gen = torch.Generator().manual_seed(0)
    y = torch.cat([torch.full((k,), c, dtype=torch.long) for c, k in RUNS])
    n = len(y)
    for name, shape in (("audio", (128,)), ("vision", (16, 16)), ("tactile", (32,)),
                        ("gravity", (3,)), ("force", (6,))):  # fmt: skip
        torch.save(torch.randn(n, *shape, generator=gen), tmp_path / f"{name}.pt")
    torch.save(y, tmp_path / "labels.pt")
    return StudyforrestRealWorld(seed=0, data_dir=tmp_path, stratified_probe=True)


def test_encoder_and_logit_readouts_are_deterministic_and_upstream() -> None:
    world = GaussianWorld(seed=0)
    loop = build_loop(world, 5, lock_after=None, emission="straight-through")
    probe = world.sample(32, seed=1)
    rep = loop.lesion_phase(None, 1, batch_size=4, seed=3, probe_demod="encoder",
                            extra_probe_readouts=(("logits", 0),), probe_sample=probe)  # fmt: skip
    assert rep.pre_lesion_codes is not None and rep.pre_lesion_codes.shape == (32, 640)
    pre_l, post_l = rep.extra_codes["logits_k0"]
    assert pre_l.shape == (32, 320)
    s = loop.sensories["vision"]
    h = s.core(s.input_proj(probe.vision)).detach()
    # batched vs per-call matmuls differ at float32 rounding level only
    assert torch.allclose(rep.post_lesion_codes[:, 128:256], h, atol=1e-5)
    assert torch.allclose(post_l[:, 64:128], s.emit_head_pi(h).detach(), atol=1e-5)
    with pytest.raises(ValueError, match="Gumbel"):
        loop.lesion_phase(None, 1, batch_size=4, seed=3, probe_demod="encoder",
                          probe_gumbel_draws=2)  # fmt: skip


def test_build_loop_passes_the_emission() -> None:
    loop = build_loop(GaussianWorld(seed=0), 1, lock_after=None, emission="straight-through")
    assert {s.emission for s in loop.sensories.values()} == {"straight-through"}
    assert build_loop(GaussianWorld(seed=0), 1, lock_after=None).sensories["audio"].emission == (
        "argmax-bridge"
    )


def test_non_degeneracy_is_checked_per_unnormalised_block() -> None:
    rng = np.random.default_rng(0)
    z = torch.from_numpy(rng.normal(size=(64, 640)))
    assert block_failures(z, ENCODER_DIMS) == []
    z2 = z.clone()
    z2[:, 128:256] = 0.0  # all-zero vision block fails (0 >= 0 is not enough)
    assert block_failures(z2, ENCODER_DIMS) == ["vision"]
    z3 = z.clone()
    z3[:, 0:128] = 5.0 + 1e-6 * z3[:, 0:128]  # tiny variance relative to its norm
    assert block_failures(z3, ENCODER_DIMS) == ["audio"]
    n = normalise(z, ENCODER_DIMS)
    for b in torch.split(n, list(ENCODER_DIMS), dim=1):
        x = b - b.mean(0)
        assert float((x**2).sum()) / 63 == pytest.approx(1.0)
    assert distance(z, z * 3.0, ENCODER_DIMS) == pytest.approx(0.0, abs=1e-12)


def _block(seed: int, lock: int, rng: np.random.Generator, dims: tuple[int, ...],
           *, alt: bool = False, noise: float = 0.3) -> BlockCodes:  # fmt: skip
    base = rng.normal(size=(64, sum(dims)))

    def z() -> torch.Tensor:
        return torch.from_numpy(base + noise * rng.normal(size=base.shape))

    labels = torch.arange(64) % 4
    return BlockCodes(seed=seed, lock=lock, phase2_seed=seed, probe_labels=labels,
                      z_pre=z(), z_control=z(),
                      z_post={(m, f): torch.from_numpy(base + 1.0 * rng.normal(size=base.shape))
                              for m in MODALITIES for f in SNR_FLOORS},
                      z_control_alt=z() if alt else None,
                      control_metrics={"full_bacc": 0.6, "best_query_bacc": 0.4})  # fmt: skip


def test_evaluate_block_and_drift_floor() -> None:
    rng = np.random.default_rng(1)
    blk = _block(2211, 25, rng, ENCODER_DIMS, alt=True)
    res, fails = evaluate_block_revc(blk, ENCODER_DIMS)
    assert res.n_valid == 15 and res.y is not None and sum(fails.values()) == 0
    fl = drift_floor(blk, res, ENCODER_DIMS)
    assert fl["F"] is not None and fl["R"] is not None and fl["R"] > 2  # lesion noise > control
    blk.z_control[:, :128] = 0.0
    res2, fails2 = evaluate_block_revc(blk, ENCODER_DIMS)
    assert res2.y is None and res2.n_valid == 0 and fails2["audio"] >= 1
    assert drift_floor(blk, res2, ENCODER_DIMS) == {"F": None, "R": None}


def test_pilot_quantities_revc_gates_on_synthetic_blocks() -> None:
    rng = np.random.default_rng(2)
    blocks = []
    for s in drv.PILOT_SEEDS_REVC:
        for lk in LOCKS:
            blk = _block(s, lk, rng, ENCODER_DIMS, alt=lk in drv.REVC_FLOOR_LOCKS)
            blk.secondary["logits_k0"] = _block(s, lk, rng, LOGIT_DIMS,
                                                alt=lk in drv.REVC_FLOOR_LOCKS)  # fmt: skip
            blocks.append(blk)
    out = drv.pilot_quantities_revc(blocks)
    g = out["gates_primary"]
    assert out["all_gates_passed"] is (all(g.values()))
    assert g["gate1_learning"] is True and g["gate2_nondegenerate"] is True
    assert g["gate4_drift_floor"] is True and g["gate5_y_nondegenerate"] is True
    assert set(out["primary_encoder"]["drift_floor"]) == {
        "25",
        "175",
        "control_only_contrast",
        "missing_controls",
    }
    # random synthetic codes have no class structure: gate 3 fails, so no N
    assert g["gate3_structure"] is False
    assert out["fileable_as_drafted"] is False and out["g_hat"] == 0.5
    assert out["primary_encoder"]["sigma_hat_df"] == 54
    with pytest.raises(ValueError, match="pilot seeds"):
        drv.pilot_quantities_revc(blocks[:7])


def _m(full: float, best: float) -> dict[str, float]:
    return {"full_bacc": full, "best_query_bacc": best}


def test_learning_gate_rule_is_applied_at_every_setting() -> None:
    good = {lk: {s: _m(0.5, 0.4) for s in range(5)} for lk in (25, 50)}
    assert learning_gate(good, need_over_single=True)["passed"] is True
    one_low = {**good, 75: {**{s: _m(0.5, 0.4) for s in range(4)}, 9: _m(0.29, 0.1)}}
    res = learning_gate(one_low, need_over_single=False)
    assert res["passed"] is False and res["n_settings_failed"] == 1
    no_gain = {25: {s: _m(0.5, 0.48) for s in range(5)}}
    assert learning_gate(no_gain, need_over_single=False)["passed"] is True
    assert learning_gate(no_gain, need_over_single=True)["passed"] is False
    # Codex r8 counterexample: 0.25 at one setting, 0.75 at another must fail
    split = {25: {s: _m(0.25, 0.1) for s in range(5)}, 50: {s: _m(0.75, 0.1) for s in range(5)}}
    assert learning_gate(split, need_over_single=True)["passed"] is False
    assert balanced_accuracy(np.array([0, 0, 0, 0]), np.array([0, 1, 2, 3])) == 0.25


def _pilot_blocks(rng: np.random.Generator, **kw) -> list[BlockCodes]:  # type: ignore[no-untyped-def]
    blocks = []
    for s in drv.PILOT_SEEDS_REVC:
        for lk in LOCKS:
            alt = lk in drv.REVC_FLOOR_LOCKS
            blk = _block(s, lk, rng, ENCODER_DIMS, alt=alt, **kw)
            blk.secondary["logits_k0"] = _block(s, lk, rng, LOGIT_DIMS, alt=alt, **kw)
            blocks.append(blk)
    return blocks


def test_drift_floor_gate_needs_every_endpoint_control() -> None:
    blocks = _pilot_blocks(np.random.default_rng(4))
    assert drv.pilot_quantities_revc(blocks)["gates_primary"]["gate4_drift_floor"] is True
    for b in blocks:
        if b.lock == 175:
            b.z_control_alt = None
    out = drv.pilot_quantities_revc(blocks)
    assert out["gates_primary"]["gate4_drift_floor"] is False
    assert len(out["primary_encoder"]["drift_floor"]["missing_controls"]) == 10
    assert out["fileable_as_drafted"] is False and out["note"]


def test_failed_learning_gate_blocks_the_n_rule() -> None:
    blocks = _pilot_blocks(np.random.default_rng(5))
    for b in blocks:
        b.control_metrics = _m(0.26, 0.2)
    out = drv.pilot_quantities_revc(blocks)
    assert out["gates_primary"]["gate1_learning"] is False
    assert out["fileable_as_drafted"] is False
    assert out["secondary_planning"]["fileable_as_drafted"] is False


def test_undefined_y_gives_no_ratio() -> None:
    rng = np.random.default_rng(6)
    blk = _block(2211, 25, rng, ENCODER_DIMS, alt=True)
    for k in list(blk.z_post)[:5]:
        blk.z_post[k] = torch.zeros_like(blk.z_post[k])
    res, _ = evaluate_block_revc(blk, ENCODER_DIMS)
    assert res.y is None and res.n_valid == 10
    fl = drift_floor(blk, res, ENCODER_DIMS)
    assert fl["R"] is None and fl["F"] is not None


def test_gate5_is_within_seed() -> None:
    blocks = _pilot_blocks(np.random.default_rng(7))
    out = drv.pilot_quantities_revc(blocks)
    within = out["primary_encoder"]["within_seed_sd"]
    assert len(within) == 10 and all(v > 0 for v in within.values())
    assert out["primary_encoder"]["gate5_y_nondegenerate"] is True


def test_heldout_task_metrics_do_not_touch_the_global_rng() -> None:
    world = GaussianWorld(seed=0)
    loop = build_loop(world, 3, lock_after=None, emission="straight-through")
    sample = world.sample(64, seed=4)
    torch.manual_seed(11)
    before = torch.get_rng_state()
    a = heldout_task_metrics(loop, sample)
    assert torch.equal(before, torch.get_rng_state())
    assert a == heldout_task_metrics(loop, sample)
    assert set(a["query_bacc"]) == set(MODALITIES)


def test_capture_block_drift_floor_shares_probe_and_z_pre() -> None:
    world = GaussianWorld(seed=0)
    loop0 = build_loop(world, 9, lock_after=None, emission="straight-through")
    ckpt = loop0.pretrain(2, batch_size=4, seed=9)
    probe = world.sample(32, seed=77)

    def make():  # type: ignore[no-untyped-def]
        return build_loop(world, 9, lock_after=None, emission="straight-through")

    blk = capture_block(make, ckpt, seed=9, lock=25, probe_demod="encoder", probe_gumbel_draws=0,
                        extra_readouts=(("logits", 0),), probe_sample=probe, floor_seed=500_009,
                        control_metric=lambda lp: {"x": 1.0}, **_SMALL)  # fmt: skip
    assert blk.z_control_alt is not None and blk.control_metrics == {"x": 1.0}
    assert not torch.equal(blk.z_control_alt, blk.z_control)
    assert blk.secondary["logits_k0"].z_control_alt is not None
    with pytest.raises(ValueError, match="probe_sample"):
        capture_block(make, ckpt, seed=9, lock=25, probe_demod="encoder", probe_gumbel_draws=0,
                      floor_seed=1, **_SMALL)  # fmt: skip


def test_revc_block_end_to_end_and_roundtrip(
    real_world: StudyforrestRealWorld, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(drv, "STEPS_TRAIN", 2)
    a = drv.run_block(real_world, 2211, 25, design="revc", **_SMALL).block
    b = drv.run_block(real_world, 2211, 25, design="revc", **_SMALL).block
    assert torch.equal(a.z_pre, b.z_pre) and torch.equal(a.z_control_alt, b.z_control_alt)
    assert a.z_pre.shape == (512, 640) and a.secondary["logits_k0"].z_pre.shape == (512, 320)
    assert np.bincount(a.probe_labels.numpy(), minlength=4).tolist() == [128] * 4
    assert set(a.control_metrics) >= {"full_bacc", "best_query_bacc", "query_bacc"}
    no_floor = drv.run_block(real_world, 2211, 50, design="revc", **_SMALL).block
    assert no_floor.z_control_alt is None
    path = drv.block_path(tmp_path, 2211, 25)
    drv.save_block(a, path)
    got = drv.load_block(path)
    assert torch.equal(got.z_control_alt, a.z_control_alt)
    assert got.control_metrics == a.control_metrics
    with pytest.raises(RuntimeError, match="stratified"):
        drv.check_world_revc(StudyforrestRealWorld(seed=0, data_dir=tmp_path))


def test_gate_stage_matches_the_capture_control(
    real_world: StudyforrestRealWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(drv, "STEPS_TRAIN", 2)
    monkeypatch.setattr(drv, "LOCKS", (25,))
    gate = drv.revc_gate_stage(real_world, [2211], steps=2, batch_size=4)
    blk = drv.run_block(real_world, 2211, 25, design="revc", **_SMALL).block
    assert gate["metrics"]["25"]["2211"] == blk.control_metrics
    assert set(gate) >= {"passed", "per_setting", "rule"}


def test_revc_capture_refuses_without_a_passed_gate(tmp_path: Path) -> None:
    import argparse
    import importlib.util
    import json

    path = Path(__file__).resolve().parents[2] / "scripts" / "run_v08_subspace.py"
    spec = importlib.util.spec_from_file_location("run_v08", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"passed": False}))
    for gf in (None, gate):
        ns = argparse.Namespace(seeds=[2211], design="revc", gate_file=gf, osf_filed="",
                                blocks_dir=tmp_path)  # fmt: skip
        with pytest.raises(SystemExit, match="PASSED learning gate"):
            mod.capture(ns)
