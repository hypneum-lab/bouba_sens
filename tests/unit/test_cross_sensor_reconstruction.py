"""v0.9 mitigation arm: auxiliary cross-sensor reconstruction (PI 2026-10-24)."""

from __future__ import annotations

from pathlib import Path

import torch

from bouba_sens.decorrelation import (
    RECON_WEIGHT,
    CrossSensorReconstruction,
    DecorrelationSchedule,
    DecorrelationWindow,
)
from bouba_sens.sensory import MODALITIES
from bouba_sens.world.studyforrest import StudyforrestRealWorld
from tests.unit.test_loop import _build_loop


def _states(ckpt) -> list[torch.Tensor]:  # type: ignore[no-untyped-def]
    out = []
    for st in (ckpt.mux_state, ckpt.nerve_state, ckpt.head_state, *ckpt.sensory_states.values()):
        out += [v.clone() for v in st.values()]
    return out


def _recon(loop, weight: float = RECON_WEIGHT) -> CrossSensorReconstruction:  # type: ignore[no-untyped-def]
    dims = loop.world.modality_dims()
    return CrossSensorReconstruction(dims, weight=weight, seed=3)


def test_registered_weight_is_one() -> None:
    assert RECON_WEIGHT == 1.0


def test_zero_weight_is_bit_identical_to_no_objective(deterministic_torch: None) -> None:
    # Linux x86 lstsq differs by ~1 ulp between identical calls (conftest).
    torch.manual_seed(0)
    a = _build_loop(n_classes=4)
    ref = a.pretrain(steps=4, batch_size=8, seed=0)
    torch.manual_seed(0)
    b = _build_loop(n_classes=4)
    rec = _recon(b, weight=0.0)
    got = b.pretrain(steps=4, batch_size=8, seed=0, reconstruction=rec)
    for x, y in zip(_states(ref), _states(got), strict=True):
        assert torch.equal(x, y)


def test_objective_trains_decoders_and_encoders_once_registered() -> None:
    torch.manual_seed(0)
    a = _build_loop(n_classes=4)
    ref = a.pretrain(steps=3, batch_size=8, seed=0)
    torch.manual_seed(0)
    b = _build_loop(n_classes=4)
    rec = _recon(b)
    dec0 = [p.detach().clone() for p in rec.parameters()]
    got = b.pretrain(steps=3, batch_size=8, seed=0, reconstruction=rec)
    assert any(not torch.equal(p, q) for p, q in zip(dec0, rec.parameters(), strict=True))
    enc_ref = ref.sensory_states["audio"]
    enc_got = got.sensory_states["audio"]
    assert any(not torch.equal(enc_ref[k], enc_got[k]) for k in enc_ref)
    n_groups = len(b.opt.param_groups)
    b.pretrain(steps=1, batch_size=8, seed=10, reconstruction=rec)
    assert len(b.opt.param_groups) == n_groups  # decoders added exactly once


def test_loss_matches_definition() -> None:
    loop = _build_loop(n_classes=4)
    rec = _recon(loop)
    sample = loop.world.sample(batch_size=8, seed=1)
    reps = {
        m: loop.sensories[m].core(loop.sensories[m].input_proj(getattr(sample, m)))
        for m in MODALITIES
    }
    terms = []
    for m in MODALITIES:
        others = torch.stack([reps[o] for o in MODALITIES if o != m]).mean(0)
        x = getattr(sample, m).reshape(8, -1)
        target = (x - x.mean(0)) / (x.std(0, unbiased=False) + 1e-6)
        terms.append(((rec.decoders[m](others) - target) ** 2).mean())
    expected = torch.stack(terms).mean()
    assert torch.allclose(rec(loop, sample), expected, rtol=1e-6, atol=1e-7)


def test_construction_leaves_global_rng_untouched() -> None:
    loop = _build_loop(n_classes=4)
    torch.manual_seed(123)
    before = torch.get_rng_state()
    _recon(loop)
    assert torch.equal(before, torch.get_rng_state())


def test_combines_with_decorrelation_window(tmp_path: Path) -> None:
    n = 256
    gen = torch.Generator().manual_seed(0)
    for name, shape in [
        ("audio", (n, 128)),
        ("vision", (n, 16, 16)),
        ("tactile", (n, 32)),
        ("gravity", (n, 3)),
        ("force", (n, 6)),
    ]:
        torch.save(torch.randn(*shape, generator=gen), tmp_path / f"{name}.pt")
    torch.save(torch.randint(0, 4, (n,), generator=gen), tmp_path / "labels.pt")
    world = StudyforrestRealWorld(seed=0, data_dir=tmp_path, probe_fraction=0.10)
    loop = _build_loop(n_classes=4)
    loop.world = world
    rec = CrossSensorReconstruction(world.modality_dims(), seed=1)
    sch = DecorrelationSchedule(DecorrelationWindow(0, 2), "decor", seed=5, world=world)
    ckpt = loop.pretrain(steps=3, batch_size=8, seed=0, decorrelation=sch, reconstruction=rec)
    assert all(torch.isfinite(v).all() for v in _states(ckpt))
