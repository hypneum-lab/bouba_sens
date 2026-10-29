"""Deterministic soft readout reproduces nerve-wml's demodulator logits (PI 2026-10-25)."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from bouba_sens.builders import build_loop
from bouba_sens.demod import demod_logits, softmax_readout
from bouba_sens.sensory import MODALITIES
from bouba_sens.world.gaussian import GaussianWorld

pytestmark = pytest.mark.usefixtures("deterministic_torch")


def _carriers():  # type: ignore[no-untyped-def]
    loop = build_loop(GaussianWorld(seed=0), 9001, lock_after=None)
    loop.pretrain(3, batch_size=8, seed=9001)
    sample = loop.world.sample(batch_size=32, seed=1)
    with torch.no_grad():
        return loop, {m: loop.sensories[m].step(getattr(sample, m)) for m in MODALITIES}


def test_logits_match_demodulate_argmax_and_gumbel_path() -> None:
    loop, carriers = _carriers()
    mux = loop.mux
    for m in MODALITIES:
        logits = demod_logits(mux, carriers[m])
        assert torch.equal(logits.argmax(-1), mux.demodulate(carriers[m], hard=True))
        torch.manual_seed(123)
        ref = mux.demodulate(carriers[m], hard=False, tau=1.0)
        torch.manual_seed(123)
        ours = F.gumbel_softmax(logits, tau=1.0, hard=False)
        assert torch.equal(ref, ours)  # the logits themselves, not only their argmax


def test_softmax_fuse_is_deterministic_and_distinct() -> None:
    loop, carriers = _carriers()
    with torch.no_grad():
        a = loop.nerve.fuse(carriers, softmax_demod=True)
        torch.rand(10)
        b = loop.nerve.fuse(carriers, softmax_demod=True)
        h = loop.nerve.fuse(carriers, hard_demod=True)
    assert torch.equal(a, b) and not torch.equal(a, h)
    p = softmax_readout(loop.mux, carriers["audio"])
    assert torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)))
    with pytest.raises(ValueError, match="exclusive"):
        loop.nerve.fuse(carriers, hard_demod=True, softmax_demod=True)


def test_lesion_phase_softmax_primary_and_gumbel_secondary() -> None:
    loop = build_loop(GaussianWorld(seed=0), 9001, lock_after=None)
    loop.pretrain(2, batch_size=4, seed=9001)
    rep = loop.lesion_phase(
        None, 2, batch_size=4, seed=9001, probe_size=16, probe_demod="softmax",
        extra_probe_readouts=(("soft", 2),),
    )  # fmt: skip
    assert set(rep.extra_codes) == {"soft_k2"}
    pre, post = rep.extra_codes["soft_k2"]
    assert pre.shape == rep.pre_lesion_codes.shape == post.shape  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        loop.lesion_phase(None, 1, batch_size=4, seed=9001, probe_demod="softmax",
                          probe_gumbel_draws=2)  # fmt: skip
