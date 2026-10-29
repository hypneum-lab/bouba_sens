"""K-draw Gumbel readout and the switch rule (PI decisions 2026-10-25)."""

from __future__ import annotations

import pytest
import torch

from bouba_sens.readout import (
    GUMBEL_DRAWS,
    SWITCH_BETWEEN,
    SWITCH_TOP1,
    code_structure,
    gumbel_draw_seed,
    gumbel_draws,
    mean_structure,
    switch_detected,
)


def test_frozen_constants_and_seeds() -> None:
    assert (GUMBEL_DRAWS, SWITCH_TOP1, SWITCH_BETWEEN) == (8, 0.25, 0.10)
    assert gumbel_draw_seed(1101, 0, 0) == 2_001_101_000
    assert gumbel_draw_seed(1101, 1, 7) == 2_001_101_107
    seeds = {gumbel_draw_seed(s, p, r) for s in (1101, 1102) for p in (0, 1) for r in range(8)}
    assert len(seeds) == 32
    with pytest.raises(ValueError):
        gumbel_draw_seed(1, 2, 0)


def test_draws_are_reproducible_and_leave_the_global_rng() -> None:
    torch.manual_seed(5)
    expected = torch.rand(3)
    torch.manual_seed(5)
    a = gumbel_draws(lambda: torch.rand(4), 1101, 0, 3)
    assert torch.equal(torch.rand(3), expected)
    b = gumbel_draws(lambda: torch.rand(4), 1101, 0, 3)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    assert not torch.equal(a[0], a[1])
    c = gumbel_draws(lambda: torch.rand(4), 1101, 1, 1)
    assert not torch.equal(a[0], c[0])


def test_code_structure_separates_isotropic_and_class_aligned_codes() -> None:
    gen = torch.Generator().manual_seed(0)
    y = torch.arange(4).repeat_interleave(128)
    iso = torch.randn(512, 64, generator=gen)
    s_iso = code_structure(iso, y)
    assert s_iso["top1_share"] < 0.1 and s_iso["between_class_share"] < 0.03
    aligned = iso.clone()
    aligned[:, 0] += 20.0 * (y >= 2).float()
    s_al = code_structure(aligned, y)
    assert s_al["top1_share"] > 0.4 and s_al["between_class_share"] > 0.4
    assert not switch_detected(s_iso) and switch_detected(s_al)
    assert code_structure(torch.zeros(10, 3), y[:10]) == {
        "top1_share": 0.0,
        "between_class_share": 0.0,
    }


def test_between_class_share_matches_its_definition() -> None:
    gen = torch.Generator().manual_seed(1)
    z = torch.randn(60, 5, generator=gen).double()
    y = torch.arange(3).repeat_interleave(20)
    zc = z - z.mean(0)
    total = float((zc**2).sum()) / 59
    between = sum(20 * float(((z[y == c].mean(0) - z.mean(0)) ** 2).sum()) for c in range(3))
    got = code_structure(z, y)["between_class_share"]
    assert got == pytest.approx(between / 59 / total)


def test_switch_needs_both_thresholds() -> None:
    assert switch_detected({"top1_share": 0.25, "between_class_share": 0.10})
    assert not switch_detected({"top1_share": 0.60, "between_class_share": 0.09})
    assert not switch_detected({"top1_share": 0.24, "between_class_share": 0.50})
    m = mean_structure([torch.eye(4), torch.eye(4) * 2], torch.arange(4))
    assert set(m) == {"top1_share", "between_class_share"}
