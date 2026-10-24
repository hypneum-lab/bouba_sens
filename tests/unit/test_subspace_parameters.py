"""v0.8 secondary metric: frozen parameter list and per-matrix shift."""

from __future__ import annotations

import torch

from bouba_sens.metrics.subspace import (
    interaction_parameter_names,
    parameter_subspace_shifts,
)
from tests.unit.test_loop import _build_loop


def test_parameter_list_is_frozen_and_ordered() -> None:
    nerve = _build_loop().nerve
    names = interaction_parameter_names(nerve)
    assert names[0] == "codebook.codebook"
    assert len(names) == 1 + 20 * 2
    assert all(
        n.startswith("transducers.") and n.endswith(("fc1.weight", "fc2.weight")) for n in names[1:]
    )
    assert len(set(names)) == len(names)
    assert names == interaction_parameter_names(_build_loop().nerve)  # stable across instances
    assert names[1:3] == [
        "transducers.audio->vision.fc1.weight",
        "transducers.audio->vision.fc2.weight",
    ]
    state = nerve.state_dict()
    assert all(state[n].dim() == 2 for n in names)


def test_unchanged_matrix_is_reported_inactive_and_changed_one_is_measured() -> None:
    nerve = _build_loop().nerve
    before = {k: v.clone() for k, v in nerve.state_dict().items()}
    after = {k: v.clone() for k, v in before.items()}
    after["codebook.codebook"] = (
        after["codebook.codebook"]
        @ torch.linalg.qr(torch.randn(128, 128, generator=torch.Generator().manual_seed(0)))[0]
    )
    out = parameter_subspace_shifts(before, after, interaction_parameter_names(nerve), k=12)
    assert out["transducers.audio->vision.fc1.weight"] is None  # inactive: no update
    s = out["codebook.codebook"]
    assert s is not None and 0.0 < s <= 1.0


def test_identical_subspace_after_rescaling_gives_zero() -> None:
    w = torch.randn(64, 128, generator=torch.Generator().manual_seed(1))
    out = parameter_subspace_shifts({"w": w}, {"w": 2.5 * w}, ["w"], k=12)
    assert out["w"] == 0.0 or abs(out["w"]) < 1e-12
