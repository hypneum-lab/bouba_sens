from __future__ import annotations

import numpy as np

from bouba_sens.metrics.mi_migration import (
    me3_delta,
    me3_delta_binning,
    me3_delta_mine,
)


def _pre_post_labels(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = rng.integers(0, 4, 512)
    pre = rng.normal(size=(512, 16))
    post = pre + rng.normal(scale=0.05, size=(512, 16))
    post[:, 0] += labels * 0.3
    return pre, post, labels


def test_binning_delta_positive_when_label_signal_increases() -> None:
    rng = np.random.default_rng(0)
    pre, post, labels = _pre_post_labels(rng)
    delta = me3_delta_binning(pre, post, labels, bins_per_dim=16)
    assert delta > 0.05


def test_mine_delta_positive_on_same_input() -> None:
    rng = np.random.default_rng(0)
    pre, post, labels = _pre_post_labels(rng)
    delta = me3_delta_mine(pre, post, labels, epochs=200, seed=0)
    assert delta > 0.02


def test_all_three_estimators_agree_on_sign() -> None:
    rng = np.random.default_rng(1)
    pre, post, labels = _pre_post_labels(rng)
    kraskov = me3_delta(pre, post, labels)
    binning = me3_delta_binning(pre, post, labels, bins_per_dim=16)
    mine = me3_delta_mine(pre, post, labels, epochs=200, seed=1)
    signs = {np.sign(kraskov), np.sign(binning), np.sign(mine)}
    assert signs == {1.0}
