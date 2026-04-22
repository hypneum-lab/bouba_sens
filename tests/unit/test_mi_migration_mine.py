"""Sprint 17 — MINE wrapper smoke tests for Me3 delta."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bouba_sens.metrics.mi_migration import me3_delta_mine, me3_mi_mine


def test_me3_mi_mine_returns_float_on_synthetic_gaussian() -> None:
    """y = x + small noise -> MI clearly > 0 on an easy 2-D Gaussian."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n = 512  # > batch_size (256), keeps the test well-conditioned
    x = rng.normal(size=n).astype(np.float32)
    labels = (x > 0.0).astype(np.int64)  # deterministic function of x -> MI high
    codes = torch.from_numpy(x)
    labels_t = torch.from_numpy(labels)

    mi_bits = me3_mi_mine(codes, labels_t, n_epochs=200, batch_size=256, tail_average=30)

    assert isinstance(mi_bits, float)
    assert np.isfinite(mi_bits)
    assert mi_bits >= 0.0  # MINE is clipped below at 0 upstream


def test_me3_delta_mine_raises_when_below_batch_size() -> None:
    """N < batch_size must raise (MINE explicitly rejects, wrapper propagates)."""
    torch.manual_seed(0)
    n = 64  # << batch_size 256
    pre = torch.randn(n)
    post = torch.randn(n)
    labels = torch.randint(0, 4, (n,))

    with pytest.raises(ValueError, match="at least"):
        me3_delta_mine(pre, post, labels, n_epochs=100, batch_size=256, tail_average=20)
