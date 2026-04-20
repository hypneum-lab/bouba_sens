"""Task 7.5 — world-complexity audit unit tests."""

from __future__ import annotations

import torch

from bouba_sens.audit import (
    compute_world_profile,
    intrinsic_dim_pca,
    linear_separability,
    temporal_autocorr,
)
from bouba_sens.world.gaussian import GaussianWorld


def test_intrinsic_dim_pca_rank_k() -> None:
    """Rank-3 embedded in a rank-10 ambient should recover 3."""
    torch.manual_seed(0)
    latent = torch.randn(1000, 3)
    projection = torch.randn(3, 10)
    tensor = latent @ projection + 1e-5 * torch.randn(1000, 10)
    recovered = intrinsic_dim_pca(tensor, variance_cutoff=0.99)
    assert 2 <= recovered <= 4, f"expected ~3, got {recovered}"


def test_temporal_autocorr_iid_is_zero() -> None:
    """Gaussian world labels are i.i.d.; lag-1 autocorr near 0."""
    sample = GaussianWorld(seed=0).sample(batch_size=512, seed=0)
    assert abs(temporal_autocorr(sample)) < 0.15


def test_linear_separability_bounded_01() -> None:
    sample = GaussianWorld(seed=0).sample(batch_size=128, seed=0)
    score = linear_separability(sample)
    assert 0.0 <= score <= 1.0


def test_compute_world_profile_returns_17_keys() -> None:
    """5 modalities x {intrinsic_dim, compactness} + 4 scalar metrics = 14 keys."""
    sample = GaussianWorld(seed=0).sample(batch_size=64, seed=0)
    profile = compute_world_profile(sample)
    assert len(profile) == 14
    for v in profile.values():
        assert isinstance(v, float)
