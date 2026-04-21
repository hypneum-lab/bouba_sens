"""Tests for Platonic-RH mutual_knn alignment vendored from nerve-wml."""

from __future__ import annotations

import torch

from bouba_sens.metrics.alignment import mutual_knn


def test_self_alignment_is_one() -> None:
    """Same tensor aligned with itself must score 1.0 exactly."""
    rng = torch.Generator().manual_seed(0)
    x = torch.randn((64, 16), generator=rng)
    assert mutual_knn(x, x, k=10) == 1.0


def test_random_alignment_is_near_chance() -> None:
    """Independent Gaussians at k=10 / N=256 score near k/N = 0.039."""
    rng = torch.Generator().manual_seed(0)
    a = torch.randn((256, 16), generator=rng)
    b = torch.randn((256, 16), generator=rng)
    score = mutual_knn(a, b, k=10)
    # Chance level k/N = 0.039; allow 3x for small-N variance
    assert 0.0 <= score < 0.12


def test_rotation_preserves_alignment() -> None:
    """A rotated copy preserves neighbour structure, so alignment == 1.0."""
    rng = torch.Generator().manual_seed(0)
    x = torch.randn((64, 8), generator=rng)
    # Random orthogonal matrix via QR
    q, _ = torch.linalg.qr(torch.randn((8, 8), generator=rng))
    y = x @ q
    assert mutual_knn(x, y, k=5) == 1.0


def test_k_must_be_less_than_n() -> None:
    import pytest

    x = torch.randn((5, 4))
    with pytest.raises(AssertionError):
        mutual_knn(x, x, k=5)


def test_batch_size_mismatch_raises() -> None:
    import pytest

    a = torch.randn((32, 8))
    b = torch.randn((33, 8))
    with pytest.raises(AssertionError):
        mutual_knn(a, b, k=5)
