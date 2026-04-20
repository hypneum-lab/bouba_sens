"""Task 3.2 acceptance tests for Me3 (MI migration)."""

from __future__ import annotations

import torch

from bouba_sens.metrics.mi_migration import me3_delta, me3_mi


def test_me3_zero_for_independent_codes_and_labels() -> None:
    """Independent random noise -> MI near zero (< 0.1 bit)."""
    torch.manual_seed(0)
    codes = torch.randn(2048)
    labels = torch.randint(0, 4, (2048,))
    mi = me3_mi(codes, labels)
    assert mi < 0.1, f"independent MI {mi:.3f} should be near zero"


def test_me3_positive_when_codes_predict_labels() -> None:
    """codes = labels + noise -> MI > 0.5 bit."""
    torch.manual_seed(0)
    labels = torch.randint(0, 4, (2048,))
    codes = labels.float() + 0.1 * torch.randn(2048)
    mi = me3_mi(codes, labels)
    assert mi > 0.5, f"aligned MI {mi:.3f} should be > 0.5"


def test_me3_delta_increases_with_post_alignment() -> None:
    """Pre random, post aligned -> positive delta."""
    torch.manual_seed(1)
    labels = torch.randint(0, 4, (2048,))
    pre = torch.randn(2048)
    post = labels.float() + 0.1 * torch.randn(2048)
    delta = me3_delta(pre, post, labels)
    assert delta > 0.5, f"alignment delta {delta:.3f} should be > 0.5"


def test_me3_delta_zero_when_both_aligned_equally() -> None:
    """Pre and post equally informative -> delta near 0."""
    torch.manual_seed(2)
    labels = torch.randint(0, 4, (2048,))
    pre = labels.float() + 0.1 * torch.randn(2048)
    post = labels.float() + 0.1 * torch.randn(2048)
    delta = me3_delta(pre, post, labels)
    assert abs(delta) < 0.5, f"symmetric-info delta {delta:.3f} should be near zero"
