"""Task 2.8 acceptance tests for `IntegrationHead`."""

from __future__ import annotations

import torch

from bouba_sens.head import IntegrationHead


def test_head_output_shape_default_10_classes() -> None:
    head = IntegrationHead()
    fused = torch.randn(4, 7, 128)
    logits = head(fused)
    assert logits.shape == (4, 10)


def test_head_respects_n_classes_kwarg() -> None:
    head = IntegrationHead(n_classes=4)
    fused = torch.randn(2, 7, 128)
    assert head(fused).shape == (2, 4)


def test_head_gradient_flows() -> None:
    head = IntegrationHead()
    fused = torch.randn(2, 7, 128, requires_grad=True)
    logits = head(fused)
    logits.pow(2).mean().backward()
    assert head.classifier.weight.grad is not None
    assert head.classifier.weight.grad.abs().sum().item() > 0.0
    # Gradient also flows back to the input (feeds into fuse backprop).
    assert fused.grad is not None
    assert fused.grad.abs().sum().item() > 0.0


def test_head_preserves_global_rng() -> None:
    torch.manual_seed(1234)
    anchor = torch.randn(1).item()
    IntegrationHead()
    IntegrationHead(n_classes=20)
    torch.manual_seed(1234)
    assert torch.randn(1).item() == anchor
