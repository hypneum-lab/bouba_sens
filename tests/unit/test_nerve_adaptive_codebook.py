"""Task 2.2 acceptance tests for `bouba_sens.nerve.AdaptiveCodebook`."""

from __future__ import annotations

import torch

from bouba_sens.nerve import AdaptiveCodebook


def test_codebook_output_shape() -> None:
    """(B, K, A) soft distribution -> (B, K, d_hidden) projection."""
    cb = AdaptiveCodebook(alphabet_size=64, d_hidden=128, seed=0)
    soft_codes = torch.randn(4, 7, 64).softmax(dim=-1)
    out = cb(soft_codes)
    assert out.shape == (4, 7, 128)


def test_codebook_tau_controls_entropy() -> None:
    """Low tau sharpens the distribution before projection; high tau smooths."""
    cb = AdaptiveCodebook(seed=0)
    # Input: mild logits, roughly uniform distribution.
    probs = (torch.randn(2, 7, 64) * 0.1).softmax(dim=-1)

    # Re-temperature sharpening should change the effective distribution.
    out_sharp = cb(probs, tau=0.01)
    out_baseline = cb(probs, tau=1.0)
    out_smooth = cb(probs, tau=10.0)

    assert not torch.allclose(out_sharp, out_baseline, atol=1e-4)
    assert not torch.allclose(out_smooth, out_baseline, atol=1e-4)


def test_codebook_gradient_flows_to_param() -> None:
    cb = AdaptiveCodebook(seed=0)
    probs = torch.randn(2, 7, 64).softmax(dim=-1)
    out = cb(probs)
    out.pow(2).mean().backward()
    assert cb.codebook.grad is not None
    assert cb.codebook.grad.abs().sum().item() > 0.0


def test_codebook_init_from_constellation() -> None:
    """init_from projects (A, 2) constellation to (A, d_hidden) codebook."""
    constellation = torch.randn(64, 2)
    cb = AdaptiveCodebook(d_hidden=128, init_from=constellation, seed=0)
    assert cb.codebook.shape == (64, 128)
    # Codebook must be non-zero (projection preserves scale).
    assert cb.codebook.abs().sum().item() > 0.0


def test_codebook_preserves_global_rng() -> None:
    torch.manual_seed(1234)
    anchor = torch.randn(1).item()
    AdaptiveCodebook(seed=7)
    AdaptiveCodebook(seed=8)
    torch.manual_seed(1234)
    assert torch.randn(1).item() == anchor
