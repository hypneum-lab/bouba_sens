"""Deterministic soft readout of a gamma/theta carrier (PI decision 2026-10-25, 19:47).

v0.8 rev. B registers a **deterministic soft readout** as the primary
probe readout: the demodulator's per-symbol logits are turned into
probabilities by ``softmax(logits / tau)`` (tau = 1, the training
temperature), with no Gumbel sample. It is a different readout from the
Gumbel one, not its expectation.

``demod_logits`` reproduces, step by step, the arithmetic of nerve-wml
``GammaThetaMultiplexer.demodulate`` (v1.7.0, the CI pin) up to the
logits: theta-envelope removal, the overlapping Gaussian x gamma
quadrature basis, the global least-squares recovery of (I, Q) and the
negative squared distance to the unit-normalised constellation. It is
copied rather than imported so that nerve-wml is not changed (HITL
dependency policy); ``tests/unit/test_demod.py`` checks it against
``demodulate``: its argmax equals ``demodulate(hard=True)``, and from
the same RNG state ``gumbel_softmax(logits)`` equals
``demodulate(hard=False)`` bit for bit, which validates the logits
themselves, not only their argmax.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

SOFTMAX_TAU = 1.0


def demod_logits(mux: Any, carrier: Tensor, *, theta_phase_offset: float = 0.0) -> Tensor:
    """Per-symbol logits ``[B, K, alphabet]`` exactly as ``mux.demodulate`` computes them."""
    batch, n_samples = carrier.shape
    t = mux._t_grid
    k_cap = mux.cfg.symbols_per_theta
    window_n = n_samples // k_cap
    two_pi_gamma_t = 2.0 * torch.pi * mux.cfg.gamma_hz * t
    gamma_i = torch.cos(two_pi_gamma_t)
    gamma_q = torch.sin(two_pi_gamma_t)
    effective_theta_hz = mux.cfg.sample_rate_hz / n_samples
    two_pi_theta_t = 2.0 * torch.pi * effective_theta_hz * t + theta_phase_offset
    theta_env = 0.55 + 0.45 * torch.cos(two_pi_theta_t)
    carrier_norm = carrier / theta_env.unsqueeze(0)
    window_centers = (torch.arange(k_cap, device=t.device, dtype=t.dtype) + 0.5) * window_n
    sigma = window_n / 4.0
    idx = torch.arange(n_samples, device=t.device, dtype=t.dtype)
    gaussian_masks = torch.exp(
        -((idx.unsqueeze(0) - window_centers.unsqueeze(1)) ** 2) / (2.0 * sigma**2)
    )
    basis_i = gaussian_masks * gamma_i.unsqueeze(0)
    basis_q = gaussian_masks * gamma_q.unsqueeze(0)
    basis_matrix = (
        torch.stack([basis_i, basis_q], dim=-1).permute(1, 0, 2).reshape(n_samples, 2 * k_cap)
    )
    sol = torch.linalg.lstsq(basis_matrix, carrier_norm.T).solution
    recovered = sol.T.reshape(batch, k_cap, 2).to(torch.float32)
    const_norm = mux.constellation / mux.constellation.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dist_sq = (recovered.reshape(-1, 1, 2) - const_norm.unsqueeze(0)).pow(2).sum(dim=-1)
    logits: Tensor = (-dist_sq).reshape(batch, k_cap, -1)
    return logits


def softmax_readout(mux: Any, carrier: Tensor, *, tau: float = SOFTMAX_TAU) -> Tensor:
    """Deterministic soft code distribution ``softmax(logits / tau)``, ``[B, K, alphabet]``."""
    return F.softmax(demod_logits(mux, carrier) / tau, dim=-1)
