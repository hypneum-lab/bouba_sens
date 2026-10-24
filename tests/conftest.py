"""Shared pytest fixtures and configuration for bouba_sens tests."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic_seed() -> None:
    """Seed all RNGs before every test to keep failures reproducible."""

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


@pytest.fixture
def small_batch() -> int:
    """Default small batch size for fast unit tests."""

    return 4


@pytest.fixture
def deterministic_torch():  # type: ignore[no-untyped-def]
    """Pin torch to deterministic kernels for bit-identical A/B comparisons.

    On Linux x86 the default CPU ``torch.linalg.lstsq`` (LAPACK gelsy, called
    by ``GammaThetaMultiplexer.demodulate`` on every ``CrossModalNerve.fuse``)
    returns results that differ by ~1 ulp (3e-7) between two calls on
    identical input, depending on heap/buffer state; Adam then amplifies that
    into different weights. It is identical-code noise, not a loop change:
    ``AdaptationLoop`` vs. its replica diverge even when the replica is run
    twice. ``use_deterministic_algorithms`` removes it (checked with 1 and 8
    threads); arm64 never showed it.
    """
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev)
