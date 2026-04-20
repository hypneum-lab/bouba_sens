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
