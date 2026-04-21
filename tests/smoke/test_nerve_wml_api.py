"""Smoke test that verifies the nerve-wml symbols bouba_sens depends on exist.

Spec §6.3 defines the contract. If any import fails, file a nerve-wml issue
*before* un-commenting the production dependency.
"""

from __future__ import annotations

import importlib

import pytest

# Symbol contract verified 2026-04-20 against nerve-wml master (PR #2 merged
# 2026-04-20, HEAD 77efb4d) at /Users/electron/Documents/Projets/nerve-wml.
# The package exposes 7 top-level modules (nerve_core, track_p, track_w,
# bridge, harness, interpret, neuromorphic) — NOT a unified `nerve_wml.*`
# namespace.
REQUIRED_SYMBOLS = [
    ("nerve_core.protocols", "Nerve"),
    ("nerve_core.neuroletter", "Neuroletter"),
    ("track_w.mlp_wml", "MlpWML"),
    ("track_p.transducer", "Transducer"),
    ("track_p.multiplexer", "GammaThetaMultiplexer"),
]


@pytest.mark.parametrize(("module_name", "symbol"), REQUIRED_SYMBOLS)
def test_nerve_wml_public_symbol_exists(module_name: str, symbol: str) -> None:
    module = importlib.import_module(module_name)
    assert hasattr(module, symbol), (
        f"{module_name}.{symbol} is required by bouba_sens. "
        f"If absent, open a nerve-wml issue before pinning the dependency."
    )
