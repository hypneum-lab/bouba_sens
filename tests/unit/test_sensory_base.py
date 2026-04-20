"""Task 1.6 acceptance tests for `bouba_sens.sensory.SensoryWML`."""

from __future__ import annotations

from torch import nn
from track_p.multiplexer import GammaThetaMultiplexer
from track_w.mlp_wml import MlpWML

from bouba_sens.sensory import MODALITIES, SensoryWML


def _dummy_proj(d_hidden: int = 128) -> nn.Module:
    """Return a simple Linear that outputs (B, d_hidden)."""
    return nn.Linear(d_hidden, d_hidden)


def test_sensorywml_is_mlpwml_subclass() -> None:
    """MRO must include MlpWML so SensoryWML inherits the WML protocol."""
    assert issubclass(SensoryWML, MlpWML)


def test_modalities_tuple_has_five_elements() -> None:
    assert len(MODALITIES) == 5
    assert set(MODALITIES) == {"audio", "vision", "tactile", "gravity", "force"}


def test_shared_mux_not_duplicated() -> None:
    """Two SensoryWMLs constructed with the same mux share constellation identity."""
    mux = GammaThetaMultiplexer(seed=0)
    proj1 = _dummy_proj()
    proj2 = _dummy_proj()
    s1 = SensoryWML(id=0, modality="audio", input_proj=proj1, mux=mux, seed=1)
    s2 = SensoryWML(id=1, modality="vision", input_proj=proj2, mux=mux, seed=2)
    assert id(s1.mux.constellation) == id(s2.mux.constellation)
    assert s1.mux is s2.mux


def test_shared_mux_not_registered_as_submodule() -> None:
    """The mux must NOT appear in SensoryWML.parameters() — shared ownership."""
    mux = GammaThetaMultiplexer(seed=0)
    proj = _dummy_proj()
    wml = SensoryWML(id=0, modality="audio", input_proj=proj, mux=mux, seed=0)
    wml_param_ids = {id(p) for p in wml.parameters()}
    assert id(mux.constellation) not in wml_param_ids, (
        "mux.constellation leaked into SensoryWML.parameters() — "
        "object.__setattr__ bypass is broken"
    )


def test_modality_and_input_proj_attributes_set() -> None:
    mux = GammaThetaMultiplexer(seed=0)
    proj = _dummy_proj()
    wml = SensoryWML(id=3, modality="tactile", input_proj=proj, mux=mux, seed=0)
    assert wml.modality == "tactile"
    assert wml.input_proj is proj
    assert wml.id == 3
