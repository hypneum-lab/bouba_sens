"""SensoryWML — per-modality subclass of track_w.mlp_wml.MlpWML.

Sprint 1 Tasks 1.6-1.8 per spec §3.2. Each of the 5 modalities (audio,
vision, tactile, gravity, force) gets its own SensoryWML instance with a
modality-specific `input_proj` encoder. All 5 share a single
`GammaThetaMultiplexer` so the 64-code alphabet is unified across the
sensory cortex (ADR-0001: shared codebook wins over local codebooks).
"""

from __future__ import annotations

from typing import Literal

from torch import nn
from track_p.multiplexer import GammaThetaMultiplexer  # type: ignore[import-not-found]
from track_w.mlp_wml import MlpWML  # type: ignore[import-not-found]

Modality = Literal["audio", "vision", "tactile", "gravity", "force"]
MODALITIES: tuple[Modality, ...] = (
    "audio",
    "vision",
    "tactile",
    "gravity",
    "force",
)


class SensoryWML(MlpWML):  # type: ignore[misc]  # MlpWML is Any due to nerve-wml stubs
    """Modality-typed sensory WML sharing a 64-code alphabet via a common mux.

    The `mux` field is stored via `object.__setattr__` to bypass the
    `nn.Module` auto-registration — otherwise 5 SensoryWMLs would each
    claim the multiplexer as a submodule, double-counting its parameters
    in `parameters()`. Consumers iterate `mux.parameters()` once plus
    each `sensory.parameters()` separately when wiring the optimiser.
    """

    modality: Modality
    input_proj: nn.Module

    def __init__(
        self,
        id: int,
        modality: Modality,
        input_proj: nn.Module,
        mux: GammaThetaMultiplexer,
        *,
        d_hidden: int = 128,
        alphabet_size: int = 64,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            id=id,
            d_hidden=d_hidden,
            alphabet_size=alphabet_size,
            input_dim=d_hidden,
            seed=seed,
        )
        self.modality = modality
        self.input_proj = input_proj
        # Shared mux: bypass nn.Module auto-registration (see class docstring).
        object.__setattr__(self, "mux", mux)
