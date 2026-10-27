"""SensoryWML — per-modality subclass of track_w.mlp_wml.MlpWML.

Sprint 1 Tasks 1.6-1.8 per spec §3.2. Each of the 5 modalities (audio,
vision, tactile, gravity, force) gets its own SensoryWML instance with a
modality-specific `input_proj` encoder. All 5 share a single
`GammaThetaMultiplexer` so the 64-code alphabet is unified across the
sensory cortex (ADR-0001: shared codebook wins over local codebooks).
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from track_p.multiplexer import GammaThetaMultiplexer
from track_w.mlp_wml import MlpWML

Modality = Literal["audio", "vision", "tactile", "gravity", "force"]
# ADR-0024 follow-up — code emission rule of SensoryWML.step.
# "argmax-bridge" (default) is the historical rule: argmax code plus a 1e-3
# softmax bridge, under which the task gradient is ~1e-6 and the network does
# not learn (docs/milestones/2026-10-26-code-collapse-diagnosis.md).
# "straight-through" sends the same argmax code forward and back-propagates
# the softmax gradient over the alphabet.
EMISSIONS: tuple[str, ...] = ("argmax-bridge", "straight-through")
DEFAULT_EMISSION = "argmax-bridge"

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
        emission: str = DEFAULT_EMISSION,
    ) -> None:
        super().__init__(
            id=id,
            d_hidden=d_hidden,
            alphabet_size=alphabet_size,
            input_dim=d_hidden,
            seed=seed,
        )
        if emission not in EMISSIONS:
            raise ValueError(f"emission must be one of {EMISSIONS}; got {emission!r}")
        self.modality = modality
        self.emission = emission
        self.input_proj = input_proj
        # Shared mux: bypass nn.Module auto-registration (see class docstring).
        object.__setattr__(self, "mux", mux)

    def step(self, x: Tensor) -> Tensor:
        """Encode a modality tensor into a gamma/theta PAC carrier.

        1. `input_proj(x)` lifts the modality to `(B, d_hidden)`.
        2. MlpWML's MLP core refines the representation.
        3. `emit_head_pi` projects to alphabet logits; argmax selects
           the code per batch item.
        4. The single code is replicated across K gamma-slots (Sprint 1
           MVP; Sprint 2 CrossModalNerve.fuse will produce K
           independent codes per slot).
        5. The shared `mux` modulates the K codes into one theta period.
        6. A small softmax-weighted bridge term preserves the
           gradient path from `input_proj` through the otherwise-
           non-differentiable argmax. Factor 1e-3 keeps the hard
           carrier dominant for downstream roundtrip tests while
           still letting backprop reach the encoder weights.

        Args:
            x: modality tensor of the native shape documented on the
                encoder (e.g. `(B, 128)` audio, `(B, 16, 16)` vision).

        Returns:
            carrier: `(B, T)` float32, with `T = mux._t_grid.numel()`.
        """
        h = self.input_proj(x)
        h = self.core(h)
        pi_logits = self.emit_head_pi(h)
        if self.emission == "straight-through":
            return self._step_straight_through(pi_logits)
        base_code = pi_logits.argmax(-1)  # (B,)
        k = self.mux.cfg.symbols_per_theta
        codes = base_code.unsqueeze(-1).expand(-1, k)  # (B, K)
        carrier: Tensor = self.mux.forward(codes)

        # Differentiable bridge — argmax blocks gradient to input_proj;
        # add a small softmax-IQ contribution so backprop reaches the
        # modality encoder. Hard carrier dominates (factor 1e-3).
        soft_probs = pi_logits.softmax(-1)  # (B, A)
        soft_iq = soft_probs @ self.mux.constellation  # (B, 2)
        bridge = soft_iq.mean(-1, keepdim=True).expand_as(carrier) * 1e-3
        return cast(Tensor, carrier + bridge)

    def _step_straight_through(self, pi_logits: Tensor) -> Tensor:
        """Straight-through emission (ADR-0024 follow-up, opt-in).

        Forward: the carrier of the argmax code, exactly as the default rule
        builds it from the code (no bridge term). Backward: the gradient of
        the softmax over the alphabet, through the carriers of all codes, so
        the task loss reaches `emit_head_pi`, the core and the encoder.
        """
        soft = pi_logits.softmax(-1)  # (B, A)
        hard = F.one_hot(soft.argmax(-1), soft.shape[-1]).to(soft.dtype)
        st = hard + (soft - soft.detach())  # value == one-hot
        k = self.mux.cfg.symbols_per_theta
        alphabet = torch.arange(soft.shape[-1], device=soft.device)
        table = self.mux.forward(alphabet.unsqueeze(-1).expand(-1, k))  # (A, T)
        return cast(Tensor, st @ table)
