"""CrossModalNerve and its plasticity mechanisms — spec §3.3.

Sprint 2 lands the plastic router in four layers:

- P1 `PlasticityGate`  : channel-wise attentional weights (Task 2.1).
- P2 `AdaptiveCodebook`: soft-assignment projection (Task 2.2).
- P3 `CrossModalTransducer`: per-pair MLP with 0/1 gating (Task 2.3).
- `CrossModalNerve`    : assembly + `fuse()` (Tasks 2.4-2.5).

All four classes live in this module so they share the Modality
vocabulary and keep the import surface tight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from bouba_sens.sensory import MODALITIES, Modality

if TYPE_CHECKING:
    from track_p.multiplexer import GammaThetaMultiplexer


class PlasticityGate(nn.Module):
    """Channel-wise attentional weights over the 5 modalities (P1).

    A single learnable `alpha` vector of shape `(5,)` is passed through
    `softmax` to produce non-negative weights summing to 1. Init is all-
    zeros so the softmax is uniform (0.2 per channel) — any early
    imbalance must emerge from training or a `CrossModalNerve.on_lesion`
    update.

    Forward takes a `dict[Modality, Tensor]` (one carrier per modality,
    same leading dims) and returns the same dict with each tensor
    scaled by its modality's softmax weight.
    """

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(len(MODALITIES)))

    def weights(self) -> Tensor:
        """Return the current softmax gate weights as a `(5,)` tensor."""
        return self.alpha.softmax(dim=-1)

    def forward(self, letters: dict[Modality, Tensor]) -> dict[Modality, Tensor]:
        w = self.weights()
        return {m: letters[m] * w[i] for i, m in enumerate(MODALITIES)}


class AdaptiveCodebook(nn.Module):
    """Soft-assignment projection from code distributions to hidden vectors (P2).

    Wraps a learnable `(alphabet_size, d_hidden)` parameter. Forward takes
    a distribution `[B, K, A]` (already softmaxed — typically from
    `mux.demodulate(hard=False)`) and matmuls to `[B, K, d_hidden]`.

    The optional `tau` kwarg re-temperatures the input distribution
    before projection: `tau < 1.0` sharpens (entropy down), `tau > 1.0`
    smooths. `tau = 1.0` (default) is a no-op. `AdaptationLoop` anneals
    tau from 1.0 to 0.1 during training per spec §4.3.

    Init can optionally seed the codebook from an existing constellation
    (e.g. `mux.constellation`, shape `(A, 2)`) by projecting through a
    random `(2, d_hidden)` map — the angular structure of the PSK
    alphabet is preserved while leaving the codebook free to adapt.
    """

    def __init__(
        self,
        alphabet_size: int = 64,
        d_hidden: int = 128,
        *,
        init_from: Tensor | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        global_state = torch.get_rng_state()
        if seed is not None:
            torch.manual_seed(seed)
        if init_from is not None:
            proj = torch.randn(init_from.shape[-1], d_hidden) * 0.1
            init = (init_from @ proj).detach()
        else:
            init = torch.randn(alphabet_size, d_hidden) * 0.1
        torch.set_rng_state(global_state)
        self.codebook = nn.Parameter(init)

    def forward(self, soft_codes: Tensor, *, tau: float = 1.0) -> Tensor:
        if tau != 1.0:
            logits = soft_codes.clamp(min=1e-9).log() / tau
            soft_codes = logits.softmax(dim=-1)
        return soft_codes @ self.codebook


class CrossModalTransducer(nn.Module):
    """Directed-edge MLP that translates source-modality hidden states
    into target-modality hidden states (P3).

    A 2-hidden-layer MLP `(d_hidden -> d_hidden -> d_hidden)` with
    0/1 gating controlled by the caller at each forward call. When
    `active=False`, forward returns the input unchanged (identity fallback).

    Activation policy per spec §4.3: `active = (gate[source] < 0.1 AND
    gate[target] > 0.3)`. The activation decision is external — this
    class only executes the MLP when instructed.

    `(source, target)` is kept as instance metadata for logging and
    MigrationReport slicing.
    """

    def __init__(
        self,
        source: Modality,
        target: Modality,
        d_hidden: int = 128,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.source = source
        self.target = target
        global_state = torch.get_rng_state()
        if seed is not None:
            torch.manual_seed(seed)
        self.fc1 = nn.Linear(d_hidden, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        torch.set_rng_state(global_state)

    def forward(
        self,
        x: Tensor,
        *,
        active: bool = True,
        soft_activate: float | None = None,
    ) -> Tensor:
        """Apply the MLP with either hard or soft gating.

        Args:
            active: legacy binary gate (v0.3 behaviour). Ignored when
                ``soft_activate`` is provided.
            soft_activate: continuous coefficient in [0, 1]. When set,
                returns ``soft_activate * mlp(x) + (1 - soft_activate) * x``
                so the network can interpolate between identity and
                transformation. Enables nerve-wml#5-style Gumbel gating
                for compound critical-period experiments (Sprint 11).
        """
        if soft_activate is not None:
            s = float(soft_activate)
            transformed = self.fc2(torch.relu(self.fc1(x)))
            return cast(Tensor, s * transformed + (1.0 - s) * x)
        if not active:
            return x
        return cast(Tensor, self.fc2(torch.relu(self.fc1(x))))


class CrossModalNerve(nn.Module):
    """Plastic router fusing 5-modality carriers into a shared representation.

    Assembles the three plasticity mechanisms per spec §3.3:

    - `gates`: `PlasticityGate` (P1) — channel weights alpha.
    - `codebook`: `AdaptiveCodebook` (P2) — soft-assignment projection,
      initialised from `mux.constellation` by default so the fused
      representation starts aligned with the shared PSK alphabet.
    - `transducers`: 20 `CrossModalTransducer` instances (5x5 minus 5
      self-loops), keyed as `"{src}->{dst}"`, activated per spec §4.3
      when `gate[src] < 0.1 AND gate[dst] > 0.3`.

    The shared multiplexer reference is stored via `object.__setattr__`
    to avoid nn.Module double-registration (SensoryWML convention).

    `fuse(letters)` takes 5 `(B, T)` carriers from SensoryWML.step and
    returns a `(B, K, d_hidden)` fused tensor consumed by
    `IntegrationHead` (Task 2.8).
    """

    # Class-level annotations so mypy resolves self.mux after the
    # object.__setattr__ bypass AND the register_buffer lifted attrs.
    mux: GammaThetaMultiplexer
    codebook_step: Tensor

    def __init__(
        self,
        mux: GammaThetaMultiplexer,
        d_hidden: int = 128,
        *,
        seed: int | None = None,
        transducer_gating: str = "hard",
        gumbel_tau: float = 1.0,
        codebook_lock_after: int | None = None,
        transducer_gating_schedule: int | None = None,
        transducer_gating_target: str | None = None,
    ) -> None:
        """Sprint 11 `transducer_gating` kwarg + Sprint 13
        `codebook_lock_after` third compound-critical-period component
        + Sprint 14 `transducer_gating_schedule` phase transition.
        ``codebook_lock_after=None`` (default) preserves v0.5 behaviour;
        when set to an int N, ``AdaptiveCodebook.codebook`` Parameter
        permanently freezes (``requires_grad=False``) once the nerve's
        internal step counter crosses N. Consumers call ``nerve.step()``
        once per training iteration (analogous to ``mux.step()``).
        When ``transducer_gating_schedule`` is an int N, the gating
        mode silently switches from ``transducer_gating`` to
        ``transducer_gating_target`` once the counter crosses N (read
        inside ``fuse`` so checkpoint restores resume the correct
        mode — see Sprint 14 ADR).
        """
        super().__init__()
        self.gates = PlasticityGate()
        self.codebook = AdaptiveCodebook(
            alphabet_size=mux.cfg.alphabet_size,
            d_hidden=d_hidden,
            init_from=mux.constellation.detach(),
            seed=seed,
        )
        self.transducers = nn.ModuleDict(
            {
                f"{src}->{dst}": CrossModalTransducer(src, dst, d_hidden, seed=seed)
                for src in MODALITIES
                for dst in MODALITIES
                if src != dst
            }
        )
        self.transducer_gating = transducer_gating
        self.gumbel_tau = float(gumbel_tau)
        self._codebook_lock_after = codebook_lock_after
        self._transducer_gating_schedule = transducer_gating_schedule
        if transducer_gating_schedule is not None and transducer_gating_target is None:
            raise ValueError("transducer_gating_schedule requires transducer_gating_target")
        self._transducer_gating_target = transducer_gating_target
        self.register_buffer("codebook_step", torch.tensor(0, dtype=torch.long))
        object.__setattr__(self, "mux", mux)
        self._lesion_log: list[tuple[Modality, float]] = []

    def step(self) -> None:
        """Advance the nerve-level plasticity clock (Sprint 13).

        Consumers call this once per training iteration. When
        ``codebook_lock_after`` is configured and the counter crosses
        that threshold, freezes the ``AdaptiveCodebook`` parameter —
        an analogue of the nerve-wml #4 constellation lock for the
        P2 plasticity component.
        """
        self.codebook_step += 1
        if (
            self._codebook_lock_after is not None
            and int(self.codebook_step) >= self._codebook_lock_after
        ):
            self.codebook.codebook.requires_grad_(False)

    def _active_gating(self) -> str:
        """Return the gating mode active at the current ``codebook_step``.

        Functional (no state mutation) so a checkpoint restore of
        ``codebook_step`` transparently resumes the correct mode on
        the very first forward pass of Phase 2.
        """
        if (
            self._transducer_gating_schedule is not None
            and self._transducer_gating_target is not None
            and int(self.codebook_step) >= self._transducer_gating_schedule
        ):
            return self._transducer_gating_target
        return self.transducer_gating

    def fuse(self, letters: dict[Modality, Tensor]) -> Tensor:
        """Fuse 5 modality carriers into a shared `(B, K, d_hidden)` tensor.

        Args:
            letters: 5-entry dict mapping each Modality to its `(B, T)` carrier.

        Returns:
            fused: `(B, K, d_hidden)` float32. Gradient flows back through
            the soft demod (Gumbel-softmax per Q2 arbitration) to the mux
            constellation AND through the codebook AND through the gates
            AND through every active transducer.
        """
        # 1. Soft demod each carrier to (B, K, A) distribution.
        soft = {m: self.mux.demodulate(letters[m], hard=False, tau=1.0) for m in MODALITIES}
        # 2. Project each modality via the adaptive codebook -> (B, K, d_hidden).
        h: dict[Modality, Tensor] = {m: self.codebook(soft[m]) for m in MODALITIES}

        # 3. Per-pair transducer activation per spec §4.3.
        w = self.gates.weights()  # (5,)
        gate_vals = [w[i].item() for i, _ in enumerate(MODALITIES)]
        use_gumbel = self._active_gating() == "gumbel"
        for src_idx, src in enumerate(MODALITIES):
            for dst_idx, dst in enumerate(MODALITIES):
                if src == dst:
                    continue
                transducer = self.transducers[f"{src}->{dst}"]
                if use_gumbel:
                    # Gumbel-softmax-free soft gate: sigmoid of the (dst - src)
                    # gate margin, smoothed by gumbel_tau. Continuous
                    # analogue of the binary v0.5 rule, preserves the
                    # same selectivity direction (src low -> dst high
                    # increases activation).
                    margin = (gate_vals[dst_idx] - 0.3) - (gate_vals[src_idx] - 0.1)
                    soft_coeff = float(torch.sigmoid(torch.tensor(margin / self.gumbel_tau)).item())
                    h[dst] = h[dst] + transducer(h[src], soft_activate=soft_coeff)
                else:
                    active = gate_vals[src_idx] < 0.1 and gate_vals[dst_idx] > 0.3
                    if active:
                        h[dst] = h[dst] + transducer(h[src], active=True)

        # 4. Gate-weighted sum across modalities.
        fused = torch.zeros_like(h[MODALITIES[0]])
        for i, m in enumerate(MODALITIES):
            fused = fused + w[i] * h[m]
        return fused

    def on_lesion(self, modality: Modality, snr_db: float, *, bias: float = -1.0) -> None:
        """Record a lesion event and nudge the affected gate alpha downward.

        The `bias` kwarg (default -1.0) is subtracted from the lesioned
        modality's alpha, roughly halving its softmax weight. This speeds
        up the Phase 2 compensation dynamics so the 100-step gate
        trajectory test (Task 2.11) can detect a clear shift. Set
        `bias=0.0` to disable the nudge and rely purely on gradient.
        """
        idx = MODALITIES.index(modality)
        with torch.no_grad():
            self.gates.alpha[idx] += bias
        self._lesion_log.append((modality, snr_db))

    def migration_stats(self) -> MigrationReport:
        """Snapshot the nerve's current plasticity state."""
        w = self.gates.weights().detach()
        gate_vals = {m: float(w[i].item()) for i, m in enumerate(MODALITIES)}

        # Codebook "utilisation entropy" over row-norm softmax. Max = log(A).
        row_norms = self.codebook.codebook.norm(dim=-1)
        probs = row_norms.softmax(dim=-1)
        entropy = float(-(probs * probs.clamp(min=1e-12).log()).sum().item())

        # Count transducers that WOULD fire under current gate values.
        active = 0
        for src in MODALITIES:
            for dst in MODALITIES:
                if src == dst:
                    continue
                if gate_vals[src] < 0.1 and gate_vals[dst] > 0.3:
                    active += 1

        return MigrationReport(
            gate_values=gate_vals,
            codebook_entropy=entropy,
            transducer_active_count=active,
            lesion_events=tuple(self._lesion_log),
        )


@dataclass(frozen=True)
class MigrationReport:
    """Per-step snapshot of CrossModalNerve plasticity state (Task 2.5)."""

    gate_values: dict[Modality, float]
    codebook_entropy: float
    transducer_active_count: int
    lesion_events: tuple[tuple[Modality, float], ...]
