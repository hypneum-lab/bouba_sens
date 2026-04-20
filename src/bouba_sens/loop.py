"""AdaptationLoop — owns pretrain/lesion/eval phases and the theta-replay buffer.

Sprint 2 Tasks 2.9-2.10 per spec §3.6. Phase 1 (`pretrain`) wires the
full forward pass: WorldSimulator sample -> 5 SensoryWMLs -> carriers
-> CrossModalNerve.fuse -> IntegrationHead -> cross-entropy on label.
Phase 2 (`lesion_phase`) adds LesionScheduler injection + theta-replay
FIFO buffer + migration snapshots.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from bouba_sens.nerve import CrossModalNerve, MigrationReport
from bouba_sens.sensory import MODALITIES, Modality, SensoryWML

if TYPE_CHECKING:
    from track_p.multiplexer import (  # type: ignore[import-not-found]
        GammaThetaMultiplexer,
    )

    from bouba_sens.head import IntegrationHead
    from bouba_sens.lesion import LesionSpec
    from bouba_sens.world.base import WorldSample, WorldSimulator


@dataclass
class Checkpoint:
    """Deep-copied module states snapshotted at Phase 1 end."""

    mux_state: dict[str, Tensor]
    nerve_state: dict[str, Tensor]
    head_state: dict[str, Tensor]
    sensory_states: dict[Modality, dict[str, Tensor]]


@dataclass
class AdaptationReport:
    """Per-step + per-snapshot trajectory produced by `lesion_phase`."""

    loss_curve: list[float] = field(default_factory=list)
    accuracy_curve: list[float] = field(default_factory=list)
    gate_trajectory: list[dict[Modality, float]] = field(default_factory=list)
    codebook_entropy_trajectory: list[float] = field(default_factory=list)
    transducer_activation_trajectory: list[int] = field(default_factory=list)
    lesion_events: list[tuple[Modality, float]] = field(default_factory=list)

    # Sprint 5 / Task 5.1 — probe-batch captures for Me3 delta. Codes
    # are the mean-pooled fused representation (shape (B,)) taken on
    # the SAME probe sample pre- vs post-lesion training; labels are
    # the probe labels (shape (B,)). Both in clean-input / stressed-net
    # framing: the pre capture happens before on_lesion fires, the post
    # capture after Phase 2 training completes.
    probe_labels: Tensor | None = None
    pre_lesion_codes: Tensor | None = None
    post_lesion_codes: Tensor | None = None


def _deepcopy_state(module: Any) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


class AdaptationLoop:
    """Owns Phase 1 pretrain and Phase 2 lesion_phase training loops.

    Instantiated with all building blocks as pre-constructed instances
    (world, mux, 5 SensoryWMLs, nerve, head). An Adam optimiser is built
    over every parameter once, avoiding the double-count that would occur
    if `mux.parameters()` appeared inside each SensoryWML or inside
    `CrossModalNerve` — both use `object.__setattr__` bypasses, so
    iterating mux + nerve + head + each sensory gives disjoint param sets.
    """

    def __init__(
        self,
        world: WorldSimulator,
        mux: GammaThetaMultiplexer,
        sensories: dict[Modality, SensoryWML],
        nerve: CrossModalNerve,
        head: IntegrationHead,
        *,
        lr: float = 1e-3,
    ) -> None:
        self.world = world
        self.mux = mux
        self.sensories = sensories
        self.nerve = nerve
        self.head = head

        params: list[Tensor] = []
        params += list(mux.parameters())
        params += list(nerve.parameters())
        params += list(head.parameters())
        for s in sensories.values():
            params += list(s.parameters())
        self.opt = torch.optim.Adam(params, lr=lr)

    def _forward(self, sample: WorldSample) -> tuple[Tensor, Tensor]:
        carriers = {m: self.sensories[m].step(getattr(sample, m)) for m in MODALITIES}
        fused = self.nerve.fuse(carriers)
        logits = self.head(fused)
        loss = F.cross_entropy(logits, sample.label)
        return loss, logits

    def pretrain(self, steps: int, *, batch_size: int = 32, seed: int = 0) -> Checkpoint:
        """Phase 1: intact training on the world. Returns snapshot."""
        for step in range(steps):
            sample = self.world.sample(batch_size=batch_size, seed=seed + step)
            loss, _ = self._forward(sample)
            self.opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            self.opt.step()
        return Checkpoint(
            mux_state=_deepcopy_state(self.mux),
            nerve_state=_deepcopy_state(self.nerve),
            head_state=_deepcopy_state(self.head),
            sensory_states={m: _deepcopy_state(self.sensories[m]) for m in MODALITIES},
        )

    def restore(self, ckpt: Checkpoint) -> None:
        """Restore every module state from a Checkpoint (tests + Phase 2)."""
        self.mux.load_state_dict(ckpt.mux_state)
        self.nerve.load_state_dict(ckpt.nerve_state)
        self.head.load_state_dict(ckpt.head_state)
        for m in MODALITIES:
            self.sensories[m].load_state_dict(ckpt.sensory_states[m])

    def lesion_phase(
        self,
        lesion: LesionSpec,
        steps: int,
        *,
        batch_size: int = 32,
        replay_buffer_size: int = 1024,
        stats_every: int = 10,
        seed: int = 10_000,
    ) -> AdaptationReport:
        """Phase 2: lesion-active training + theta-replay buffer.

        Samples are lesioned via `LesionScheduler.apply` before SensoryWML
        consumption. `on_lesion` fires once at t=0. Migration stats are
        captured every `stats_every` steps.

        Replay buffer is a FIFO of raw unlesioned `WorldSample` instances
        drawn at Phase-2 entry; half of each step's batch is taken from
        the buffer so the model does not over-fit the lesion distribution.
        """
        from bouba_sens.lesion import LesionScheduler  # local import avoids cycle

        scheduler = LesionScheduler(lesion)

        # Capture baseline BEFORE on_lesion so gate_trajectory[0] reflects
        # the pre-lesion (uniform) state; acceptance tests compare [0] vs [-1]
        # to detect compensation dynamics.
        baseline = self.nerve.migration_stats()
        report = AdaptationReport()
        report.gate_trajectory.append(baseline.gate_values)
        report.codebook_entropy_trajectory.append(baseline.codebook_entropy)
        report.transducer_activation_trajectory.append(baseline.transducer_active_count)

        # Task 5.1 — probe-batch capture for Me3 delta. Same probe sample
        # is fed clean through the intact (pre-lesion) network and again
        # through the lesion-adapted network after training. Mean-pool the
        # fused representation to a 1-D scalar per sample so Me3 kNN MI
        # accepts it as `codes` directly. Seed stepped far from the Phase-2
        # training seeds to avoid accidental overlap.
        probe = self.world.sample(batch_size=batch_size, seed=seed - 99_999)
        with torch.no_grad():
            carriers_pre = {m: self.sensories[m].step(getattr(probe, m)) for m in MODALITIES}
            fused_pre = self.nerve.fuse(carriers_pre)
        report.probe_labels = probe.label.detach().clone()
        # flatten(1).mean(-1) collapses any (B, *) fused shape to (B,).
        report.pre_lesion_codes = fused_pre.flatten(1).mean(dim=-1).detach().clone()

        self.nerve.on_lesion(lesion.modality, lesion.schedule(0))
        report.lesion_events = list(self.nerve._lesion_log)

        replay_buffer: list[WorldSample] = [
            self.world.sample(batch_size=batch_size, seed=seed - 1 - i)
            for i in range(min(replay_buffer_size, 16))
        ]

        def replay_draw(i: int) -> WorldSample:
            return replay_buffer[i % len(replay_buffer)]

        for step in range(steps):
            fresh = self.world.sample(batch_size=batch_size, seed=seed + step)
            lesioned = scheduler.apply(fresh, step)
            loss_fresh, logits = self._forward(lesioned)

            # Replay a clean batch with half weight (theta-phase signal).
            replay_sample = replay_draw(step)
            loss_replay, _ = self._forward(replay_sample)
            loss = loss_fresh + 0.5 * loss_replay

            self.opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            self.opt.step()

            report.loss_curve.append(loss.item())
            preds = logits.argmax(-1)
            report.accuracy_curve.append((preds == lesioned.label).float().mean().item())

            if step % stats_every == 0 or step == steps - 1:
                snap: MigrationReport = self.nerve.migration_stats()
                report.gate_trajectory.append(snap.gate_values)
                report.codebook_entropy_trajectory.append(snap.codebook_entropy)
                report.transducer_activation_trajectory.append(snap.transducer_active_count)

            # FIFO: update buffer with a fresh *clean* sample at the tail,
            # evict the oldest.
            if replay_buffer_size > 0:
                replay_buffer.append(
                    self.world.sample(batch_size=batch_size, seed=seed - steps - step - 1)
                )
                if len(replay_buffer) > replay_buffer_size:
                    replay_buffer.pop(0)

        # Task 5.1 — post-lesion probe capture. Feed the same clean probe
        # batch through the lesion-adapted network; mean-pool to (B,) 1-D.
        with torch.no_grad():
            carriers_post = {m: self.sensories[m].step(getattr(probe, m)) for m in MODALITIES}
            fused_post = self.nerve.fuse(carriers_post)
        report.post_lesion_codes = fused_post.flatten(1).mean(dim=-1).detach().clone()

        return report

    def query_accuracy(
        self,
        query_modality: Modality,
        *,
        batch_size: int = 64,
        seed: int = 987_654,
    ) -> float:
        """Accuracy on a clean probe when only `query_modality` carries signal.

        Sprint 5 / Task 5.2 — builds the 5x5 Me6 perf matrix after
        Phase 2. All four non-query modalities are zeroed out on the
        sample (in-place style via a fresh dataclass) so the network
        must produce its prediction from a single modality. Result is
        a float in [0, 1].
        """
        sample = self.world.sample(batch_size=batch_size, seed=seed)
        masked_tensors: dict[str, Tensor] = {}
        for m in MODALITIES:
            tensor = getattr(sample, m)
            masked_tensors[m] = tensor if m == query_modality else torch.zeros_like(tensor)
        masked = replace(
            sample,
            audio=masked_tensors["audio"],
            vision=masked_tensors["vision"],
            tactile=masked_tensors["tactile"],
            gravity=masked_tensors["gravity"],
            force=masked_tensors["force"],
        )
        with torch.no_grad():
            _, logits = self._forward(masked)
            preds = logits.argmax(-1)
            return float((preds == sample.label).float().mean().item())

    # Convenience hook so Checkpoint round-trips in tests.
    def snapshot(self) -> Checkpoint:
        return copy.deepcopy(
            Checkpoint(
                mux_state=_deepcopy_state(self.mux),
                nerve_state=_deepcopy_state(self.nerve),
                head_state=_deepcopy_state(self.head),
                sensory_states={m: _deepcopy_state(self.sensories[m]) for m in MODALITIES},
            )
        )
