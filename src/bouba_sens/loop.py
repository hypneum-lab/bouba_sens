"""AdaptationLoop — owns pretrain/lesion/eval phases and the theta-replay buffer.

Sprint 2 Tasks 2.9-2.10 per spec §3.6. Phase 1 (`pretrain`) wires the
full forward pass: WorldSimulator sample -> 5 SensoryWMLs -> carriers
-> CrossModalNerve.fuse -> IntegrationHead -> cross-entropy on label.
Phase 2 (`lesion_phase`) adds LesionScheduler injection + theta-replay
FIFO buffer + migration snapshots.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from bouba_sens.linear_probe import ProbeFit, fit_linear_probe
from bouba_sens.nerve import CrossModalNerve, MigrationReport
from bouba_sens.sensory import MODALITIES, Modality, SensoryWML

if TYPE_CHECKING:
    from track_p.multiplexer import GammaThetaMultiplexer

    from bouba_sens.decorrelation import DecorrelationSchedule
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
    # are the FULL fused representation (shape (B, K*d), default
    # probe_pool="full") taken on the SAME probe sample pre- vs
    # post-lesion training; with probe_pool="mean" they collapse to the
    # legacy 1-D scalar (B,). Labels are the probe labels (shape (B,)).
    # Both in clean-input / stressed-net framing: the pre capture happens
    # before on_lesion fires, the post capture after Phase 2 completes.
    probe_labels: Tensor | None = None
    pre_lesion_codes: Tensor | None = None
    post_lesion_codes: Tensor | None = None


def _deepcopy_state(module: Any) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


# ADR-0023 — probe v2 sizes (held-out test items, training pool).
PROBE_V2_TEST_SIZE = 256
PROBE_V2_TRAIN_SIZE = 512


@dataclass(frozen=True)
class ProbeV2Result:
    """One modality's v2 probe: the linear fit plus two diagnostics.

    `n_train_dropped` counts training-pool items removed because their raw
    inputs equal a test item's; `n_codes_test` is the number of distinct
    argmax codes the modality emits on the test items (1 = the code channel
    the nerve sees is collapsed, whatever the hidden state carries).
    """

    fit: ProbeFit
    n_train_dropped: int
    n_codes_test: int


def _rows_not_in(pool: WorldSample, test: WorldSample) -> np.ndarray:
    """Boolean mask over `pool` items whose raw inputs differ from every test item."""

    def _rows(sample: WorldSample) -> list[bytes]:
        flat = torch.cat(
            [getattr(sample, m).reshape(sample.label.shape[0], -1) for m in MODALITIES], dim=1
        )
        return [row.numpy().tobytes() for row in flat]

    test_rows = set(_rows(test))
    return np.array([row not in test_rows for row in _rows(pool)], dtype=bool)


def successes_to_accuracy(successes: Sequence[int]) -> float:
    """Mean of 0/1 outcomes, computed exactly as `query_accuracy` always did.

    ADR-0022 — `query_accuracy` used to return
    `float((preds == label).float().mean().item())`: a float32 mean of the
    0/1 outcomes, widened to a Python float. Taking the same float32 mean
    here (rather than `sum / len` in double precision) keeps the result
    bit-identical to that path for every batch size: the partial sums of
    0/1 values are exact integers in float32, so only the final division
    rounds, and it rounds identically in both paths.
    """
    if len(successes) == 0:
        raise ValueError("successes must be non-empty")
    return float(torch.tensor(successes, dtype=torch.float32).mean().item())


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

    def pretrain(
        self,
        steps: int,
        *,
        batch_size: int = 32,
        seed: int = 0,
        decorrelation: DecorrelationSchedule | None = None,
    ) -> Checkpoint:
        """Phase 1: intact training on the world. Returns snapshot.

        Calls `mux.step()` once per iteration so any
        `constellation_lock_after` configured at mux construction
        can fire during Phase 1 (congenital vs late-acquired
        critical-period semantics, nerve-wml#4).

        `decorrelation` (v0.9) swaps the batch source inside its window;
        None keeps the original behaviour.
        """
        mux_step = getattr(self.mux, "step", None)
        nerve_step = getattr(self.nerve, "step", None)
        for step in range(steps):
            if decorrelation is None:
                sample = self.world.sample(batch_size=batch_size, seed=seed + step)
            else:
                sample = decorrelation.sample(
                    self.world,  # type: ignore[arg-type]
                    batch_size=batch_size,
                    seed=seed + step,
                    step=step,
                )
            loss, _ = self._forward(sample)
            self.opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            self.opt.step()
            if callable(mux_step):
                mux_step()
            if callable(nerve_step):
                nerve_step()
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
        lesion: LesionSpec | None,
        steps: int,
        *,
        batch_size: int = 32,
        replay_buffer_size: int = 1024,
        stats_every: int = 10,
        seed: int = 10_000,
        probe_pool: str = "full",
        probe_size: int | None = None,
        probe_seed: int | None = None,
        probe_demod: str = "soft",
    ) -> AdaptationReport:
        """Phase 2: lesion-active training + theta-replay buffer.

        Samples are lesioned via `LesionScheduler.apply` before SensoryWML
        consumption. `on_lesion` fires once at t=0. Migration stats are
        captured every `stats_every` steps.

        Replay buffer is a FIFO of raw unlesioned `WorldSample` instances
        drawn at Phase-2 entry; half of each step's batch is taken from
        the buffer so the model does not over-fit the lesion distribution.

        Amendment v0.8 (subspace shift): ``probe_size`` decouples the size
        of the clean probe from the training ``batch_size`` (``None`` keeps
        the legacy ``batch_size`` probe; the amendment uses 512 with batch
        16) and ``probe_seed`` overrides the probe draw (default
        ``seed - 99_999``). ``lesion=None`` runs the **no-lesion control
        arm**: identical data schedule (same fresh and replay batches for
        the same ``seed``), no ``on_lesion``, no noise injection, so that
        ``Z_post`` shows the shift caused by continued training alone.

        ``fuse`` demodulates through Gumbel-softmax by default, which draws
        from the global torch RNG, so probe captures are stochastic
        (legacy behaviour, ``probe_demod="soft"``). ``probe_demod="hard"``
        captures both Z_pre and Z_post with deterministic argmax
        demodulation (``fuse(hard_demod=True)``): same checkpoint, same
        codes, no RNG use. Training steps are unaffected either way.
        """
        from bouba_sens.lesion import LesionScheduler  # local import avoids cycle

        scheduler = LesionScheduler(lesion) if lesion is not None else None

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
        # through the lesion-adapted network after training. The fused
        # representation is retained at FULL dimensionality (B, K*d) so the
        # MI estimators (Kraskov-KSG, MINE) see the true dimensionality of
        # the surviving channel. `probe_pool="mean"` restores the legacy
        # 1-D mean-pooled scalar (Sprint 5..17) for reproducibility.
        # Seed stepped far from the Phase-2 training seeds to avoid overlap.
        if probe_demod not in ("soft", "hard"):
            raise ValueError(f"probe_demod must be 'soft' or 'hard'; got {probe_demod!r}")

        def _capture(sample: WorldSample) -> Tensor:
            with torch.no_grad():
                carriers = {m: self.sensories[m].step(getattr(sample, m)) for m in MODALITIES}
                return self.nerve.fuse(carriers, hard_demod=probe_demod == "hard")

        def _pool(fused: Tensor) -> Tensor:
            flat = fused.flatten(1)  # (B, K*d)
            if probe_pool == "mean":
                return flat.mean(dim=-1)  # (B,) legacy scalar sonde
            return flat  # (B, K*d) full sonde

        probe = self.world.sample(
            batch_size=batch_size if probe_size is None else probe_size,
            seed=seed - 99_999 if probe_seed is None else probe_seed,
        )
        fused_pre = _capture(probe)
        report.probe_labels = probe.label.detach().clone()
        report.pre_lesion_codes = _pool(fused_pre).detach().clone()

        if lesion is not None:
            self.nerve.on_lesion(lesion.modality, lesion.schedule(0))
            report.lesion_events = list(self.nerve._lesion_log)

        replay_buffer: list[WorldSample] = [
            self.world.sample(batch_size=batch_size, seed=seed - 1 - i)
            for i in range(min(replay_buffer_size, 16))
        ]

        def replay_draw(i: int) -> WorldSample:
            return replay_buffer[i % len(replay_buffer)]

        mux_step = getattr(self.mux, "step", None)
        nerve_step = getattr(self.nerve, "step", None)
        for step in range(steps):
            fresh = self.world.sample(batch_size=batch_size, seed=seed + step)
            lesioned = scheduler.apply(fresh, step) if scheduler is not None else fresh
            loss_fresh, logits = self._forward(lesioned)

            # Replay a clean batch with half weight (theta-phase signal).
            replay_sample = replay_draw(step)
            loss_replay, _ = self._forward(replay_sample)
            loss = loss_fresh + 0.5 * loss_replay

            self.opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            self.opt.step()
            if callable(mux_step):
                mux_step()
            if callable(nerve_step):
                nerve_step()

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
        # batch through the lesion-adapted network; pooled identically to
        # the pre-lesion capture (full (B, K*d) by default).
        fused_post = _capture(probe)
        report.post_lesion_codes = _pool(fused_post).detach().clone()

        return report

    def query_successes(
        self,
        query_modality: Modality,
        *,
        batch_size: int = 64,
        seed: int = 987_654,
    ) -> list[int]:
        """Per-item outcomes (1 = correct) on a clean probe with a single live modality.

        ADR-0022 — the per-sample record behind `query_accuracy`. The probe
        batch is `self.world.sample(batch_size=batch_size, seed=seed)`; the
        four non-query modalities are zeroed out on a fresh dataclass so the
        network must predict from `query_modality` alone. Item `i` of the
        returned list is the outcome on probe item `i`, so two calls with the
        same `seed` and `batch_size` (e.g. the five query modalities of one
        lesion cell) are scored on the same probe items and can be paired
        item by item. The list has exactly `batch_size` entries.

        The forward pass draws from the global torch RNG (the modules stay
        in training mode), so repeated calls reproduce each other only from
        the same RNG state; the probe items themselves depend on `seed` only.
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
            return [int(v) for v in (preds == sample.label).tolist()]

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

        ADR-0022 — defined as the mean of `query_successes` (same probe,
        same masking); see `successes_to_accuracy` for why the mean is
        taken in float32, which keeps the value bit-identical to the
        pre-refactor tensor path for every `batch_size`.
        """
        return successes_to_accuracy(
            self.query_successes(query_modality, batch_size=batch_size, seed=seed)
        )

    # ------------------------------------------------------------------
    # ADR-0023 — probe v2: linear read-out of each modality's own frozen
    # representation. The v1 methods above are left byte-for-byte intact.
    # ------------------------------------------------------------------

    def modality_features(self, modality: Modality, x: Tensor) -> Tensor:
        """Frozen pre-quantisation representation of one modality, `(B, d_hidden)`.

        ADR-0023 — the hidden state `core(input_proj(x))` that
        `SensoryWML.step` computes before the argmax over the 64-code
        alphabet. It depends on `x` (this modality's clean input) only, so
        no other modality is touched and the pathway sees exactly the kind
        of input it was trained on. Deterministic: no dropout, no sampling,
        no global RNG use; computed under `no_grad`.
        """
        sensory = self.sensories[modality]
        with torch.no_grad():
            return cast(Tensor, sensory.core(sensory.input_proj(x))).detach()

    def probe_v2(
        self,
        *,
        seed: int = 987_654,
        test_size: int = PROBE_V2_TEST_SIZE,
        train_size: int = PROBE_V2_TRAIN_SIZE,
        modalities: Sequence[Modality] = MODALITIES,
    ) -> dict[Modality, ProbeV2Result]:
        """Linear probe per modality on frozen representations (ADR-0023).

        Held-out test items: `world.sample(test_size, seed=seed)`; training
        pool: `world.sample(train_size, seed=seed + 1)`, minus any item whose
        raw inputs (all five modalities) equal a test item's (contiguous-
        window worlds can overlap). For each modality, a cross-validated
        logistic regression (`linear_probe.fit_linear_probe`) is fit on the
        training pool's frozen features and scored item by item on the test
        items. Item `i` is the same world item for every modality, and for
        every model probed with the same `seed`.
        """
        test = self.world.sample(batch_size=test_size, seed=seed)
        pool = self.world.sample(batch_size=train_size, seed=seed + 1)
        keep = _rows_not_in(pool, test)
        y_test = test.label.numpy()
        y_train = pool.label.numpy()[keep]
        results: dict[Modality, ProbeV2Result] = {}
        for m in modalities:
            f_test = self.modality_features(m, getattr(test, m))
            f_train = self.modality_features(m, getattr(pool, m))[torch.from_numpy(keep)]
            fit = fit_linear_probe(f_train.numpy(), y_train, f_test.numpy(), y_test)
            with torch.no_grad():
                codes = self.sensories[m].emit_head_pi(f_test).argmax(-1)
            results[m] = ProbeV2Result(
                fit=fit,
                n_train_dropped=int(train_size - int(keep.sum())),
                n_codes_test=int(codes.unique().numel()),
            )
        return results

    def query_successes_v2(
        self,
        query_modality: Modality,
        *,
        seed: int = 987_654,
        test_size: int = PROBE_V2_TEST_SIZE,
        train_size: int = PROBE_V2_TRAIN_SIZE,
    ) -> list[int]:
        """Per-item outcomes of the v2 linear probe on `query_modality` (ADR-0023)."""
        result = self.probe_v2(
            seed=seed, test_size=test_size, train_size=train_size, modalities=(query_modality,)
        )
        return list(result[query_modality].fit.successes)

    def query_accuracy_v2(
        self,
        query_modality: Modality,
        *,
        seed: int = 987_654,
        test_size: int = PROBE_V2_TEST_SIZE,
        train_size: int = PROBE_V2_TRAIN_SIZE,
    ) -> float:
        """Held-out accuracy of the v2 linear probe (mean of `query_successes_v2`)."""
        return successes_to_accuracy(
            self.query_successes_v2(
                query_modality, seed=seed, test_size=test_size, train_size=train_size
            )
        )

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
