"""Probe capture + common Phase-2 schedule for the v0.8 subspace shift.

Amendment v0.8 section 3 needs, for every (seed ``s``, lock ``l``) block,

* one fixed clean probe of 512 samples (``world.sample(512, seed=
  phase2_seed - 99_999)``), decoupled from the training batch of 16;
* ``Z_pre`` captured once at Phase-2 entry from the Phase-1 checkpoint;
* ``Z_post`` for each of the 15 lesion cells (5 modalities x 3 SNR floors);
* the no-lesion control's end matrix, run on the **same** Phase-1
  checkpoint, probe and Phase-2 data seeds.

``capture_block`` runs all of these arms from fresh loops restored from
one checkpoint (so Adam state and module counters start identically)
under one ``phase2_seed``. Design choices that the amendment leaves open
are documented on the function and listed in the PR description.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from bouba_sens.lesion import LesionSpec, m2_snr_schedule
from bouba_sens.loop import AdaptationLoop, Checkpoint
from bouba_sens.sensory import MODALITIES, Modality

PROBE_SIZE = 512
TRAIN_BATCH_SIZE = 16
STEPS_LESION = 100
SNR_FLOORS: tuple[float, ...] = (-20.0, -10.0, 10.0)
# Amendment v0.8 section 4 (ADR-0020, PI decision 2026-10-23): the M2 ramp
# reaches each floor at lesion step 50 and holds it to step 99.
K_STEPS = 50
# Amendment v0.8 section 3 item 2 (PI decision 2026-10-25, after the v0.7
# B-2 pilot): the probe captures use the Gumbel readout. The hard (argmax)
# path mapped every probe row to one code (rank 0).
PROBE_DEMOD = "soft"
# PI decision 2026-10-25 (13:12): each Gumbel capture is the mean of K = 8
# Gumbel draws under fixed seeds (``bouba_sens.readout``).
PROBE_GUMBEL_DRAWS = 8

CellKey = tuple[Modality, float]  # (lesioned modality, SNR floor in dB)


@dataclass
class BlockCodes:
    """Raw code matrices of one (seed, lock) block, before any rank check."""

    seed: int
    lock: int
    phase2_seed: int
    probe_labels: Tensor
    z_pre: Tensor
    z_control: Tensor
    z_post: dict[CellKey, Tensor] = field(default_factory=dict)


def capture_block(
    make_loop: Callable[[], AdaptationLoop],
    ckpt: Checkpoint,
    *,
    seed: int,
    lock: int,
    modalities: Sequence[Modality] = MODALITIES,
    snr_floors: Sequence[float] = SNR_FLOORS,
    steps: int = STEPS_LESION,
    batch_size: int = TRAIN_BATCH_SIZE,
    probe_size: int = PROBE_SIZE,
    phase2_seed: int | None = None,
    snr_init: float = 20.0,
    k_steps: int = K_STEPS,
    probe_demod: str = PROBE_DEMOD,
    probe_gumbel_draws: int | None = None,
) -> BlockCodes:
    """Run the control arm and every lesion cell of one (seed, lock) block.

    ``make_loop`` builds a fresh, identically configured loop (including the
    ``constellation_lock_after=lock`` mux); each arm restores ``ckpt`` into
    its own loop. All arms use the same ``phase2_seed`` (default: the
    experiment ``seed``), hence the same probe, the same fresh and replay
    batches, and Gumbel-readout captures (``probe_demod="soft"``, PI
    decision 2026-10-25; ``"hard"`` is kept only for diagnostics), each
    the mean of ``PROBE_GUMBEL_DRAWS`` = 8 draws seeded by
    ``readout.gumbel_draw_seed(phase2_seed, phase, r)``.
    Before each arm the global torch RNG is re-seeded with
    ``phase2_seed``, so the lesion noise stream and the Gumbel draws are
    reproducible, and ``Z_pre`` (captured before any Phase-2 step) is
    common to all arms. ``lock`` is recorded; the lock itself is applied by
    ``make_loop`` / the checkpoint.

    Raises ``RuntimeError`` if ``Z_pre`` differs between arms (the shared
    Phase-2 entry state would not be shared, a pipeline defect).
    """
    p2_seed = seed if phase2_seed is None else phase2_seed
    k_draws = (
        (PROBE_GUMBEL_DRAWS if probe_demod == "soft" else 0)
        if probe_gumbel_draws is None
        else probe_gumbel_draws
    )
    probe_seed = p2_seed - 99_999

    def run(spec: LesionSpec | None):  # type: ignore[no-untyped-def]
        loop = make_loop()
        loop.restore(ckpt)
        torch.manual_seed(p2_seed)
        return loop.lesion_phase(
            spec,
            steps,
            batch_size=batch_size,
            seed=p2_seed,
            probe_size=probe_size,
            probe_seed=probe_seed,
            probe_demod=probe_demod,
            probe_gumbel_draws=k_draws,
        )

    control = run(None)
    assert control.pre_lesion_codes is not None and control.post_lesion_codes is not None
    assert control.probe_labels is not None
    block = BlockCodes(
        seed=seed,
        lock=lock,
        phase2_seed=p2_seed,
        probe_labels=control.probe_labels,
        z_pre=control.pre_lesion_codes,
        z_control=control.post_lesion_codes,
    )
    for modality in modalities:
        for floor in snr_floors:

            def schedule(step: int, _floor: float = floor) -> float:
                return m2_snr_schedule(step, snr_init=snr_init, snr_floor=_floor, k=k_steps)

            spec = LesionSpec(modality=modality, mode="M2", timing="T2", schedule=schedule)
            rep = run(spec)
            assert rep.pre_lesion_codes is not None and rep.post_lesion_codes is not None
            if not torch.equal(rep.pre_lesion_codes, block.z_pre):
                raise RuntimeError(
                    f"Z_pre differs between arms for ({modality}, {floor}) at seed={seed}, "
                    f"lock={lock}: the Phase-2 entry state is not shared"
                )
            block.z_post[(modality, floor)] = rep.post_lesion_codes
    return block
