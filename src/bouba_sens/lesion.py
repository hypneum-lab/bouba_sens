"""Lesion protocol per spec §3.4 and §4 — LesionSpec + SNR schedules.

Sprint 2 lands the M2 (progressive SNR degradation) protocol as the v0.1
default, with `LesionSpec` as the frozen container and `m2_snr_schedule`
as the canonical schedule callable. `LesionScheduler.apply` (Task 2.7)
consumes both.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import torch

from bouba_sens.sensory import Modality
from bouba_sens.world.base import WorldSample

LesionMode = Literal["M1", "M2", "M3", "M4"]
LesionTiming = Literal["T1", "T2", "T3", "T4"]


@dataclass(frozen=True)
class LesionSpec:
    """Immutable lesion configuration per spec §3.4.

    - `modality`: which of the 5 sensory channels is lesioned.
    - `mode`: lesion mode (v0.1 defaults to M2 progressive SNR).
    - `timing`: T1 congenital (Phase 1 skipped) vs T2 late-acquired
      (intact pretraining, then lesion) vs T3/T4 reserved.
    - `schedule`: callable `step -> SNR_dB` driving the per-step noise level.
    """

    modality: Modality
    mode: LesionMode
    timing: LesionTiming
    schedule: Callable[[int], float]


def m2_snr_schedule(
    step: int,
    *,
    snr_init: float = 20.0,
    snr_floor: float = -20.0,
    k: int = 5000,
) -> float:
    """M2 progressive SNR degradation per spec §4.1.

    SNR(t) = max(SNR_floor, SNR_init + (SNR_floor - SNR_init) · min(1, t / K))

    The schedule linearly interpolates from SNR_init at step=0 down to
    SNR_floor at step=K, then clamps at SNR_floor for all subsequent
    steps. v0.1 defaults: +20 dB -> -20 dB over 5000 steps.
    """
    ramp = min(1.0, step / k) if k > 0 else 1.0
    return max(snr_floor, snr_init + (snr_floor - snr_init) * ramp)


class LesionScheduler:
    """Injects SNR-scaled additive Gaussian noise onto the targeted modality.

    `apply(sample, step)` pulls the current SNR from `spec.schedule(step)`,
    scales Gaussian noise to `signal_std · 10^(-SNR/20)`, adds it to the
    lesioned modality, and returns a NEW frozen `WorldSample` via
    `dataclasses.replace`. Other modalities and the label are returned
    byte-identical (same tensor objects) so the rest of the pipeline sees
    only the targeted corruption.

    The noise is *relative* to the modality's running signal magnitude so
    the SNR definition is preserved across worlds with different natural
    scales (spec §4.1).
    """

    def __init__(self, spec: LesionSpec) -> None:
        self.spec = spec

    def apply(self, sample: WorldSample, step: int) -> WorldSample:
        snr_db = self.spec.schedule(step)
        noise_factor = 10.0 ** (-snr_db / 20.0)
        tensor = getattr(sample, self.spec.modality)
        signal_std = tensor.std().item()
        noise = torch.randn_like(tensor) * signal_std * noise_factor
        lesioned = tensor + noise
        return replace(sample, **{self.spec.modality: lesioned})
