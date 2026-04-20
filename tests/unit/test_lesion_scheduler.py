"""Task 2.7 acceptance tests for `LesionScheduler.apply`."""

from __future__ import annotations

import torch

from bouba_sens.lesion import LesionScheduler, LesionSpec, m2_snr_schedule
from bouba_sens.world.gaussian import GaussianWorld


def _make_spec(modality: str = "audio") -> LesionSpec:
    return LesionSpec(
        modality=modality,  # type: ignore[arg-type]
        mode="M2",
        timing="T2",
        schedule=m2_snr_schedule,
    )


def test_apply_modifies_only_lesioned_modality() -> None:
    """Non-lesioned modalities must be returned byte-identical."""
    world = GaussianWorld(seed=0)
    sample = world.sample(batch_size=8, seed=42)
    sched = LesionScheduler(_make_spec("audio"))
    torch.manual_seed(0)
    lesioned = sched.apply(sample, step=100)
    # Audio changed.
    assert not torch.equal(lesioned.audio, sample.audio)
    # Others untouched.
    for m in ("vision", "tactile", "gravity", "force"):
        assert torch.equal(getattr(lesioned, m), getattr(sample, m)), (
            f"{m} was mutated but should not have been"
        )


def test_apply_respects_snr_schedule() -> None:
    """Higher step -> lower SNR -> larger noise magnitude."""
    world = GaussianWorld(seed=0)
    sample = world.sample(batch_size=32, seed=7)
    sched = LesionScheduler(_make_spec("vision"))
    torch.manual_seed(0)
    early = sched.apply(sample, step=10)
    torch.manual_seed(0)
    late = sched.apply(sample, step=5000)
    early_noise_std = (early.vision - sample.vision).std().item()
    late_noise_std = (late.vision - sample.vision).std().item()
    assert late_noise_std > early_noise_std * 5  # SNR drops 40 dB → ~100x factor


def test_apply_preserves_label() -> None:
    world = GaussianWorld(seed=0)
    sample = world.sample(batch_size=8, seed=42)
    sched = LesionScheduler(_make_spec("tactile"))
    lesioned = sched.apply(sample, step=500)
    assert torch.equal(lesioned.label, sample.label)


def test_apply_preserves_latent_z() -> None:
    """Latent z drives the world, not a modality — must stay untouched."""
    world = GaussianWorld(seed=0)
    sample = world.sample(batch_size=8, seed=42)
    sched = LesionScheduler(_make_spec("gravity"))
    lesioned = sched.apply(sample, step=500)
    assert torch.equal(lesioned.z, sample.z)


def test_apply_returns_new_frozen_sample() -> None:
    """apply must not mutate the input — WorldSample is frozen."""
    world = GaussianWorld(seed=0)
    sample = world.sample(batch_size=4, seed=42)
    audio_before = sample.audio.clone()
    sched = LesionScheduler(_make_spec("audio"))
    _ = sched.apply(sample, step=1000)
    assert torch.equal(sample.audio, audio_before)
