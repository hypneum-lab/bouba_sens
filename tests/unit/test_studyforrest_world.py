"""Task 7.6 — StudyforrestWorld bridge stub tests (mock mode, no network)."""

from __future__ import annotations

import torch

from bouba_sens.audit import compute_world_profile
from bouba_sens.world.studyforrest import StudyforrestWorld


def test_studyforrest_mock_sample_shapes() -> None:
    world = StudyforrestWorld(seed=0, n_mock_samples=512)
    sample = world.sample(batch_size=16, seed=42)
    assert world.mode == "mock"
    assert sample.audio.shape == (16, 128)
    assert sample.vision.shape == (16, 16, 16)
    assert sample.tactile.shape == (16, 32)
    assert sample.gravity.shape == (16, 3)
    assert sample.force.shape == (16, 6)
    assert sample.label.shape == (16,)
    assert sample.label.dtype == torch.long
    # Zero-modality tensors are exactly zero.
    assert torch.all(sample.tactile == 0)
    assert torch.all(sample.gravity == 0)
    assert torch.all(sample.force == 0)


def test_studyforrest_mock_is_reproducible() -> None:
    a = StudyforrestWorld(seed=7, n_mock_samples=256).sample(batch_size=8, seed=3)
    b = StudyforrestWorld(seed=7, n_mock_samples=256).sample(batch_size=8, seed=3)
    assert torch.equal(a.audio, b.audio)
    assert torch.equal(a.label, b.label)


def test_studyforrest_mock_has_non_trivial_audio_vision_mi() -> None:
    """GaussianWorld has MI(audio,vision) near 0; mock should be higher."""
    world = StudyforrestWorld(seed=0, n_mock_samples=2048)
    sample = world.sample(batch_size=1024, seed=0)
    profile = compute_world_profile(sample)
    # Cross-modal MI is driven by the shared AR(1) scene latent;
    # should clearly exceed GaussianWorld's ~0.02 baseline.
    assert profile["mi_pairwise"] > 0.05, (
        f"mock MI should exceed 0.05; got {profile['mi_pairwise']}"
    )


def test_studyforrest_mock_has_temporal_structure() -> None:
    """The AR(1) scene latent must produce non-zero lag-1 autocorr."""
    world = StudyforrestWorld(seed=0, n_mock_samples=4096)
    sample = world.sample(batch_size=2048, seed=0)
    profile = compute_world_profile(sample)
    assert abs(profile["temporal_autocorr"]) > 0.05, (
        f"mock should show temporal structure; got {profile['temporal_autocorr']}"
    )
