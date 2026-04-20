"""StudyforrestWorld — biological-adjacent bridge stub (Sprint 7 / Task 7.6).

Scope (honest):
- Loads 2 real modalities (audio + vision) from a pre-fetched
  Studyforrest subset when `data_dir` points to `audio.pt`,
  `vision.pt`, and `labels.pt`. The 3 non-biological modalities
  (tactile, gravity, force) are zeroed.
- When no `data_dir` is provided, generates a `mock=True` stream
  whose statistics are deliberately more realistic than
  GaussianWorld: audio-vision cross-modal correlation, temporal
  autocorrelation on the label, non-uniform support. This mock
  is NOT Studyforrest data; it exists so the bridge API can be
  tested and audited without a network download.

Not a scientific replication — a shape-test that proves
`WorldSimulator` accepts a non-synthetic producer and feeds the
audit pipeline. Full biological validation is deferred to
Sprint 9+ (see ADR-0007).
"""

from __future__ import annotations

from pathlib import Path

import torch

from bouba_sens.world.base import WorldSample

_AUDIO_DIM = 128
_VISION_SHAPE = (16, 16)
_TACTILE_DIM = 32
_GRAVITY_DIM = 3
_FORCE_DIM = 6


class StudyforrestWorld:
    """Bridge: 2 real-or-mock biological modalities + 3 zeroed."""

    def __init__(
        self,
        *,
        seed: int = 0,
        data_dir: Path | None = None,
        n_mock_samples: int = 4096,
    ) -> None:
        self._seed = seed
        if data_dir is not None:
            dir_path = Path(data_dir)
            self._audio = torch.load(dir_path / "audio.pt")
            vision = torch.load(dir_path / "vision.pt")
            self._vision = vision.reshape(vision.shape[0], *_VISION_SHAPE)
            self._labels = torch.load(dir_path / "labels.pt").long()
            self._mode = "real"
        else:
            self._audio, self._vision, self._labels = _build_mock_cache(n_mock_samples, seed)
            self._mode = "mock"
        self._n = self._labels.shape[0]

    @property
    def mode(self) -> str:
        return self._mode

    def sample(self, batch_size: int, seed: int) -> WorldSample:
        gen = torch.Generator().manual_seed(seed)
        # Sample a contiguous temporal slice so lag-1 autocorr is non-zero
        # (the film has scene continuity). Start index is stochastic.
        start = int(torch.randint(0, max(1, self._n - batch_size), (1,), generator=gen).item())
        idx = torch.arange(start, start + batch_size) % self._n
        audio = self._audio[idx]
        vision = self._vision[idx]
        labels = self._labels[idx]
        zero_flat = torch.zeros(batch_size, 1)  # placeholder for latent
        return WorldSample(
            z=zero_flat,  # no canonical latent for real data
            audio=audio,
            vision=vision,
            tactile=torch.zeros(batch_size, _TACTILE_DIM),
            gravity=torch.zeros(batch_size, _GRAVITY_DIM),
            force=torch.zeros(batch_size, _FORCE_DIM),
            label=labels,
        )

    def modality_dims(self) -> dict[str, tuple[int, ...]]:
        return {
            "audio": (_AUDIO_DIM,),
            "vision": _VISION_SHAPE,
            "tactile": (_TACTILE_DIM,),
            "gravity": (_GRAVITY_DIM,),
            "force": (_FORCE_DIM,),
        }


def _build_mock_cache(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (audio, vision, label) tensors with bio-plausible stats.

    Design choices:
    - A slow-varying scene latent s_t (AR(1) with alpha=0.9) drives
      both audio + vision — creates audio-vision cross-modal MI
      that GaussianWorld lacks.
    - Additive modality-specific noise keeps each modality
      identifiable (audio != vision).
    - Label is a quantised slow drift of s_t — lag-1 autocorr on
      labels becomes significantly non-zero.
    """
    gen = torch.Generator().manual_seed(seed)
    alpha = 0.9

    # AR(1) scene latent, 4-D.
    s = torch.zeros(n, 4)
    s[0] = torch.randn(4, generator=gen)
    noise = torch.randn(n, 4, generator=gen)
    for t in range(1, n):
        s[t] = alpha * s[t - 1] + (1 - alpha) * noise[t] * 3.0

    # Audio: projection of s biased to a common direction so mean-pooling
    # preserves the shared-scene signal (the audit's mi_pairwise collapses
    # each modality via flatten(1).mean(-1), so the shared component must
    # survive that collapse).
    audio_proj = torch.randn(4, _AUDIO_DIM, generator=gen) * 0.3
    audio_proj[0] += 1.0  # bias the first latent dim toward a DC direction
    audio = s @ audio_proj + 0.15 * torch.randn(n, _AUDIO_DIM, generator=gen)

    # Vision: same DC-biased shared channel so audio/vision means co-vary
    # strongly (this is the audible-visual scene coherence proxy).
    vision_dim = _VISION_SHAPE[0] * _VISION_SHAPE[1]
    vision_proj = torch.randn(4, vision_dim, generator=gen) * 0.3
    vision_proj[0] += 1.0
    vision_flat = s @ vision_proj + 0.15 * torch.randn(n, vision_dim, generator=gen)
    vision = vision_flat.reshape(n, *_VISION_SHAPE)

    # Label: quantise s[:, 0] into 4 bins, offset by 2-step lag for realism.
    s0_shifted = torch.cat([s[:2, 0], s[:-2, 0]])
    bins = torch.quantile(s0_shifted, torch.tensor([0.25, 0.5, 0.75]))
    labels = torch.bucketize(s0_shifted, bins).long()

    return audio, vision, labels
