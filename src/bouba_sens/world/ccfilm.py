"""CCFilmWorld — real audio+vision perceptual bridge (ADR-0019 §E1.3d).

Scope (honest, per-channel provenance):

- `audio`  : **REAL** signal. 128-D mel-spectrogram of the film soundtrack
  (scipy STFT + a mel filter-bank written from scratch; see
  `scripts/extract_ccfilm_features.py`). This is the perceptual audio
  stimulus itself, not an annotation proxy.
- `vision` : **REAL** signal. (16,16) per-frame mean-intensity patch grid of
  decoded movie frames (PIL), normalised. The perceptual visual stimulus
  itself, not a cut-structure proxy.
- `labels` : **REAL**, derived from the film's own shot cuts (frame-to-frame
  difference threshold) quantised to 4 classes.
- `tactile`, `gravity`, `force` : **SYNTHETIC**. A film carries no
  proprioceptive channel; these are RNG-seeded streams derived from the
  audio-energy envelope flux so they remain temporally coherent with the
  scene but are explicitly NOT measured signal.

Contrast with the Studyforrest bridge: Studyforrest gives REAL
proprioception (physio / head-motion / scene annotations) but only
annotation **proxies** for audio/vision. CCFilm inverts that trade-off —
REAL audio+vision (the perceptual core Studyforrest lacked) at the cost of
SYNTHETIC proprioception and the loss of Studyforrest's biological
anchoring. It is a *different* bridge, not a replacement.

Contract matches `WorldSimulator`: identical `sample()` / `modality_dims()`
as `GaussianWorld`, so SensoryWML + encoders + nerve wire unchanged.
Temporally contiguous sampling preserves lag-1 autocorr (scene continuity).
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


class CCFilmWorld:
    """CC-BY film real-AV bridge: REAL audio+vision+labels, SYNTHETIC proprio."""

    def __init__(self, *, seed: int = 0, data_dir: Path) -> None:
        dir_path = Path(data_dir)
        self._seed = seed
        self._audio = torch.load(dir_path / "audio.pt")
        vision = torch.load(dir_path / "vision.pt")
        self._vision = (
            vision if vision.ndim == 3 else vision.reshape(vision.shape[0], *_VISION_SHAPE)
        )
        self._tactile = torch.load(dir_path / "tactile.pt")
        self._gravity = torch.load(dir_path / "gravity.pt")
        self._force = torch.load(dir_path / "force.pt")
        self._labels = torch.load(dir_path / "labels.pt").long()
        self._n = self._labels.shape[0]
        self._mode = "real_av"

    @property
    def mode(self) -> str:
        return self._mode

    def sample(self, batch_size: int, seed: int) -> WorldSample:
        gen = torch.Generator().manual_seed(seed)
        start = int(torch.randint(0, max(1, self._n - batch_size), (1,), generator=gen).item())
        idx = torch.arange(start, start + batch_size) % self._n
        return WorldSample(
            z=torch.zeros(batch_size, 1),
            audio=self._audio[idx],
            vision=self._vision[idx],
            tactile=self._tactile[idx],
            gravity=self._gravity[idx],
            force=self._force[idx],
            label=self._labels[idx],
        )

    def modality_dims(self) -> dict[str, tuple[int, ...]]:
        return {
            "audio": (_AUDIO_DIM,),
            "vision": _VISION_SHAPE,
            "tactile": (_TACTILE_DIM,),
            "gravity": (_GRAVITY_DIM,),
            "force": (_FORCE_DIM,),
        }
