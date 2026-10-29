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

import math
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


class StudyforrestRealWorld:
    """5-modality real biological bridge (Sprint 9 / Task 9.3).

    Reads 6 pre-extracted tensors (output of
    `scripts/extract_studyforrest_features.py`) and serves temporally
    contiguous batches. No zeroed modalities: tactile, gravity, force
    all carry real signal from motion annotations, fMRI head-motion
    regressors, and ECG+respiration respectively.

    Contract matches `WorldSimulator`: identical `sample()` and
    `modality_dims()` signatures as `GaussianWorld`, so downstream
    SensoryWML + encoders + nerve need no change.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        data_dir: Path,
        probe_fraction: float = 0.0,
        stratified_probe: bool = False,
    ) -> None:
        """Load the cache; ``probe_fraction`` > 0 holds out the last frames.

        ``stratified_probe`` (amendment v0.9 section 3.4, PI decision
        2026-10-25; excludes ``probe_fraction``) holds out instead the 512
        rows of ``probe.stratified_heldout_indices``: 32 label-run blocks
        of 16 frames, apportioned to the four classes in proportion to
        their frame counts (at least one block each), drawn with the
        frozen seed ``V09_HELDOUT_SEED``. Training windows then avoid
        every held-out frame, donor pools exclude them, and the probe is
        exactly the held-out rows.

        Amendment v0.9 (section 3.4) reserves the last
        ``ceil(probe_fraction * n)`` frames for the fusion-gain probe.
        Training draws (``sample``, ``sample_with_indices``,
        ``dataset_labels`` for donor pools) then use only the first
        ``n_train`` frames, windows never straddle the boundary, and
        ``probe_sample_with_indices`` serves windows inside the probe
        frames. The default 0.0 keeps the legacy behaviour unchanged.
        """
        if not 0.0 <= probe_fraction < 1.0:
            raise ValueError(f"probe_fraction must be in [0, 1); got {probe_fraction}")
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
        self._mode = "real5"
        self.n_probe = math.ceil(probe_fraction * self._n) if probe_fraction > 0 else 0
        self.n_train = self._n - self.n_probe
        if self.n_probe and self.n_train < 1:
            raise ValueError(f"probe_fraction {probe_fraction} leaves no training frame")
        self._heldout: torch.Tensor | None = None
        self._held_mask = torch.zeros(self._n, dtype=torch.bool)
        self._valid_starts: dict[int, torch.Tensor] = {}
        if stratified_probe:
            if probe_fraction:
                raise ValueError("stratified_probe and probe_fraction are exclusive")
            from bouba_sens.probe import stratified_heldout_indices

            self._heldout = stratified_heldout_indices(self._labels)
            self._held_mask[self._heldout] = True
            self.n_probe = len(self._heldout)
            self.n_train = self._n - self.n_probe

    @property
    def stratified(self) -> bool:
        return self._heldout is not None

    def _training_starts(self, batch_size: int) -> torch.Tensor:
        """Window starts t with [t, t + batch_size) free of held-out frames (no wrap)."""
        if batch_size not in self._valid_starts:
            held = torch.cat([torch.zeros(1), self._held_mask.double().cumsum(0)])
            t = torch.arange(0, self._n - batch_size + 1)
            ok = (held[t + batch_size] - held[t]) == 0
            starts = t[ok]
            if len(starts) == 0:
                raise ValueError(f"no training window of {batch_size} frames avoids the probe")
            self._valid_starts[batch_size] = starts
        return self._valid_starts[batch_size]

    @property
    def mode(self) -> str:
        return self._mode

    def sample(self, batch_size: int, seed: int) -> WorldSample:
        return self.sample_with_indices(batch_size, seed)[0]

    @property
    def dataset_labels(self) -> torch.Tensor:
        """Labels of the training rows (donor pools for v0.9 decorrelation).

        Every row without a held-out probe; the first ``n_train`` rows
        with one, so donor indices never point into the probe frames.
        """
        if self._heldout is not None:
            # Full length, held-out rows relabelled -1: row index = dataset
            # index, and no training anchor shares the -1 class.
            masked: torch.Tensor = self._labels.clone()
            masked[self._heldout] = -1
            return masked
        labels: torch.Tensor = self._labels[: self.n_train]
        return labels

    def sample_with_indices(self, batch_size: int, seed: int) -> tuple[WorldSample, torch.Tensor]:
        """Same batch as `sample`, plus the dataset row index of each sample."""
        gen = torch.Generator().manual_seed(seed)
        if self._heldout is not None:
            starts = self._training_starts(batch_size)
            j = int(torch.randint(0, len(starts), (1,), generator=gen).item())
            idx = torch.arange(int(starts[j]), int(starts[j]) + batch_size)
            return self.gather(idx), idx
        if not self.n_probe:
            start = int(torch.randint(0, max(1, self._n - batch_size), (1,), generator=gen).item())
            idx = torch.arange(start, start + batch_size) % self._n
            return self.gather(idx), idx
        if batch_size > self.n_train:
            raise ValueError(
                f"batch_size {batch_size} exceeds the training split ({self.n_train} frames)"
            )
        start = int(torch.randint(0, self.n_train - batch_size + 1, (1,), generator=gen).item())
        idx = torch.arange(start, start + batch_size)
        return self.gather(idx), idx

    def probe_sample_with_indices(
        self, batch_size: int, seed: int
    ) -> tuple[WorldSample, torch.Tensor]:
        """Contiguous window inside the held-out probe frames (v0.9, 3.4).

        With ``stratified_probe`` the probe is every held-out row (sorted),
        the same for all seeds; ``batch_size`` must equal ``n_probe`` and
        ``seed`` is not used.
        """
        if self._heldout is not None:
            if batch_size != self.n_probe:
                raise ValueError(f"the stratified probe has {self.n_probe} rows; got {batch_size}")
            return self.gather(self._heldout), self._heldout
        if not self.n_probe:
            raise ValueError("this world has no held-out probe (probe_fraction = 0)")
        if batch_size > self.n_probe:
            raise ValueError(
                f"batch_size {batch_size} exceeds the probe split ({self.n_probe} frames)"
            )
        gen = torch.Generator().manual_seed(seed)
        offset = int(torch.randint(0, self.n_probe - batch_size + 1, (1,), generator=gen).item())
        idx = torch.arange(self.n_train + offset, self.n_train + offset + batch_size)
        return self.gather(idx), idx

    def gather(self, idx: torch.Tensor) -> WorldSample:
        """Rows `idx` of the dataset as a `WorldSample`."""
        return WorldSample(
            z=torch.zeros(len(idx), 1),
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
