"""Sprint 9.5 — adapted 4.5-modal extraction from actual Studyforrest phase-2.

Implements ADR-0012 path (b) — CC-licensed soundtrack substitution — so
the artefact chain is publishable to Zenodo without copyright bind.

- audio   : CC-BY-4.0 LibriSpeech sample ``libri1`` bundled by librosa
            (~3 s real English speech), tiled to the extraction duration
            and passed through a 128-bin mel-spectrogram. Substitute for
            the Forrest Gump soundtrack which is not redistributable.
- vision  : real VGG16 conv4_3 features over movie_localizer.mkv (from
            studyforrest-data-phase2), pooled to 256 dims reshaped 16x16.
- tactile : scene-cut one-hot via ffmpeg scene-change detection on
            movie_localizer.mkv, projected to 32 dims (deterministic seed 0).
            Replaces the absent motion-locations annotation CSV.
- gravity : zeros, documented fallback — rp_*.txt head-motion regressors
            are not published with phase-2 BOLD. Log-level warning emitted.
- force   : real ECG + respiration from sub-01 ses-localizer
            task-movielocalizer run-1 cardresp physio.
- labels  : 4-quantile binning of audio RMS per 100 ms window.

The resulting 6 .pt tensors load via `StudyforrestRealWorld`.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from bouba_sens.world.studyforrest_features import (
    TARGET_HZ,
    extract_force,
    extract_vision,
)

_LIBRI_SAMPLE_KEY = "libri1"  # CC-BY-4.0 English speech, bundled by librosa
_AUDIO_SAMPLE_RATE = 22050
_VIDEO_PATH_REL = "code/stimulus/movie_localizer/videos/movie_localizer.mkv"
_PHYSIO_PATH_REL = (
    "sub-01/ses-localizer/func/"
    "sub-01_ses-localizer_task-movielocalizer_run-1_recording-cardresp_physio.tsv.gz"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_audio(n_frames: int) -> torch.Tensor:
    """Mel-spectrogram of a CC-BY-4.0 LibriSpeech speech sample.

    Loads ``librosa.ex("libri1")`` (~3 s), tiles to cover the extraction
    duration, computes a 128-mel-bin spectrogram at TARGET_HZ frame rate
    (hop_length = sr / TARGET_HZ). Returns (n_frames, 128) float32.
    """
    import librosa

    path = librosa.ex(_LIBRI_SAMPLE_KEY)
    y, sr = librosa.load(path, sr=_AUDIO_SAMPLE_RATE, mono=True)
    needed_samples = int(n_frames * sr / TARGET_HZ) + sr
    if y.shape[0] < needed_samples:
        reps = needed_samples // y.shape[0] + 1
        y = np.tile(y, reps)[:needed_samples]

    hop_length = int(sr) // TARGET_HZ
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max).T  # (T_native, 128)
    if mel_db.shape[0] >= n_frames:
        mel_db = mel_db[:n_frames]
    else:
        pad = np.zeros((n_frames - mel_db.shape[0], 128), dtype=np.float32)
        mel_db = np.concatenate([mel_db, pad], axis=0)
    return torch.from_numpy(mel_db.astype(np.float32))


def build_labels(audio: torch.Tensor, n_classes: int = 4) -> torch.Tensor:
    rms = audio.pow(2).mean(dim=-1).sqrt()
    quantiles = torch.quantile(rms, torch.tensor([0.25, 0.5, 0.75]))
    return torch.bucketize(rms, quantiles).long()


def build_tactile(video_path: Path, n_frames: int) -> torch.Tensor:
    """Scene-cut one-hot from ffmpeg + projection to 32 dims."""
    probe = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            "select='gt(scene,0.3)',showinfo",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # ffmpeg writes scene timestamps to stderr as "pts_time:X.YYY".
    times_s: list[float] = []
    for line in probe.stderr.splitlines():
        idx = line.find("pts_time:")
        if idx < 0:
            continue
        tail = line[idx + len("pts_time:") :].split()[0]
        try:
            times_s.append(float(tail))
        except ValueError:
            continue
    rng = np.random.default_rng(0)
    flat = np.zeros((n_frames, 64), dtype=np.float32)
    for t in times_s:
        frame = int(t * TARGET_HZ)
        if 0 <= frame < n_frames:
            flat[frame, int(t) % 64] = 1.0
    proj = rng.standard_normal((64, 32)).astype(np.float32)
    return torch.from_numpy(flat @ proj)


def build_gravity(n_frames: int) -> torch.Tensor:
    import warnings

    warnings.warn(
        "build_gravity: rp_*.txt head-motion regressors not available in "
        "studyforrest-data-phase2; returning zero tensor. Sprint 10+ scope.",
        stacklevel=2,
    )
    return torch.zeros(n_frames, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=int, default=180)
    args = parser.parse_args()

    n_frames = args.duration_seconds * TARGET_HZ

    video_src = args.phase2_root / _VIDEO_PATH_REL
    physio_src = args.phase2_root / _PHYSIO_PATH_REL
    for src in (video_src, physio_src):
        if not src.exists():
            print(f"[extract] FATAL: {src} missing", file=sys.stderr)
            sys.exit(1)

    print(f"[extract] audio <- librosa {_LIBRI_SAMPLE_KEY} (CC-BY-4.0)")
    audio = build_audio(n_frames)
    print(f"[extract] vision <- {video_src}")
    vision = extract_vision(video_src, n_frames)
    print(f"[extract] tactile <- scene-cuts of {video_src}")
    tactile = build_tactile(video_src, n_frames)
    print("[extract] gravity <- zeros fallback")
    gravity = build_gravity(n_frames)
    print(f"[extract] force <- {physio_src}")
    force = extract_force(physio_src, n_frames)
    print("[extract] labels <- audio RMS quantiles")
    labels = build_labels(audio)

    args.out.mkdir(parents=True, exist_ok=True)
    tensors = {
        "audio.pt": audio,
        "vision.pt": vision,
        "tactile.pt": tactile,
        "gravity.pt": gravity,
        "force.pt": force,
        "labels.pt": labels,
    }
    for name, tensor in tensors.items():
        torch.save(tensor, args.out / name)

    lines = [
        "# Studyforrest phase-2 adapted 4.5-modal manifest (Sprint 9.5)",
        "",
        "Mapping (honest per ADR-0012 path (b) workaround):",
        "- audio   : librosa libri1 CC-BY-4.0 LibriSpeech, mel-spectrogram",
        "- vision  : VGG16 features of movie_localizer.mkv (REAL video)",
        "- tactile : ffmpeg scene-cut one-hot + 32-d projection",
        "- gravity : zeros (rp_*.txt not in phase-2)",
        "- force   : REAL cardiac+respiration from sub-01 ses-localizer run-1",
        "- labels  : 4-quantile binning of audio RMS",
        "",
        f"Duration: {args.duration_seconds} s at {TARGET_HZ} Hz ({n_frames} frames)",
        "",
        "## Tensor shapes",
        "",
    ]
    for name, tensor in tensors.items():
        lines.append(f"- `{name}`: shape {tuple(tensor.shape)}, dtype {tensor.dtype}")
    lines.append("")
    lines.append("## SHA256")
    lines.append("")
    for name in tensors:
        lines.append(f"- `{name}`: `{_sha256(args.out / name)}`")
    (args.out / "MANIFEST.md").write_text("\n".join(lines) + "\n")
    print(f"[extract] wrote {len(tensors)} tensors + MANIFEST to {args.out}")


if __name__ == "__main__":
    main()
