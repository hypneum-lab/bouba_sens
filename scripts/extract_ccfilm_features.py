"""Extract a real audio+vision perceptual cache from a CC-BY film.

ADR-0019 §E1.3d — CC-film real-AV bridge.

Produces the 6-tensor `data/ccfilm_real/` cache consumed by `CCFilmWorld`:

  audio   (N,128)  f32   REAL mel-spectrogram of the soundtrack
  vision  (N,16,16) f32  REAL per-frame 16x16 mean-intensity patch grid
  labels  (N,)     long  REAL shot-cut classes (frame-difference quantised)
  tactile (N,32)   f32   SYNTHETIC (RNG seeded by audio flux)
  gravity (N,3)    f32   SYNTHETIC
  force   (N,6)    f32   SYNTHETIC

The mel filter-bank and STFT framing are written from scratch (scipy STFT +
HTK-style mel triangles) — no torchaudio / librosa. Frames are decoded by
ffmpeg at a fixed fps and aligned to the audio frame grid on a common N.

Usage:
    uv run python scripts/extract_ccfilm_features.py
    uv run python scripts/extract_ccfilm_features.py --keep-tmp
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from PIL import Image
from scipy.signal import stft

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FILM_URL = "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4"

SR = 16000
FPS = 2.0  # frame grid: 2 frames/sec == 0.5s per row == audio hop window
N_MELS = 128
VISION_SHAPE = (16, 16)
TACTILE_DIM = 32
GRAVITY_DIM = 3
FORCE_DIM = 6


# --------------------------------------------------------------------------
# Mel filter-bank from scratch (HTK mel scale).
# --------------------------------------------------------------------------
def hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int, n_fft: int, sr: int, fmin: float = 0.0, fmax: float | None = None
) -> np.ndarray:
    """(n_mels, n_fft//2+1) triangular mel filter-bank."""
    fmax = fmax if fmax is not None else sr / 2.0
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2.0, n_bins)
    mel_pts = np.linspace(
        hz_to_mel(np.array([fmin]))[0], hz_to_mel(np.array([fmax]))[0], n_mels + 2
    )
    hz_pts = mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(1, n_mels + 1):
        lo, ctr, hi = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
        left = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        right = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        fb[m - 1] = np.clip(np.minimum(left, right), 0.0, None)
    return fb


def compute_mel(wav: np.ndarray, sr: int, n_frames: int) -> np.ndarray:
    """Return (n_frames, N_MELS) log-mel, resampled onto the frame grid."""
    # STFT window sized so we get a dense time axis we can pool to n_frames.
    n_fft = 1024
    hop = 256
    _f, _t, zxx = stft(wav, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    power = np.abs(zxx) ** 2  # (n_bins, n_stft_frames)
    fb = mel_filterbank(N_MELS, n_fft, sr)  # (N_MELS, n_bins)
    mel = fb @ power  # (N_MELS, n_stft_frames)
    log_mel = np.log(mel + 1e-6)  # (N_MELS, T)
    # Pool the STFT time axis onto the n_frames common grid by averaging.
    n_t = log_mel.shape[1]
    edges = np.linspace(0, n_t, n_frames + 1).astype(int)
    out = np.zeros((n_frames, N_MELS), dtype=np.float32)
    for i in range(n_frames):
        a, b = edges[i], max(edges[i] + 1, edges[i + 1])
        out[i] = log_mel[:, a : min(b, n_t)].mean(axis=1)
    # Per-band standardise (zero-mean unit-var) — keeps the contract f32.
    out = (out - out.mean(0, keepdims=True)) / (out.std(0, keepdims=True) + 1e-6)
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# Vision: decoded frames -> 16x16 mean-intensity grid.
# --------------------------------------------------------------------------
def frame_to_grid(img: Image.Image) -> np.ndarray:
    g = img.convert("L").resize((VISION_SHAPE[1] * 4, VISION_SHAPE[0] * 4))
    arr = np.asarray(g, dtype=np.float32) / 255.0
    h, w = arr.shape
    ph, pw = h // VISION_SHAPE[0], w // VISION_SHAPE[1]
    grid = arr[: ph * VISION_SHAPE[0], : pw * VISION_SHAPE[1]]
    grid = grid.reshape(VISION_SHAPE[0], ph, VISION_SHAPE[1], pw).mean(axis=(1, 3))
    return grid.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/ccfilm_real"))
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=300.0,
        help="trim the film to this many seconds (cache size cap)",
    )
    ap.add_argument("--keep-tmp", action="store_true")
    ap.add_argument(
        "--mp4", type=Path, default=None, help="reuse a pre-downloaded film instead of fetching"
    )
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="ccfilm_"))
    print(f"tmp = {tmp}", flush=True)

    if args.mp4 is not None:
        mp4 = args.mp4
        print(f"reusing {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)", flush=True)
    else:
        mp4 = tmp / "film.mp4"
        print(f"downloading {FILM_URL} ...", flush=True)
        # download.blender.org 403s the default urllib UA; send a browser UA.
        req = urllib.request.Request(FILM_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, mp4.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
        print(f"  -> {mp4.stat().st_size / 1e6:.1f} MB", flush=True)

    # --- audio: mp4 -> mono 16k wav ---
    wav_path = tmp / "audio.wav"
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-i",
            str(mp4),
            "-t",
            str(args.max_seconds),
            "-ac",
            "1",
            "-ar",
            str(SR),
            str(wav_path),
        ],
        check=True,
        capture_output=True,
    )
    wav, sr = sf.read(wav_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    print(f"audio: {wav.shape[0]} samples @ {sr} Hz ({wav.shape[0] / sr:.1f}s)", flush=True)

    # --- frames: mp4 -> png at FPS ---
    frame_dir = tmp / "frames"
    frame_dir.mkdir()
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-i",
            str(mp4),
            "-t",
            str(args.max_seconds),
            "-vf",
            f"fps={FPS}",
            str(frame_dir / "%05d.png"),
        ],
        check=True,
        capture_output=True,
    )
    frame_files = sorted(frame_dir.glob("*.png"))
    print(f"frames: {len(frame_files)} @ {FPS} fps", flush=True)
    if not frame_files:
        raise SystemExit("ffmpeg produced no frames")

    # Common grid N = number of frames (audio mel pooled onto it).
    n_frames = len(frame_files)

    vision = np.stack([frame_to_grid(Image.open(f)) for f in frame_files])  # (N,16,16)
    audio = compute_mel(wav, sr, n_frames)  # (N,128)

    # --- labels: REAL shot cuts via frame-to-frame difference ---
    flat = vision.reshape(n_frames, -1)
    diff = np.zeros(n_frames, dtype=np.float32)
    diff[1:] = np.abs(flat[1:] - flat[:-1]).mean(axis=1)
    # Quantise the per-frame cut-energy into 4 classes by quantiles -> real,
    # data-driven shot-change labels.
    qs = np.quantile(diff, [0.25, 0.5, 0.75])
    labels = np.digitize(diff, qs).astype(np.int64)  # 0..3

    # --- SYNTHETIC proprioception, seeded by audio flux (temporally coherent
    #     with the scene but explicitly NOT measured signal) ---
    audio_energy = np.linalg.norm(audio, axis=1)  # (N,)
    flux = np.zeros(n_frames, dtype=np.float32)
    flux[1:] = np.abs(audio_energy[1:] - audio_energy[:-1])
    rng = np.random.default_rng(0)
    base = (flux - flux.mean()) / (flux.std() + 1e-6)
    tactile = (
        base[:, None] * rng.standard_normal((1, TACTILE_DIM))
        + 0.1 * rng.standard_normal((n_frames, TACTILE_DIM))
    ).astype(np.float32)
    gravity = (
        base[:, None] * rng.standard_normal((1, GRAVITY_DIM))
        + 0.1 * rng.standard_normal((n_frames, GRAVITY_DIM))
    ).astype(np.float32)
    force = (
        base[:, None] * rng.standard_normal((1, FORCE_DIM))
        + 0.1 * rng.standard_normal((n_frames, FORCE_DIM))
    ).astype(np.float32)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tensors = {
        "audio": torch.from_numpy(audio),
        "vision": torch.from_numpy(vision),
        "tactile": torch.from_numpy(tactile),
        "gravity": torch.from_numpy(gravity),
        "force": torch.from_numpy(force),
        "labels": torch.from_numpy(labels),
    }
    for name, t in tensors.items():
        torch.save(t, out / f"{name}.pt")
        print(f"wrote {name}.pt {tuple(t.shape)} {t.dtype}", flush=True)

    # MANIFEST with SHA256 + per-channel provenance.
    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    n_classes = int(labels.max()) + 1
    lines = [
        "# CC-film real-AV bridge cache manifest (ADR-0019 §E1.3d)",
        "",
        f"- Source film: {FILM_URL}",
        "- License: CC-BY 3.0 (Big Buck Bunny, Blender Foundation 2008)",
        f"- N frames: {n_frames} @ {FPS} fps  ({n_frames / FPS:.0f}s of film)",
        f"- Audio: mono {SR} Hz, mel-{N_MELS} (scipy STFT n_fft=1024 hop=256, "
        "HTK mel bank from scratch)",
        f"- Label classes populated: {n_classes}",
        "",
        "## Per-channel provenance",
        "",
        "- `audio.pt`  : REAL — 128-D log-mel spectrogram of the film soundtrack "
        "(the perceptual audio stimulus itself).",
        "- `vision.pt` : REAL — (16,16) per-frame mean-intensity patch grid of "
        "decoded movie frames (the perceptual visual stimulus itself).",
        "- `labels.pt` : REAL — shot-cut classes from frame-to-frame difference, "
        "quantised to 4 by quantiles.",
        "- `tactile.pt`: SYNTHETIC — RNG(seed=0) stream modulated by the audio "
        "energy-flux envelope. A film has no tactile channel; NOT measured signal.",
        "- `gravity.pt`: SYNTHETIC — same construction (3-D).",
        "- `force.pt`  : SYNTHETIC — same construction (6-D).",
        "",
        "## SHA256",
        "",
    ]
    for name in tensors:
        lines.append(f"- `{name}.pt`: `{sha(out / (name + '.pt'))}`")
    (out / "MANIFEST.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}/MANIFEST.md", flush=True)

    if args.keep_tmp:
        print(f"kept tmp at {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
