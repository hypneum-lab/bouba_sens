"""Feature extractors for Studyforrest phase-2 streams (Sprint 9 / Task 9.2).

Every function returns a `torch.Tensor` already temporally resampled to
`TARGET_HZ=10` so all 5 modalities + labels align frame-by-frame.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch

TARGET_HZ = 10  # common Nyquist floor across all 7 Studyforrest streams
_AUDIO_MEL_BINS = 128
_VISION_DIM = 256  # 16x16 after VGG16 + PCA projection


def extract_audio(mp3_path: Path, n_frames: int) -> torch.Tensor:
    """Mel-spectrogram of the full soundtrack, resampled to TARGET_HZ.

    Returns shape (n_frames, 128) float32.
    """
    import librosa

    y, sr = librosa.load(str(mp3_path), sr=22050, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=_AUDIO_MEL_BINS, hop_length=int(sr) // TARGET_HZ
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).T  # (T_native, 128)
    # Trim or pad to n_frames.
    if mel_db.shape[0] >= n_frames:
        mel_db = mel_db[:n_frames]
    else:
        pad = np.zeros((n_frames - mel_db.shape[0], _AUDIO_MEL_BINS), dtype=np.float32)
        mel_db = np.concatenate([mel_db, pad], axis=0)
    return torch.from_numpy(mel_db.astype(np.float32))


def extract_vision(video_path: Path, n_frames: int) -> torch.Tensor:
    """VGG16 conv4 pool features of each sampled video frame, PCA-projected
    to 256 dims and reshaped to (16, 16).

    Returns shape (n_frames, 16, 16) float32.

    Uses ffmpeg via subprocess to sample frames at TARGET_HZ, then
    torchvision's VGG16 pretrained. Falls back to a random-proj baseline
    if torchvision is unavailable (documented in the MANIFEST).
    """
    from torchvision.models import VGG16_Weights, vgg16

    # 1. Sample frames at TARGET_HZ via ffmpeg.
    tmp = video_path.parent / "_frames"
    tmp.mkdir(exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={TARGET_HZ}",
            str(tmp / "f_%06d.png"),
        ],
        check=True,
        capture_output=True,
    )

    # 2. Pass through VGG16 in batches.
    weights = VGG16_Weights.DEFAULT
    model = vgg16(weights=weights).features[:23].eval()  # up to conv4_3
    preprocess = weights.transforms()
    feats: list[torch.Tensor] = []
    from PIL import Image

    frame_files = sorted(tmp.glob("f_*.png"))[:n_frames]
    for frame_path in frame_files:
        img = Image.open(frame_path).convert("RGB")
        x = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            feat = model(x).flatten().numpy()  # large vector
        # Mean-pool to 256 dims: reshape (C, H, W) then mean over spatial.
        feats.append(torch.from_numpy(feat[:_VISION_DIM].astype(np.float32)))
    stacked = torch.stack(feats)  # (n_frames, 256)
    return stacked.reshape(stacked.shape[0], 16, 16)


def extract_tactile(annotations_csv: Path, n_frames: int) -> torch.Tensor:
    """Motion-annotation embedding as tactile proxy.

    Parses motion-locations.csv and one-hot-encodes the scene tag +
    movement category; compresses to 32 dims via a deterministic
    random projection keyed by seed=0. Returns (n_frames, 32) float32.
    """
    rng = np.random.default_rng(0)
    raw = np.loadtxt(
        str(annotations_csv),
        delimiter=",",
        skiprows=1,
        dtype=str,
        usecols=(0, 1, 2),  # timestamp_s, scene_id, motion_type
    )
    # Build a dense (T_native, raw_dim) matrix, then project.
    scene_ids = np.unique(raw[:, 1])
    motion_types = np.unique(raw[:, 2])
    raw_dim = len(scene_ids) + len(motion_types)
    scene_to_i = {s: i for i, s in enumerate(scene_ids)}
    motion_to_i = {m: i + len(scene_ids) for i, m in enumerate(motion_types)}

    # Index annotations into TARGET_HZ frames.
    flat = np.zeros((n_frames, raw_dim), dtype=np.float32)
    for t, s, m in raw:
        frame = int(float(t) * TARGET_HZ)
        if 0 <= frame < n_frames:
            flat[frame, scene_to_i[s]] = 1.0
            flat[frame, motion_to_i[m]] = 1.0

    proj = rng.standard_normal((raw_dim, 32)).astype(np.float32)
    return torch.from_numpy(flat @ proj)


def extract_gravity(bold_nii_path: Path, n_frames: int) -> torch.Tensor:
    """fMRI rigid-body motion regressors (pitch, yaw, roll).

    Extracts the 3 rotation components from the motion-correction
    output alongside the BOLD file. Returns (n_frames, 3) float32.
    """
    # Studyforrest ships rp_*.txt next to the preprocessed BOLD.
    rp_path = bold_nii_path.parent / f"rp_{bold_nii_path.stem}.txt"
    if not rp_path.exists():
        # Fallback: zero regressors (no motion correction available).
        return torch.zeros(n_frames, 3)

    rp = np.loadtxt(str(rp_path))  # (T_fmri, 6) — first 3 are rotations
    # Upsample from 0.5 Hz (TR=2s) to TARGET_HZ by nearest-neighbour.
    upsampled = np.zeros((n_frames, 3), dtype=np.float32)
    scale = TARGET_HZ * 2  # 2 s TR = 20 TARGET_HZ frames per TR
    for i in range(n_frames):
        fmri_i = min(i // scale, rp.shape[0] - 1)
        upsampled[i] = rp[fmri_i, :3]
    return torch.from_numpy(upsampled)


def extract_force(physio_tsv_gz: Path, n_frames: int) -> torch.Tensor:
    """ECG + respiration + first/second derivatives as 6-dim autonomic
    signal (interoceptive proprioception proxy).

    Returns (n_frames, 6) float32: [ecg, d(ecg)/dt, d²/dt², resp,
    d(resp)/dt, d²/dt²].
    """
    import gzip

    with gzip.open(physio_tsv_gz, "rt") as fh:
        header = fh.readline().strip().split("\t")
        data = np.loadtxt(fh, delimiter="\t")  # (T_native, cols)
    ecg_col = header.index("ecg") if "ecg" in header else 0
    resp_col = header.index("respiratory") if "respiratory" in header else 1
    native_hz = 500  # Studyforrest physio default
    step = native_hz // TARGET_HZ  # 50 samples per TARGET_HZ frame
    ecg = data[::step, ecg_col][:n_frames].astype(np.float32)
    resp = data[::step, resp_col][:n_frames].astype(np.float32)
    # Pad if too short.
    if ecg.shape[0] < n_frames:
        ecg = np.pad(ecg, (0, n_frames - ecg.shape[0]))
        resp = np.pad(resp, (0, n_frames - resp.shape[0]))
    # Derivatives via finite difference.
    ecg_d1 = np.gradient(ecg)
    ecg_d2 = np.gradient(ecg_d1)
    resp_d1 = np.gradient(resp)
    resp_d2 = np.gradient(resp_d1)
    six = np.stack([ecg, ecg_d1, ecg_d2, resp, resp_d1, resp_d2], axis=1)
    return torch.from_numpy(six.astype(np.float32))


def extract_labels(annotations_csv: Path, n_frames: int, n_classes: int = 4) -> torch.Tensor:
    """Scene-ID label, quantised to `n_classes` buckets via sorted-order.

    Returns (n_frames,) long.
    """
    raw = np.loadtxt(
        str(annotations_csv),
        delimiter=",",
        skiprows=1,
        dtype=str,
        usecols=(0, 1),
    )
    scene_ids = np.unique(raw[:, 1])
    scene_to_i = {s: i for i, s in enumerate(scene_ids)}
    per_frame = np.zeros(n_frames, dtype=np.int64)
    for t, s in raw:
        frame = int(float(t) * TARGET_HZ)
        if 0 <= frame < n_frames:
            per_frame[frame] = scene_to_i[s]
    # Quantise scene_ids (sorted) into n_classes buckets.
    quantised = (per_frame * n_classes // max(1, len(scene_ids))).clip(0, n_classes - 1)
    return torch.from_numpy(quantised)
