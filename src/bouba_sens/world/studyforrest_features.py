"""Feature extractors for Studyforrest phase-2 streams (Sprint 9 / Task 9.2).

Every function returns a `torch.Tensor` already temporally resampled to
`TARGET_HZ=10` so all 5 modalities + labels align frame-by-frame.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

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

    Frames are written to a fresh temporary directory that is removed
    on return, so frames left over from an earlier extraction can never
    be read (issue #45). The row count is the number of frames the video
    actually yields, capped at ``n_frames``; it is NOT padded, so a short
    video gives fewer rows and the caller must check alignment.
    """
    from PIL import Image
    from torchvision.models import VGG16_Weights, vgg16

    weights = VGG16_Weights.DEFAULT
    model = vgg16(weights=weights).features[:23].eval()  # up to conv4_3
    preprocess = weights.transforms()
    feats: list[torch.Tensor] = []

    with tempfile.TemporaryDirectory(prefix="bouba_sens_frames_") as tmp_name:
        tmp = Path(tmp_name)
        # 1. Sample frames at TARGET_HZ via ffmpeg.
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

        # 2. Pass each frame through VGG16.
        frame_files = sorted(tmp.glob("f_*.png"))[:n_frames]
        for frame_path in frame_files:
            with Image.open(frame_path) as raw:
                img = raw.convert("RGB")
            x = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                feat = model(x).flatten().numpy()  # large vector
            # Keep the first 256 dims of the flattened conv4_3 map.
            feats.append(torch.from_numpy(feat[:_VISION_DIM].astype(np.float32)))
    stacked = torch.stack(feats)  # (n_rows, 256), n_rows <= n_frames
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


# Studyforrest phase-2 cardresp layout, used only when no BIDS sidecar is
# found for a physio file. Source: the dataset-level sidecar
# `recording-cardresp_physio.json` of studyforrest-data-phase2
# (SamplingFrequency 500.0, Columns [trigger, cardiac, respiratory],
# "Activity recorded with a pulse oximeter (cardiac) and respiration
# belt."). BIDS physio TSVs have no header line; column names live in
# the sidecar (issue #72).
_PHASE2_PHYSIO_COLUMNS = ("trigger", "cardiac", "respiratory")
_PHASE2_PHYSIO_HZ = 500.0


def _bids_entities(path: Path) -> set[str]:
    """Key-value entities of a BIDS filename, without the suffix.

    ``sub-01_task-x_recording-cardresp_physio.tsv.gz`` gives
    ``{"sub-01", "task-x", "recording-cardresp"}``.
    """
    stem = path.name.split(".", 1)[0]
    return set(stem.split("_")[:-1])


def find_physio_sidecar(physio_tsv_gz: Path) -> dict[str, object]:
    """Merge the ``*_physio.json`` sidecars that apply to a physio TSV.

    Follows the BIDS inheritance principle: a sidecar applies when its
    entities are a subset of the file's entities; sidecars in deeper
    directories, and with more entities, override shallower ones. The
    walk stops at the dataset root (the directory holding
    ``dataset_description.json``). Returns ``{}`` when none applies.
    """
    import json

    entities = _bids_entities(physio_tsv_gz)
    levels: list[list[Path]] = []
    for directory in [physio_tsv_gz.parent, *physio_tsv_gz.parent.parents]:
        found = [j for j in directory.glob("*_physio.json") if _bids_entities(j) <= entities]
        levels.append(sorted(found, key=lambda j: len(_bids_entities(j))))
        if (directory / "dataset_description.json").exists():
            break
    merged: dict[str, object] = {}
    for level in reversed(levels):  # root first, file directory last
        for sidecar in level:
            merged.update(json.loads(sidecar.read_text()))
    return merged


def _is_numeric_row(fields: list[str]) -> bool:
    try:
        [float(f) for f in fields]
    except ValueError:
        return False
    return True


def read_physio(physio_tsv_gz: Path) -> tuple[np.ndarray, list[str], float]:
    """Read a BIDS physio TSV with its column names and sampling rate.

    The first line is consumed as a header only if it is not numeric
    (BIDS files have none, issue #72). Column names come from the
    sidecar ``Columns``, else from a header line, else from the
    phase-2 cardresp defaults; the rate from ``SamplingFrequency``,
    else the phase-2 default (500 Hz).

    Returns ``(data (T, n_cols), columns, sampling_hz)``.
    """
    import gzip

    with gzip.open(physio_tsv_gz, "rt") as fh:
        lines = [line for line in fh.read().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{physio_tsv_gz}: empty physio file")
    first = lines[0].strip().split("\t")
    header: list[str] | None = None
    if not _is_numeric_row(first):
        header = first
        lines = lines[1:]
    data = np.loadtxt(lines, delimiter="\t", ndmin=2)

    sidecar = find_physio_sidecar(physio_tsv_gz)
    columns_raw = sidecar.get("Columns")
    if isinstance(columns_raw, list):
        columns = [str(c) for c in columns_raw]
    elif header is not None:
        columns = header
    else:
        logger.warning(
            "%s: no physio sidecar and no header; assuming phase-2 cardresp columns %s",
            physio_tsv_gz,
            _PHASE2_PHYSIO_COLUMNS,
        )
        columns = list(_PHASE2_PHYSIO_COLUMNS)
    rate_raw = sidecar.get("SamplingFrequency")
    if isinstance(rate_raw, int | float):
        sampling_hz = float(rate_raw)
    else:
        logger.warning(
            "%s: no SamplingFrequency in a sidecar; assuming %.0f Hz",
            physio_tsv_gz,
            _PHASE2_PHYSIO_HZ,
        )
        sampling_hz = _PHASE2_PHYSIO_HZ
    if data.shape[1] != len(columns):
        raise ValueError(
            f"{physio_tsv_gz}: {data.shape[1]} data columns but {len(columns)} names {columns}"
        )
    return data, columns, sampling_hz


def extract_force(physio_tsv_gz: Path, n_frames: int) -> torch.Tensor:
    """Cardiac (pulse oximetry) + respiration + first/second derivatives
    as 6-dim autonomic signal (interoceptive proprioception proxy).

    Columns are mapped by name (``cardiac``, ``respiratory``) from the
    BIDS sidecar (see :func:`read_physio`); the signal is decimated from
    the sidecar rate to TARGET_HZ and zero-padded to ``n_frames``.

    Returns (n_frames, 6) float32: [cardiac, d(cardiac)/dt, d²/dt², resp,
    d(resp)/dt, d²/dt²].
    """
    data, columns, sampling_hz = read_physio(physio_tsv_gz)
    for name in ("cardiac", "respiratory"):
        if name not in columns:
            raise ValueError(f"{physio_tsv_gz}: no {name!r} column in {columns}")
    cardiac_col = columns.index("cardiac")
    resp_col = columns.index("respiratory")
    step = round(sampling_hz / TARGET_HZ)  # 50 samples per frame at 500 Hz
    if step < 1 or abs(step * TARGET_HZ - sampling_hz) > 1e-6:
        raise ValueError(f"{physio_tsv_gz}: {sampling_hz} Hz is not a multiple of {TARGET_HZ}")
    cardiac = data[::step, cardiac_col][:n_frames].astype(np.float32)
    resp = data[::step, resp_col][:n_frames].astype(np.float32)
    # Pad if too short.
    if cardiac.shape[0] < n_frames:
        cardiac = np.pad(cardiac, (0, n_frames - cardiac.shape[0]))
        resp = np.pad(resp, (0, n_frames - resp.shape[0]))
    # Derivatives via finite difference.
    cardiac_d1 = np.gradient(cardiac)
    cardiac_d2 = np.gradient(cardiac_d1)
    resp_d1 = np.gradient(resp)
    resp_d2 = np.gradient(resp_d1)
    six = np.stack([cardiac, cardiac_d1, cardiac_d2, resp, resp_d1, resp_d2], axis=1)
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
