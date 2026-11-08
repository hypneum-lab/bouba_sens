"""Feature extractors for Studyforrest phase-2 streams (Sprint 9 / Task 9.2).

Every function returns a `torch.Tensor` already temporally resampled to
`TARGET_HZ=10` so all 5 modalities + labels align frame-by-frame.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Mapping
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


_PHYSIO_SUFFIX = "_physio.tsv.gz"


def _bids_entities(name: str, suffix: str) -> dict[str, str] | None:
    """Key-value entities of a BIDS file name ending in ``suffix``.

    ``sub-01_task-x_physio.tsv.gz`` gives ``{"sub": "01", "task": "x"}``.
    Returns None when the name does not end in ``suffix`` or a part
    before it is not ``key-value``.
    """
    if not name.endswith(suffix):
        return None
    stem = name.removesuffix(suffix)
    entities: dict[str, str] = {}
    if not stem:
        return entities
    for part in stem.split("_"):
        key, sep, value = part.partition("-")
        if not sep or not key or not value:
            return None
        entities[key] = value
    return entities


def read_physio_sidecar(physio_tsv_gz: Path) -> dict[str, object]:
    """Merged BIDS sidecar metadata for a ``*_physio.tsv.gz`` file.

    BIDS physio TSVs have no header line; their column names and rate
    live in ``*_physio.json`` sidecars, inherited from the dataset root
    down to the file's directory (e.g. ds000113 phase 2 keeps
    ``recording-cardresp_physio.json`` at the root). A sidecar applies
    when its entities are a subset of the file's entities; deeper and
    more specific sidecars override shallower ones. The dataset root is
    the nearest parent holding ``dataset_description.json``; without
    one only the file's own directory is searched.
    """
    import json

    file_entities = _bids_entities(physio_tsv_gz.name, _PHYSIO_SUFFIX)
    if file_entities is None:
        raise ValueError(f"{physio_tsv_gz.name} is not a BIDS *{_PHYSIO_SUFFIX} file")
    directory = physio_tsv_gz.parent
    chain = [directory]
    for parent in directory.parents:
        if (chain[-1] / "dataset_description.json").exists():
            break
        chain.append(parent)
    if not (chain[-1] / "dataset_description.json").exists():
        chain = [directory]
    merged: dict[str, object] = {}
    for level in reversed(chain):  # dataset root first
        candidates: list[tuple[int, Path]] = []
        for path in level.glob("*_physio.json"):
            entities = _bids_entities(path.name, "_physio.json")
            if entities is None:
                continue
            if all(file_entities.get(k) == v for k, v in entities.items()):
                candidates.append((len(entities), path))
        for _, path in sorted(candidates):
            content = json.loads(path.read_text())
            if not isinstance(content, dict):
                raise ValueError(f"sidecar {path} is not a JSON object")
            merged.update({str(k): v for k, v in content.items()})
    return merged


def physio_layout(sidecar: Mapping[str, object], n_cols: int) -> tuple[float, int, int]:
    """Return (sampling Hz, cardiac column, respiratory column).

    Columns are looked up by name in the sidecar ``Columns`` list, never
    assumed from their position. Raises ``ValueError`` when ``Columns``
    or ``SamplingFrequency`` is missing, when ``Columns`` does not match
    the file's column count, or when ``cardiac`` or ``respiratory`` is
    absent.
    """
    columns_raw = sidecar.get("Columns")
    if not isinstance(columns_raw, list):
        raise ValueError("physio sidecar has no 'Columns' list")
    columns = [str(c).lower() for c in columns_raw]
    if len(columns) != n_cols:
        raise ValueError(
            f"sidecar lists {len(columns)} columns {columns} but the file has {n_cols}"
        )
    missing = [name for name in ("cardiac", "respiratory") if name not in columns]
    if missing:
        raise ValueError(f"physio columns {columns} lack {missing}")
    hz_raw = sidecar.get("SamplingFrequency")
    if not isinstance(hz_raw, int | float) or isinstance(hz_raw, bool):
        raise ValueError("physio sidecar has no numeric 'SamplingFrequency'")
    return float(hz_raw), columns.index("cardiac"), columns.index("respiratory")


def extract_force(
    physio_tsv_gz: Path,
    n_frames: int,
    sidecar: Mapping[str, object] | None = None,
) -> torch.Tensor:
    """Cardiac pulse + respiration + first/second derivatives as 6-dim
    autonomic signal (interoceptive proprioception proxy).

    The TSV is read with no header line. Column names and sampling rate
    come from ``sidecar`` when given, else from the BIDS sidecars found
    by ``read_physio_sidecar`` (issue #72: earlier code consumed the
    first data line as a header and fell back to column 0, the trigger,
    as "ecg" and column 1, the pulse, as "resp").

    Returns (n_frames, 6) float32: [cardiac, d/dt, d²/dt², resp,
    d(resp)/dt, d²/dt²]. The first channel keeps its historical name
    "ecg" elsewhere in the code; in ds000113 it is a pulse-oximeter
    trace, not an ECG.
    """
    import gzip

    with gzip.open(physio_tsv_gz, "rt") as fh:
        data = np.loadtxt(fh, delimiter="\t", ndmin=2)  # (T_native, cols)
    meta = read_physio_sidecar(physio_tsv_gz) if sidecar is None else sidecar
    native_hz, ecg_col, resp_col = physio_layout(meta, int(data.shape[1]))
    step_f = native_hz / TARGET_HZ
    if step_f < 1 or abs(step_f - round(step_f)) > 1e-9:
        raise ValueError(
            f"cannot decimate {native_hz} Hz physio to {TARGET_HZ} Hz by an integer step"
        )
    step = round(step_f)  # 50 samples per TARGET_HZ frame at 500 Hz
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
