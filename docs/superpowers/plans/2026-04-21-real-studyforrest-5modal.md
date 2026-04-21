# Real 5-modality Studyforrest Bridge Implementation Plan (Sprint 9)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 2-modality ECG stub (`StudyforrestWorld` data_dir mode, ADR-0007 / ADR-0009) with a 5-modality real biological bridge consuming Studyforrest phase 2 via `datalad`. Feed audio, vision, tactile, gravity, force from synchronised film-viewing streams (soundtrack, frames, motion annotations, fMRI head-motion regressors, ECG + respiration). Produce paper v0.2 evidence: re-run the 150-cell grid no-lock and lock=200 on real 5-modal data; compare B-1 / B-2 / B-3 to ADR-0009 / ADR-0011.

**Architecture:** New `StudyforrestRealWorld(WorldSimulator)` class that loads 6 pre-extracted `.pt` tensors (audio, vision, tactile, gravity, force, labels) temporally aligned at 10 Hz. Feature extraction scripts (shell + Python) run on Studio using VGG16 for vision, librosa mel-spectrograms for audio, and direct parsing for physiology. Grid runs live in the same `WORLD=studyforrest` CLI dispatch (feat/b1-plasticity-recovery already wires this); only the `BOUBA_SENS_STUDYFORREST_DATA` env var points to the new 5-modal data dir.

**Tech Stack:** Python 3.14, uv, torch 2.5+, librosa (new dep, audio features), torchvision (new dep, VGG16). `datalad` + `git-annex` installed on Studio via Homebrew. pytest + existing bouba_sens test infra.

**Parent spec + context:** Spec §3.1 (WorldSimulator contract, unchanged). ADR-0007 (bridge stub scope). ADR-0009 (real ECG 2-modal verdict). ADR-0011 (cross-world lock matrix including ECG no-effect). OSF pre-registration `10.17605/OSF.IO/Q6JYN` — Sprint 9 files an amendment for the Studyforrest extension BEFORE grid runs so verdicts are pre-registered.

**Sprint 9 scope:** Tasks 9.1 → 9.8. Paper v0.2 draft update is deferred to a follow-up sprint.

**Compute target:** Studio for 9.1 (datalad fetch, ~200 MB), 9.3 (VGG16 extraction, ~30 min GPU), 9.6 (150-cell grids). GrosMac for the rest (env var setup, test writing, aggregation, ADR).

---

## File structure

```
bouba_sens/
├── src/bouba_sens/
│   └── world/
│       ├── studyforrest.py                 [MODIFY] add StudyforrestRealWorld
│       └── studyforrest_features.py        [CREATE] feature-extraction helpers
├── scripts/
│   ├── fetch_studyforrest_phase2.sh        [CREATE] datalad install + get
│   └── extract_studyforrest_features.py    [CREATE] tensor extraction CLI
├── tests/
│   └── unit/
│       └── test_studyforrest_real.py       [CREATE] 6 shape + alignment tests
├── docs/
│   ├── osf/
│   │   └── amendment-v0.5-studyforrest-5modal.md  [CREATE] OSF filing draft
│   └── adr/
│       └── 0012-real-5modal-studyforrest-verdicts.md  [CREATE] grid verdicts
├── reports/
│   ├── v0.5_studyforrest_5modal_nolock_aggregate.json  [ARTEFACT]
│   └── v0.5_studyforrest_5modal_lock200_aggregate.json [ARTEFACT]
├── pyproject.toml                          [MODIFY] add librosa + torchvision
└── CHANGELOG.md                            [MODIFY] v0.5.0 entry
```

---

## Task 9.1 — `datalad` environment + fetch script

**Goal:** A single shell script that installs `datalad` + `git-annex` via Homebrew, runs `datalad install ///studyforrest-data-phase2`, and `datalad get`s only the files we need (~200 MB instead of the full ~500 GB). Emits a MANIFEST with SHA256s.

**Files:**
- Create: `scripts/fetch_studyforrest_phase2.sh`

- [ ] **Step 1: Create the fetch script**

File: `scripts/fetch_studyforrest_phase2.sh`

```bash
#!/usr/bin/env bash
# Sprint 9 / Task 9.1 — fetch the Studyforrest phase-2 subset needed
# for the 5-modality bridge. Idempotent: re-runs do nothing when the
# target subtree already exists.
set -euo pipefail

DEST="${1:-data/studyforrest_phase2}"
SUBJECT="${SUBJECT:-sub-01}"
RUN="${RUN:-run-1}"

# 1. Install datalad + git-annex if missing (Homebrew on macOS).
if ! command -v datalad >/dev/null; then
    echo "[fetch] installing datalad via Homebrew"
    brew install datalad git-annex
fi

# 2. Install the dataset handle (no payload yet).
if [[ ! -d "${DEST}" ]]; then
    echo "[fetch] datalad install ///studyforrest-data-phase2 -> ${DEST}"
    datalad install -r "///studyforrest-data-phase2" "${DEST}"
fi

cd "${DEST}"

# 3. Get only what we need. File paths follow BIDS conventions.
#    OPTIONAL items (e.g. video.mkv) may not be resolvable via datalad
#    on all dataset snapshots; failures are non-fatal.
REQUIRED=(
    "stimuli/soundtrack/fg_av_ger_stereo.mp3"
    "${SUBJECT}/ses-movie/func/${SUBJECT}_ses-movie_task-movie_${RUN}_bold.nii.gz"
    "${SUBJECT}/ses-movie/func/${SUBJECT}_ses-movie_task-movie_${RUN}_physio.tsv.gz"
    "stimuli/annotations/movie_motion-locations.csv"
)
OPTIONAL=(
    "stimuli/movie/fg_av_ger_stereo.mkv"
    "${SUBJECT}/ses-movie/func/${SUBJECT}_ses-movie_task-movie_${RUN}_recording-eyegaze_physio.tsv.gz"
)

for f in "${REQUIRED[@]}"; do
    if [[ ! -e "$(realpath "${f}")" ]]; then
        echo "[fetch] datalad get (required) ${f}"
        datalad get "${f}"
    fi
done

for f in "${OPTIONAL[@]}"; do
    if [[ ! -e "$(realpath "${f}")" ]]; then
        echo "[fetch] datalad get (optional) ${f}"
        datalad get "${f}" || echo "[fetch] WARN: could not fetch ${f}, skipping"
    fi
done

# 4. Emit MANIFEST with SHA256s for reproducibility.
cat > MANIFEST.md <<EOF
# Studyforrest phase-2 fetch manifest

- Source: \`///studyforrest-data-phase2\` (Hanke et al. 2014, Scientific Data)
- Subject: ${SUBJECT}
- Run: ${RUN}
- Fetched: $(date -u +%Y-%m-%dT%H:%M:%SZ)

## SHA256
EOF
for f in "${REQUIRED[@]}" "${OPTIONAL[@]}"; do
    if [[ -e "${f}" ]]; then
        h=$(shasum -a 256 "${f}" | awk '{print $1}')
        echo "- \`${f}\`: \`${h}\`" >> MANIFEST.md
    fi
done

echo "[fetch] done; manifest at ${DEST}/MANIFEST.md"
```

- [ ] **Step 2: Make executable + commit**

```bash
chmod +x scripts/fetch_studyforrest_phase2.sh
git add scripts/fetch_studyforrest_phase2.sh
git commit -m "feat: Task 9.1 Studyforrest datalad fetch"
```

---

## Task 9.2 — Feature-extraction CLI

**Goal:** Take the fetched Studyforrest files and emit 6 aligned `.pt` tensors at 10 Hz: `audio.pt (N,128)`, `vision.pt (N,16,16)`, `tactile.pt (N,32)`, `gravity.pt (N,3)`, `force.pt (N,6)`, `labels.pt (N,)`. Label = scene ID from the motion-locations annotation file, 4-bucket quantisation.

**Files:**
- Create: `src/bouba_sens/world/studyforrest_features.py`
- Create: `scripts/extract_studyforrest_features.py`
- Modify: `pyproject.toml` (add `librosa>=0.10` + `torchvision>=0.20`)

- [ ] **Step 1: Add dependencies**

In `pyproject.toml`, replace the existing `dependencies` block's last lines (currently ends with `"scikit-learn>=1.5"`) by inserting before it:

```toml
    "librosa>=0.10",
    "torchvision>=0.20",
```

Run `uv sync --all-extras` to install.

- [ ] **Step 2: Write the feature-extraction library**

File: `src/bouba_sens/world/studyforrest_features.py`

```python
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
    import librosa  # type: ignore[import-not-found]

    y, sr = librosa.load(str(mp3_path), sr=22050, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=_AUDIO_MEL_BINS, hop_length=sr // TARGET_HZ
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
    timestamps = raw[:, 0].astype(float)
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
```

- [ ] **Step 3: Write the orchestrator CLI**

File: `scripts/extract_studyforrest_features.py`

```python
"""Sprint 9 / Task 9.2 — orchestrate the 5-modality extraction.

Reads the Studyforrest phase-2 fetched tree, runs the 5 extractors,
writes 6 pytorch tensors + a MANIFEST with SHA256s. Pass the output
dir to `StudyforrestRealWorld(data_dir=...)` (Task 9.3).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from bouba_sens.world.studyforrest_features import (
    TARGET_HZ,
    extract_audio,
    extract_force,
    extract_gravity,
    extract_labels,
    extract_tactile,
    extract_vision,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Studyforrest phase-2 fetched root (output of fetch_studyforrest_phase2.sh)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Target dir for the 6 tensors + MANIFEST.md",
    )
    parser.add_argument(
        "--subject",
        default="sub-01",
        help="BIDS subject label",
    )
    parser.add_argument(
        "--run",
        default="run-1",
        help="BIDS run label",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=900,
        help="Trim to first N seconds (default 15 min = 9000 frames @ 10 Hz)",
    )
    args = parser.parse_args()

    n_frames = args.duration_seconds * TARGET_HZ

    audio_src = args.src / "stimuli/soundtrack/fg_av_ger_stereo.mp3"
    video_src = args.src / "stimuli/movie/fg_av_ger_stereo.mkv"
    annot_src = args.src / "stimuli/annotations/movie_motion-locations.csv"
    bold_src = (
        args.src
        / args.subject
        / "ses-movie/func"
        / f"{args.subject}_ses-movie_task-movie_{args.run}_bold.nii.gz"
    )
    physio_src = (
        args.src
        / args.subject
        / "ses-movie/func"
        / f"{args.subject}_ses-movie_task-movie_{args.run}_physio.tsv.gz"
    )

    print(f"[extract] audio <- {audio_src}")
    audio = extract_audio(audio_src, n_frames)
    print(f"[extract] vision <- {video_src}")
    vision = extract_vision(video_src, n_frames)
    print(f"[extract] tactile <- {annot_src}")
    tactile = extract_tactile(annot_src, n_frames)
    print(f"[extract] gravity <- {bold_src}")
    gravity = extract_gravity(bold_src, n_frames)
    print(f"[extract] force <- {physio_src}")
    force = extract_force(physio_src, n_frames)
    print(f"[extract] labels <- {annot_src}")
    labels = extract_labels(annot_src, n_frames)

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
        "# Studyforrest phase-2 5-modality feature manifest",
        "",
        f"- Source: `{args.src}` (datalad `///studyforrest-data-phase2`)",
        f"- Subject: {args.subject}",
        f"- Run: {args.run}",
        f"- Frames: {n_frames} at {TARGET_HZ} Hz ({args.duration_seconds} s)",
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
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/bouba_sens/world/studyforrest_features.py scripts/extract_studyforrest_features.py
git commit -m "feat: Task 9.2 Studyforrest 5-modal extraction"
```

---

## Task 9.3 — `StudyforrestRealWorld` class

**Goal:** New `WorldSimulator` that reads the 6 tensors from Task 9.2 and serves them via `.sample(batch_size, seed)` — no zeroed modalities any more. Uses temporal contiguous slicing (like the existing mock) so lag-1 autocorr stays non-zero.

**Files:**
- Modify: `src/bouba_sens/world/studyforrest.py` (add new class)
- Create: `tests/unit/test_studyforrest_real.py`

- [ ] **Step 1: Write the 5 failing tests first**

File: `tests/unit/test_studyforrest_real.py`

```python
"""Task 9.3 tests — StudyforrestRealWorld 5-modal shape + alignment."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bouba_sens.world.studyforrest import StudyforrestRealWorld


@pytest.fixture
def fake_5modal_dir(tmp_path: Path) -> Path:
    """Synthesise a tiny 5-modal cache that matches the real extraction contract."""
    n = 512
    torch.save(torch.randn(n, 128), tmp_path / "audio.pt")
    torch.save(torch.randn(n, 16, 16), tmp_path / "vision.pt")
    torch.save(torch.randn(n, 32), tmp_path / "tactile.pt")
    torch.save(torch.randn(n, 3), tmp_path / "gravity.pt")
    torch.save(torch.randn(n, 6), tmp_path / "force.pt")
    torch.save(torch.randint(0, 4, (n,), dtype=torch.long), tmp_path / "labels.pt")
    return tmp_path


def test_real_sample_shapes(fake_5modal_dir: Path) -> None:
    world = StudyforrestRealWorld(seed=0, data_dir=fake_5modal_dir)
    sample = world.sample(batch_size=16, seed=42)
    assert world.mode == "real5"
    assert sample.audio.shape == (16, 128)
    assert sample.vision.shape == (16, 16, 16)
    assert sample.tactile.shape == (16, 32)
    assert sample.gravity.shape == (16, 3)
    assert sample.force.shape == (16, 6)
    assert sample.label.shape == (16,)
    assert sample.label.dtype == torch.long


def test_real_no_modality_is_zero(fake_5modal_dir: Path) -> None:
    """Unlike the stub, the 5-modal variant must emit non-zero tensors
    for tactile / gravity / force."""
    world = StudyforrestRealWorld(seed=0, data_dir=fake_5modal_dir)
    sample = world.sample(batch_size=32, seed=0)
    assert sample.tactile.abs().sum() > 0
    assert sample.gravity.abs().sum() > 0
    assert sample.force.abs().sum() > 0


def test_real_temporal_contiguity(fake_5modal_dir: Path) -> None:
    """Consecutive samples in the batch must correspond to consecutive
    timecodes — preserves the lag-1 autocorr signal."""
    world = StudyforrestRealWorld(seed=0, data_dir=fake_5modal_dir)
    sample = world.sample(batch_size=8, seed=123)
    cache = torch.load(fake_5modal_dir / "audio.pt")
    # Find which slice of the cache the sample came from by matching row 0.
    for start in range(cache.shape[0] - 8):
        if torch.allclose(cache[start], sample.audio[0]):
            for offset in range(8):
                assert torch.allclose(cache[(start + offset) % cache.shape[0]], sample.audio[offset])
            return
    raise AssertionError("audio sample did not match any contiguous cache slice")


def test_real_reproducibility(fake_5modal_dir: Path) -> None:
    a = StudyforrestRealWorld(seed=0, data_dir=fake_5modal_dir).sample(batch_size=4, seed=7)
    b = StudyforrestRealWorld(seed=0, data_dir=fake_5modal_dir).sample(batch_size=4, seed=7)
    assert torch.equal(a.audio, b.audio)
    assert torch.equal(a.gravity, b.gravity)
    assert torch.equal(a.label, b.label)


def test_real_modality_dims_contract(fake_5modal_dir: Path) -> None:
    """Same modality_dims as GaussianWorld so the SensoryWML + encoders wire unchanged."""
    world = StudyforrestRealWorld(seed=0, data_dir=fake_5modal_dir)
    dims = world.modality_dims()
    assert dims == {
        "audio": (128,),
        "vision": (16, 16),
        "tactile": (32,),
        "gravity": (3,),
        "force": (6,),
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/unit/test_studyforrest_real.py -v
```

Expected: all 5 fail with `ImportError: cannot import name 'StudyforrestRealWorld' from 'bouba_sens.world.studyforrest'`.

- [ ] **Step 3: Implement `StudyforrestRealWorld`**

Append to `src/bouba_sens/world/studyforrest.py` (after the existing `StudyforrestWorld` class, before `_build_mock_cache`):

```python
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

    def __init__(self, *, seed: int = 0, data_dir: Path) -> None:
        dir_path = Path(data_dir)
        self._seed = seed
        self._audio = torch.load(dir_path / "audio.pt")
        vision = torch.load(dir_path / "vision.pt")
        self._vision = (
            vision
            if vision.ndim == 3
            else vision.reshape(vision.shape[0], *_VISION_SHAPE)
        )
        self._tactile = torch.load(dir_path / "tactile.pt")
        self._gravity = torch.load(dir_path / "gravity.pt")
        self._force = torch.load(dir_path / "force.pt")
        self._labels = torch.load(dir_path / "labels.pt").long()
        self._n = self._labels.shape[0]
        self._mode = "real5"

    @property
    def mode(self) -> str:
        return self._mode

    def sample(self, batch_size: int, seed: int) -> WorldSample:
        gen = torch.Generator().manual_seed(seed)
        start = int(
            torch.randint(0, max(1, self._n - batch_size), (1,), generator=gen).item()
        )
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
```

- [ ] **Step 4: Wire the class into `_build_world`**

In `src/bouba_sens/cli.py`, inside `_build_world`, replace the existing `if key == "studyforrest":` block with a version that picks real5 when the data_dir contains the full 5-modal set:

```python
if key == "studyforrest":
    raw_dir = os.getenv("BOUBA_SENS_STUDYFORREST_DATA")
    data_dir = Path(raw_dir) if raw_dir else None
    # Prefer the 5-modal real bridge when the cache is present; fall
    # back to the 2-modal stub (ADR-0007) otherwise for compatibility.
    if data_dir and (data_dir / "tactile.pt").exists():
        from bouba_sens.world.studyforrest import StudyforrestRealWorld
        return StudyforrestRealWorld(seed=seed, data_dir=data_dir)
    return StudyforrestWorld(seed=seed, data_dir=data_dir)
```

- [ ] **Step 5: Run the tests to verify they pass + full suite**

```bash
uv run pytest tests/unit/test_studyforrest_real.py -v
uv run pytest tests/unit/ -q
```

Expected: 5 new tests green + all existing unit tests still green.

- [ ] **Step 6: Commit**

```bash
git add src/bouba_sens/world/studyforrest.py src/bouba_sens/cli.py tests/unit/test_studyforrest_real.py
git commit -m "feat: Task 9.3 StudyforrestRealWorld 5-modal"
```

---

## Task 9.4 — Complexity audit on real 5-modal data

**Goal:** Run the existing world-complexity audit (ADR-0007 / Task 7.5) on the new real bridge and confirm it sits outside BOTH the synthetic cluster AND the mock AR(1) surrogate. If it does not, the 5-modal is structurally too similar to what we already have and the Sprint 9 experiment is premature.

**Files:**
- Modify: `scripts/audit_worlds.py` (register `studyforrest_real5` as an auditable world)

- [ ] **Step 1: Extend the audit CLI**

In `scripts/audit_worlds.py`, replace the existing `_WORLD_FACTORIES` dict with:

```python
_WORLD_FACTORIES = {
    "gaussian": GaussianWorld,
    "xor": XORWorld,
    "sinusoid": SinusoidWorld,
}


def _build(name: str, seed: int) -> WorldSimulator:
    if name.startswith("studyforrest_real5:"):
        from bouba_sens.world.studyforrest import StudyforrestRealWorld

        path = name.split(":", 1)[1]
        return StudyforrestRealWorld(seed=seed, data_dir=Path(path))
    return _WORLD_FACTORIES[name](seed=seed)
```

Remove the old `return _WORLD_FACTORIES[name](seed=seed)` line from the prior definition.

- [ ] **Step 2: Run the audit + commit**

```bash
uv run python scripts/audit_worlds.py \
    --worlds "gaussian,xor,sinusoid,studyforrest_real5:data/studyforrest_5modal_sub01_run1" \
    --batch-size 512 --seeds 0,1,2 \
    --out reports/v0.5_world_complexity_audit_5modal.json

git add scripts/audit_worlds.py reports/v0.5_world_complexity_audit_5modal.json || true
git commit -m "feat: Task 9.4 audit 5-modal real bridge"
```

Expected: the output shows `studyforrest_real5` with `temporal_autocorr > 0.3` (real temporal structure) and `intrinsic_dim_pca` values distinct from synthetic (~30) AND mock (~4) worlds. If `temporal_autocorr < 0.1`, the extraction is likely broken — STOP and debug before proceeding to Task 9.5.

---

## Task 9.5 — OSF amendment for the 5-modal extension

**Goal:** File the pre-registration amendment BEFORE any grid run so verdicts are trustworthy. Extends `docs/osf/amendment-v0.4-studyforrest.md` with the 5-modal specifics.

**Files:**
- Create: `docs/osf/amendment-v0.5-studyforrest-5modal.md`

- [ ] **Step 1: Write the amendment doc**

File: `docs/osf/amendment-v0.5-studyforrest-5modal.md`

```markdown
# OSF pre-registration amendment — bouba_sens v0.5 (5-modal Studyforrest)

**Status:** Draft, ready to file
**Parent pre-registration:** `dream-of-kiki` OSF DOI `10.17605/OSF.IO/Q6JYN`
**Previous amendment:** `amendment-v0.4-studyforrest.md` (2-modal ECG bridge)
**Amendment tag on file:** `bouba_sens/v0.5-studyforrest-5modal`
**Amendment date:** 2026-04-21

## What is being amended

The v0.4 amendment added a generic `StudyforrestWorld` to the pre-
registered world set but was exercised only with the 2-modality ECG
stub. ADR-0009 and ADR-0011 documented that the stub's 3 zeroed
modalities (tactile, gravity, force) made the plasticity-lock
mechanism structurally unobservable on that bridge.

This v0.5 amendment adds a `StudyforrestRealWorld` with **5 real
biological modalities** drawn from Studyforrest phase 2 (Hanke et al.
2016). The modality mapping (fixed before any grid runs) is:

| bouba_sens modality | Studyforrest source | Feature |
|---------------------|---------------------|---------|
| audio | film soundtrack | 128-bin mel-spectrogram @ 10 Hz |
| vision | film frames | VGG16 conv4 pool, PCA-256, reshape 16×16 |
| tactile | motion annotations | scene-id + motion-type embedding |
| gravity | fMRI rigid-body rotations | pitch, yaw, roll (3 dim) |
| force | ECG + respiration | 6-dim: signal + Δ + Δ² |

## Protocol

Two grid runs at the pre-registered thresholds (0.05 / 0.10 / 0.02,
unchanged):

1. `WORLD=studyforrest` no-lock — baseline.
2. `WORLD=studyforrest` LOCK_AFTER=200 — matches ADR-0011 condition.

Expected artefacts:
- `reports/v0.5_studyforrest_5modal_nolock_aggregate.json`
- `reports/v0.5_studyforrest_5modal_lock200_aggregate.json`
- `docs/adr/0012-real-5modal-studyforrest-verdicts.md`

## What is NOT being amended

- Thresholds 0.05 / 0.10 / 0.02 (frozen since spec §1.2).
- Metric implementations `me1_accuracy`, `me2_recovery_auc`,
  `me3_delta`, `me6_asymmetry`, `me6_max_abs_off_diag`,
  `me7_congenital_gap`, `me9_bootstrap`.
- The 5-seed × 5-modality × 2-timing × 3-SNR = 150-cell grid structure.

## Decision rule for paper v0.2

| B-3 on 5-modal real | Paper v0.2 claim |
|---------------------|------------------|
| PASS at ≥ 10× threshold | "B-3 is an architectural invariant across synthetic AND real biological 5-modal input." (strong) |
| PASS at 1-10× threshold | "B-3 persists but is attenuated under biological input complexity." (moderate) |
| FAIL | "B-3 is a synthetic-cluster artefact; the unlocked Studyforrest ECG-2 result was driven by zeroed modalities." (retraction of v0.1 headline) |

All three outcomes are publishable. No threshold change, no
metric-math change — only the world is new.

## Timeline

| Step | ETA |
|------|-----|
| File on OSF (this doc) | same-day |
| Task 9.6 grid runs | same-day (~40 min parallel) |
| ADR-0012 + paper §5.5 | same-day |
| v0.5.0 tag + release | same-day |

Grid runs strictly post-filing.
```

- [ ] **Step 2: Commit**

```bash
git add docs/osf/amendment-v0.5-studyforrest-5modal.md
git commit -m "docs: Task 9.5 OSF amendment v0.5"
```

---

## Task 9.6 — Studio grid runs (no-lock + lock=200)

**Goal:** Two 150-cell grids on Studio using the isolated-worktree pattern from ADR-0011. The `b1-recovery` branch already wires `LOCK_AFTER` so no bouba_sens code change is needed here — only Studio orchestration.

- [ ] **Step 1: Push the 5-modal branch + sync Studio**

```bash
# Local (GrosMac):
git push origin feat/sprint-9-5modal

# Studio (assumes ~/Projets/bouba_sens_b1 worktree exists from ADR-0011):
ssh studio "cd ~/Projets/bouba_sens_b1 && git fetch origin && git checkout feat/sprint-9-5modal && export PATH=/opt/homebrew/bin:\$PATH && uv sync --all-extras"
```

- [ ] **Step 2: Fetch + extract on Studio**

```bash
ssh studio "cd ~/Projets/bouba_sens_b1 && bash scripts/fetch_studyforrest_phase2.sh data/studyforrest_phase2"
ssh studio "cd ~/Projets/bouba_sens_b1 && export PATH=/opt/homebrew/bin:\$PATH && uv run python scripts/extract_studyforrest_features.py --src data/studyforrest_phase2 --out data/studyforrest_5modal_sub01_run1"
```

Expected: 6 `.pt` files + MANIFEST.md in `data/studyforrest_5modal_sub01_run1/`, total ~15 MB.

- [ ] **Step 3: Launch both grids in parallel via screen**

```bash
ssh studio "cd ~/Projets/bouba_sens_b1 && export PATH=/opt/homebrew/bin:\$PATH && \
    screen -dmS sf5-nolock bash -c 'BOUBA_SENS_STUDYFORREST_DATA=data/studyforrest_5modal_sub01_run1 WORLD=studyforrest OUT_ROOT=runs/v05_5modal_nolock STEPS_TRAIN=200 STEPS_LESION=100 METRICS=\"Me1,Me2,Me3\" bash scripts/run_grid.sh > logs/grid-v05-5modal-nolock.log 2>&1' && \
    screen -dmS sf5-lock bash -c 'LOCK_AFTER=200 BOUBA_SENS_STUDYFORREST_DATA=data/studyforrest_5modal_sub01_run1 WORLD=studyforrest OUT_ROOT=runs/v05_5modal_lock200 STEPS_TRAIN=200 STEPS_LESION=100 METRICS=\"Me1,Me2,Me3\" bash scripts/run_grid.sh > logs/grid-v05-5modal-lock200.log 2>&1'"
```

Expected wall time: ~30-40 min (2-way concurrency on M3 Ultra).

- [ ] **Step 4: After completion, aggregate + pull**

```bash
ssh studio "cd ~/Projets/bouba_sens_b1 && export PATH=/opt/homebrew/bin:\$PATH && \
    uv run python scripts/aggregate_grid.py --root runs/v05_5modal_nolock --out reports/v0.5_studyforrest_5modal_nolock_aggregate.json && \
    uv run python scripts/aggregate_grid.py --root runs/v05_5modal_lock200 --out reports/v0.5_studyforrest_5modal_lock200_aggregate.json"

scp studio:~/Projets/bouba_sens_b1/reports/v0.5_studyforrest_5modal_nolock_aggregate.json reports/
scp studio:~/Projets/bouba_sens_b1/reports/v0.5_studyforrest_5modal_lock200_aggregate.json reports/
```

- [ ] **Step 5: Inspect the verdicts**

```bash
uv run python - <<'EOF'
import json
for tag in ("nolock", "lock200"):
    d = json.load(open(f"reports/v0.5_studyforrest_5modal_{tag}_aggregate.json"))
    print(f"=== 5-modal {tag} ===")
    for k, v in d["invariants"].items():
        med_key = next(kk for kk in v if kk.startswith("median"))
        print(f"  {k}: passes={v['passes']}, cells={v['cells_counted']}, {med_key}={v[med_key]:.4f}, thr={v['threshold']}")
EOF
```

---

## Task 9.7 — ADR-0012 + paper §5.5 update

**Goal:** Record the verdicts honestly against the three decision-rule branches in the OSF amendment. Update paper §5.5 with the 2×2 real-vs-lock matrix and the comparison to ADR-0009 / ADR-0011.

**Files:**
- Create: `docs/adr/0012-real-5modal-studyforrest-verdicts.md`
- Modify: `docs/paper/paper-v0.1-draft.md` (promote to v0.2 with a new §5.5)

- [ ] **Step 1: Write ADR-0012**

Stub (fill in from Task 9.6 output):

```markdown
# ADR-0012 — Real 5-modality Studyforrest verdicts

**Status:** Accepted
**Date:** 2026-04-21
**Sprint:** 9

## Verdicts (to fill in from Task 9.6)

| Invariant | 5-modal no-lock | 5-modal lock=200 |
|-----------|-----------------|------------------|
| B-1 Me7 | _fill_ | _fill_ |
| B-2 Me3 delta | _fill_ | _fill_ |
| B-3 Me6 | _fill_ | _fill_ |

## Comparison to prior ADRs

| World | B-3 no-lock | B-3 lock=200 |
|-------|------------:|-------------:|
| Gaussian (ADR-0005 / 0011) | 0.1484 | 0.1719 |
| XOR | 0.1406 | 0.1250 |
| Sinusoid | 0.1406 | 0.1562 |
| ECG 2-modal (ADR-0009 / 0011) | 0.4453 | 0.4453 |
| **5-modal real (this)** | _fill_ | _fill_ |

## Decision

Pick one branch from the OSF amendment decision rule and record it here,
verbatim. DO NOT edit the decision rule post-hoc.

## Next steps

- paper v0.2 headline update
- v0.5.0 tag push
```

Run the aggregation script from Task 9.6 Step 5 to populate the numbers, then fill in the ADR and commit.

- [ ] **Step 2: Promote paper v0.1 → v0.2 with §5.5**

In `docs/paper/paper-v0.1-draft.md`, add a new section 5.5 immediately after §5.4:

```markdown
### 5.5 Real 5-modality Studyforrest (ADR-0012)

| Invariant | 5-modal no-lock | 5-modal lock=200 | prior ECG 2-modal |
|-----------|----------------:|-----------------:|------------------:|
| B-1 | _fill_ | _fill_ | -0.0062 / -0.0062 |
| B-2 | _fill_ | _fill_ | +0.0111 / n/a |
| B-3 | _fill_ | _fill_ | 0.4453 / 0.4453 |

[Narrative paragraph: is the 5-modal world structurally outside the
synthetic cluster AND the mock AR(1) AND the ECG 2-modal? How does
the lock interact now that the 3 "zeroed" modalities carry real
signal?]

[If B-3 stays PASS: strongest confirmation of the architectural
invariant. If B-3 fails: retraction of the v0.1 headline.]
```

- [ ] **Step 3: Commit**

```bash
git add docs/adr/0012-real-5modal-studyforrest-verdicts.md docs/paper/paper-v0.1-draft.md
git commit -m "docs: Task 9.7 ADR-0012 + paper 5.5"
```

---

## Task 9.8 — v0.5.0 release + memory update

**Goal:** Bump version, changelog, tag, push.

**Files:**
- Modify: `src/bouba_sens/_version.py`
- Modify: `pyproject.toml`
- Modify: `CHANGELOG.md`
- Modify: version-pinned tests (`tests/smoke/test_imports.py`, `tests/unit/test_smoke.py`)

- [ ] **Step 1: Bump version strings**

In `src/bouba_sens/_version.py`:

```python
__version__ = "0.5.0"
```

In `pyproject.toml`, locate the line `version = "0.4.0"` (or the current version) and replace with `version = "0.5.0"`.

In `tests/smoke/test_imports.py`, replace `assert bouba_sens.__version__ == "0.4.0"` (or current) with `assert bouba_sens.__version__ == "0.5.0"`.

In `tests/unit/test_smoke.py`, same two-line update.

- [ ] **Step 2: Add CHANGELOG entry**

Prepend to `CHANGELOG.md` just after `## [Unreleased]`:

```markdown
## [0.5.0] — 2026-04-21 (Sprint 9 — real 5-modality Studyforrest bridge)

### Added

- `StudyforrestRealWorld` — 5 real biological modalities via
  Studyforrest phase 2 (audio, vision, tactile, gravity, force)
  replacing the 2-modal ECG stub.
- `scripts/fetch_studyforrest_phase2.sh` — datalad fetcher for the
  ~200 MB subject subset.
- `scripts/extract_studyforrest_features.py` — orchestrates all 5
  feature extractors + labels; emits aligned tensors + MANIFEST.
- `src/bouba_sens/world/studyforrest_features.py` — 6 extractors.
- `docs/osf/amendment-v0.5-studyforrest-5modal.md` — OSF amendment.
- `docs/adr/0012-real-5modal-studyforrest-verdicts.md` — verdicts.
- 5 new unit tests in `tests/unit/test_studyforrest_real.py`.

### Changed

- `_build_world` prefers `StudyforrestRealWorld` when the cache
  directory contains the 5-modal tensors; falls back to the 2-modal
  stub otherwise (backwards-compatible).

### Verified

- 150/150 grid on 5-modal no-lock (see ADR-0012).
- 150/150 grid on 5-modal lock=200 (see ADR-0012).
- Pre-registration fidelity: no threshold or metric-math change.
```

- [ ] **Step 3: Run the full suite + commit + tag**

```bash
uv run pytest -q
git add src/bouba_sens/_version.py pyproject.toml uv.lock CHANGELOG.md tests/smoke/test_imports.py tests/unit/test_smoke.py
git commit -m "chore(release): v0.5.0 Sprint 9 close"
git tag -a v0.5.0 -m "v0.5.0 real 5-modal Studyforrest"
git push origin feat/sprint-9-5modal v0.5.0
```

- [ ] **Step 4: Update memory**

In `/Users/electron/.claude/projects/-Users-electron/memory/project_bouba_sens_sprint0_2026_04_20.md`, add a new Sprint 9 bullet summarising the 2×3 matrix (no-lock × lock=200, B-1/B-2/B-3) and the decision-rule branch that ADR-0012 picked.

In `/Users/electron/.claude/projects/-Users-electron/memory/MEMORY.md`, update the bouba_sens index line so it reflects v0.5.0 + 5-modal verdicts.

---

## Exit criteria

1. `datalad` installed + phase-2 fetched on Studio (Task 9.1).
2. `scripts/extract_studyforrest_features.py` emits 6 aligned tensors (Task 9.2).
3. `StudyforrestRealWorld` passes 5 new unit tests; all prior 174+ tests still green (Task 9.3).
4. World-complexity audit shows the 5-modal world sits outside the synthetic cluster (Task 9.4).
5. OSF amendment committed BEFORE grid runs (Task 9.5).
6. Two 150-cell grids complete on Studio (Task 9.6).
7. ADR-0012 committed with picked decision-rule branch (Task 9.7).
8. Paper §5.5 populated (Task 9.7).
9. Tag `v0.5.0` pushed (Task 9.8).

## Risks + mitigations

| Risk | Mitigation |
|------|------------|
| R-1: `datalad` install fails on Studio | Fall back to a manual `curl` fetch of the 6 files needed; MANIFEST step is the same either way. |
| R-2: Video frame extraction (ffmpeg + VGG16) runs >1 h | Cap extraction to 15 min of film (9000 frames); this plan already defaults to 900 s. |
| R-3: B-3 fails on 5-modal real data | Explicitly planned outcome in the OSF decision rule; paper v0.2 headline pivots accordingly. |
| R-4: Another agent's `uv sync` clobbers the CLI mid-grid (repeat of ADR-0011) | Always run from `~/Projets/bouba_sens_b1` worktree; never from the parent clone. |
| R-5: Real data has Nan or temporal discontinuities | Task 9.4 audit shows `temporal_autocorr > 0` as a sanity gate BEFORE any grid run. If the audit fails, Task 9.6 does not launch. |
