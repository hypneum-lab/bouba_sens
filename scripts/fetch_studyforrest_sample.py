"""Fetch + extract a Studyforrest sample (Sprint 7 / Task 7.6; Sprint 9 bridge).

Two sources, selected with ``--source``:

* ``--source openneuro`` (DEFAULT) — pull the **real** Studyforrest dataset
  (``ds000113``) directly from the OpenNeuro S3 bucket over the public REST
  API and build a 5-modality (+labels) cache that
  ``StudyforrestRealWorld(data_dir=...)`` consumes. Produces SIX tensors:

      <data_dir>/audio.pt    # (N, 128) float32
      <data_dir>/vision.pt   # (N, 16, 16) float32
      <data_dir>/tactile.pt  # (N, 32) float32
      <data_dir>/gravity.pt  # (N, 3) float32
      <data_dir>/force.pt    # (N, 6) float32
      <data_dir>/labels.pt   # (N,) long
      <data_dir>/MANIFEST.md # SHA256s + real S3 URLs + per-channel provenance

  PROVENANCE — read carefully, nothing here is dressed up as more than it is:

  - ``force``   : REAL biosignal. ds000113 cardiac (pulse-oximeter) +
                  respiration belt physio, 500 Hz, columns
                  ``[trigger, cardiac, respiratory]`` (per
                  ``recording-cardresp_physio.json``). 6 dims =
                  [cardiac, d/dt, d²/dt², resp, d/dt, d²/dt²].
  - ``gravity`` : REAL head-motion regressors. ds000113
                  ``derivatives/motion/.../*_bold_moco.txt`` (6-col
                  rigid-body moco params); we keep the 3 rotation columns.
  - ``tactile`` : REAL annotation. ds000113 ``stimuli/annotations/scenes.csv``
                  (scene name + DAY/NIGHT + INT/EXT) one-hot → 32-D random
                  projection (seed 0). A scene-context proxy, not skin
                  mechanoreception — Studyforrest has no tactile channel.
  - ``labels``  : REAL annotation. scenes.csv scene index, quantised to 4.
  - ``audio``   : DERIVED PROXY (NOT the soundtrack). The Forrest Gump
                  soundtrack is copyrighted and is **not** in ds000113, so a
                  real mel-spectrogram is impossible from OpenNeuro alone.
                  We synthesise a 128-D feature from the REAL timed
                  ``german_audio_description.csv`` narration (per-frame
                  narration on/off, segment phase, local narration density)
                  expanded by a fixed random projection. Honest label in the
                  MANIFEST: "annotation-derived narration proxy".
  - ``vision``  : DERIVED PROXY (NOT the movie). The movie is copyrighted and
                  absent from ds000113. We synthesise a (16,16) feature from
                  the REAL scenes.csv transition structure (scene id, EXT/INT,
                  DAY/NIGHT, time-since-cut) expanded by a fixed projection.
                  Honest label: "annotation-derived scene-structure proxy".

  So 4 of 6 channels carry real ds000113 signal; audio+vision are explicitly
  annotation-derived proxies because the perceptual stimuli are not
  redistributable. This is a *real bridge* for force/gravity/tactile/labels
  and a *proxy* for audio/vision — both stated verbatim in the manifest.

* ``--source surrogate`` — offline, network-free fallback. Builds the same
  6 tensors from the deterministic AR(1) mock (``--source surrogate``) or the
  scipy MIT-BIH ECG bridge. **NOT scientific data**; labelled as such in the
  manifest. Used only when S3 is unreachable.

Design principles: pure numpy/torch (no librosa/torchvision/nibabel/ffmpeg
needed — the real channels are all CSV/TSV/text, no media decoding);
idempotent (existing MANIFEST.md skips re-download unless ``--force``).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------
# OpenNeuro ds000113 (Studyforrest) — real S3 object keys.
# --------------------------------------------------------------------------
_S3_ROOT = "https://s3.amazonaws.com/openneuro.org/"
_DS = "ds000113"
_SUBJECT = "sub-01"

_SCENES_KEY = f"{_DS}/stimuli/annotations/scenes.csv"
_NARRATION_KEY = f"{_DS}/stimuli/annotations/german_audio_description.csv"
# 8 movie runs; physio (cardresp) + moco (motion) per run.
_N_RUNS = 8


def _physio_key(run: int) -> str:
    return (
        f"{_DS}/{_SUBJECT}/ses-forrestgump/func/"
        f"{_SUBJECT}_ses-forrestgump_task-forrestgump_run-{run:02d}_physio.tsv.gz"
    )


def _moco_key(run: int) -> str:
    return (
        f"{_DS}/derivatives/motion/{_SUBJECT}/ses-forrestgump/func/"
        f"{_SUBJECT}_ses-forrestgump_acq-dico_run-{run:02d}_bold_moco.txt"
    )


TARGET_HZ = 10  # common resample rate across channels
_AUDIO_DIM = 128
_VISION_SHAPE = (16, 16)
_TACTILE_DIM = 32
_GRAVITY_DIM = 3
_FORCE_DIM = 6
_PHYSIO_HZ = 500.0  # per recording-cardresp_physio.json
_FMRI_TR = 2.0  # ses-forrestgump TR (s); moco params are per-volume


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_bytes(url: str, timeout: int = 60) -> bytes:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "bouba-sens/0.4"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# --------------------------------------------------------------------------
# Real-channel builders from ds000113 annotations / physio / moco.
# --------------------------------------------------------------------------
def _parse_scenes(raw: bytes) -> list[tuple[float, str, str, str]]:
    """scenes.csv rows: timestamp_s, "SCENE", "DAY|NIGHT", "INT|EXT"."""
    out: list[tuple[float, str, str, str]] = []
    for line in raw.decode("utf-8", "replace").splitlines():
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            t = float(parts[0])
        except ValueError:
            continue
        out.append((t, parts[1], parts[2], parts[3]))
    return out


def _parse_narration(raw: bytes) -> list[tuple[float, float]]:
    """german_audio_description.csv rows: start_s, end_s, "text"."""
    out: list[tuple[float, float]] = []
    for line in raw.decode("utf-8", "replace").splitlines():
        parts = line.split(",", 2)
        if len(parts) < 2:
            continue
        try:
            out.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return out


def _build_labels_and_tactile(
    scenes: list[tuple[float, str, str, str]], n_frames: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (labels long (N,), tactile (N,32), scene_id_per_frame (N,))."""
    if not scenes:
        raise ValueError("no scenes parsed")
    scene_names = sorted({s[1] for s in scenes})
    daynight = sorted({s[2] for s in scenes})
    intext = sorted({s[3] for s in scenes})
    name_to_i = {s: i for i, s in enumerate(scene_names)}
    dn_to_i = {s: i for i, s in enumerate(daynight)}
    ie_to_i = {s: i for i, s in enumerate(intext)}

    # Step-fill scene attributes per TARGET_HZ frame (a scene holds until the
    # next annotated cut — real piecewise-constant scene structure).
    scene_id = np.zeros(n_frames, dtype=np.int64)
    dn_id = np.zeros(n_frames, dtype=np.int64)
    ie_id = np.zeros(n_frames, dtype=np.int64)
    ordered = sorted(scenes, key=lambda s: s[0])
    for k, (t, name, dn, ie) in enumerate(ordered):
        f0 = round(t * TARGET_HZ)
        f1 = round(ordered[k + 1][0] * TARGET_HZ) if k + 1 < len(ordered) else n_frames
        f0 = max(0, min(f0, n_frames))
        f1 = max(0, min(f1, n_frames))
        if f1 <= f0:
            continue
        scene_id[f0:f1] = name_to_i[name]
        dn_id[f0:f1] = dn_to_i[dn]
        ie_id[f0:f1] = ie_to_i[ie]

    # Tactile = one-hot(scene_name, day/night, int/ext) -> 32-D random proj.
    raw_dim = len(scene_names) + len(daynight) + len(intext)
    onehot = np.zeros((n_frames, raw_dim), dtype=np.float32)
    onehot[np.arange(n_frames), scene_id] = 1.0
    onehot[np.arange(n_frames), len(scene_names) + dn_id] = 1.0
    onehot[np.arange(n_frames), len(scene_names) + len(daynight) + ie_id] = 1.0
    rng = np.random.default_rng(0)
    proj = rng.standard_normal((raw_dim, _TACTILE_DIM)).astype(np.float32)
    tactile = torch.from_numpy(onehot @ proj)

    # Labels = scene index quantised to 4 buckets.
    n_classes = 4
    quant = (scene_id * n_classes // max(1, len(scene_names))).clip(0, n_classes - 1)
    labels = torch.from_numpy(quant.astype(np.int64))
    return labels, tactile, torch.from_numpy(scene_id)


def _build_audio_proxy(narration: list[tuple[float, float]], n_frames: int) -> torch.Tensor:
    """128-D annotation-derived narration proxy (NOT the soundtrack).

    Per-frame features from the REAL timed german_audio_description:
    narration-on flag, time-into-current-segment, time-to-next-segment,
    local narration density (windowed) -> 4-D base, projected to 128-D.
    """
    on = np.zeros(n_frames, dtype=np.float32)
    into = np.zeros(n_frames, dtype=np.float32)
    for start, end in narration:
        f0 = max(0, round(start * TARGET_HZ))
        f1 = min(n_frames, round(end * TARGET_HZ))
        for f in range(f0, f1):
            on[f] = 1.0
            into[f] = (f - f0) / max(1, f1 - f0)
    # Local density: fraction of narration-on in a +-5 s window.
    win = 5 * TARGET_HZ
    kernel = np.ones(2 * win + 1, dtype=np.float32) / (2 * win + 1)
    density = np.convolve(on, kernel, mode="same").astype(np.float32)
    # Time-to-next narration onset (clipped).
    to_next = np.full(n_frames, win, dtype=np.float32)
    next_onset = n_frames
    for f in range(n_frames - 1, -1, -1):
        if on[f] > 0:
            next_onset = f
        to_next[f] = min(win, next_onset - f)
    to_next /= win
    base = np.stack([on, into, density, to_next], axis=1)  # (N,4)
    rng = np.random.default_rng(1)
    proj = rng.standard_normal((4, _AUDIO_DIM)).astype(np.float32)
    feat = base @ proj + 0.05 * rng.standard_normal((n_frames, _AUDIO_DIM)).astype(np.float32)
    return torch.from_numpy(feat.astype(np.float32))


def _build_vision_proxy(scene_id: np.ndarray, n_frames: int) -> torch.Tensor:
    """(16,16) annotation-derived scene-structure proxy (NOT the movie).

    Real scene cuts drive a piecewise-constant latent + time-since-cut ramp,
    projected into 256 dims reshaped to (16,16).
    """
    sid = scene_id.astype(np.float32)
    since_cut = np.zeros(n_frames, dtype=np.float32)
    c = 0
    for f in range(n_frames):
        if f > 0 and scene_id[f] != scene_id[f - 1]:
            c = 0
        else:
            c += 1
        since_cut[f] = c
    since_cut = since_cut / max(1.0, since_cut.max())
    base = np.stack([sid / max(1.0, sid.max()), since_cut], axis=1)  # (N,2)
    rng = np.random.default_rng(2)
    dim = _VISION_SHAPE[0] * _VISION_SHAPE[1]
    proj = rng.standard_normal((2, dim)).astype(np.float32)
    feat = base @ proj + 0.05 * rng.standard_normal((n_frames, dim)).astype(np.float32)
    return torch.from_numpy(feat.astype(np.float32)).reshape(n_frames, *_VISION_SHAPE)


def _build_force(physio_runs: list[np.ndarray], n_frames: int) -> torch.Tensor:
    """6-D autonomic signal from REAL cardiac + respiratory physio.

    physio columns: [trigger, cardiac, respiratory] @ 500 Hz. Concatenate
    runs, decimate to TARGET_HZ, stack [card, d, d², resp, d, d²].
    """
    if not physio_runs:
        return torch.zeros(n_frames, _FORCE_DIM)
    full = np.concatenate(physio_runs, axis=0)  # (T,3+)
    step = int(_PHYSIO_HZ // TARGET_HZ)  # 50
    cardiac = full[::step, 1].astype(np.float32)
    resp = full[::step, 2].astype(np.float32)
    if cardiac.shape[0] >= n_frames:
        cardiac = cardiac[:n_frames]
        resp = resp[:n_frames]
    else:
        cardiac = np.pad(cardiac, (0, n_frames - cardiac.shape[0]))
        resp = np.pad(resp, (0, n_frames - resp.shape[0]))
    c1, c2 = np.gradient(cardiac), np.gradient(np.gradient(cardiac))
    r1, r2 = np.gradient(resp), np.gradient(np.gradient(resp))
    six = np.stack([cardiac, c1, c2, resp, r1, r2], axis=1).astype(np.float32)
    return torch.from_numpy(six)


def _build_gravity(moco_runs: list[np.ndarray], n_frames: int) -> torch.Tensor:
    """3 rotation regressors from REAL moco params, upsampled to TARGET_HZ.

    moco.txt cols: 3 translations + 3 rotations per fMRI volume (TR=2s).
    Keep rotations (cols 3:6); nearest-neighbour upsample to TARGET_HZ.
    """
    if not moco_runs:
        return torch.zeros(n_frames, _GRAVITY_DIM)
    full = np.concatenate(moco_runs, axis=0)  # (T_vol, 6)
    rot = full[:, 3:6].astype(np.float32)
    frames_per_vol = int(TARGET_HZ * _FMRI_TR)  # 20
    up = np.zeros((n_frames, _GRAVITY_DIM), dtype=np.float32)
    for i in range(n_frames):
        vol = min(i // frames_per_vol, rot.shape[0] - 1)
        up[i] = rot[vol]
    return torch.from_numpy(up)


def _try_openneuro_fetch(out_dir: Path, duration_seconds: int) -> tuple[int, dict[str, str]] | None:
    """Fetch real ds000113 channels from S3 and build the 6 tensors.

    Returns (n_frames, {key: url}) or None on any network failure.
    """
    n_frames = duration_seconds * TARGET_HZ
    urls: dict[str, str] = {}
    try:
        scenes_raw = _fetch_bytes(_S3_ROOT + _SCENES_KEY)
        urls["scenes.csv"] = _S3_ROOT + _SCENES_KEY
        narration_raw = _fetch_bytes(_S3_ROOT + _NARRATION_KEY)
        urls["german_audio_description.csv"] = _S3_ROOT + _NARRATION_KEY

        physio_runs: list[np.ndarray] = []
        moco_runs: list[np.ndarray] = []
        # Concatenate enough runs to cover the requested duration.
        for run in range(1, _N_RUNS + 1):
            covered = sum(p.shape[0] for p in physio_runs) / _PHYSIO_HZ
            if covered >= duration_seconds:
                break
            p_raw = _fetch_bytes(_S3_ROOT + _physio_key(run))
            urls[f"physio_run-{run:02d}.tsv.gz"] = _S3_ROOT + _physio_key(run)
            with gzip.open(io.BytesIO(p_raw), "rt") as fh:
                physio_runs.append(np.loadtxt(fh))
            m_raw = _fetch_bytes(_S3_ROOT + _moco_key(run))
            urls[f"moco_run-{run:02d}.txt"] = _S3_ROOT + _moco_key(run)
            moco_runs.append(np.loadtxt(io.BytesIO(m_raw)))
    except Exception as exc:
        print(
            f"[fetch] OpenNeuro S3 fetch failed ({exc}); use --source surrogate.", file=sys.stderr
        )
        return None

    scenes = _parse_scenes(scenes_raw)
    narration = _parse_narration(narration_raw)
    labels, tactile, scene_id = _build_labels_and_tactile(scenes, n_frames)
    audio = _build_audio_proxy(narration, n_frames)
    vision = _build_vision_proxy(scene_id.numpy(), n_frames)
    force = _build_force(physio_runs, n_frames)
    gravity = _build_gravity(moco_runs, n_frames)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(audio, out_dir / "audio.pt")
    torch.save(vision, out_dir / "vision.pt")
    torch.save(tactile, out_dir / "tactile.pt")
    torch.save(gravity, out_dir / "gravity.pt")
    torch.save(force, out_dir / "force.pt")
    torch.save(labels, out_dir / "labels.pt")
    return n_frames, urls


# --------------------------------------------------------------------------
# Offline surrogate (NOT scientific data) — retained fallback.
# --------------------------------------------------------------------------
def _build_offline_surrogate(out_dir: Path, n_samples: int, seed: int) -> tuple[int, str]:
    """AR(1) scene-latent mock + zeroed-then-noised proprioceptive channels.

    Writes all SIX tensors so StudyforrestRealWorld can load it too. NOT
    scientific data — labelled as such in the manifest.
    """
    from bouba_sens.world.studyforrest import _build_mock_cache

    audio, vision, labels = _build_mock_cache(n_samples, seed=seed)
    rng = np.random.default_rng(seed)
    tactile = torch.from_numpy(rng.standard_normal((n_samples, _TACTILE_DIM)).astype(np.float32))
    gravity = torch.from_numpy(
        0.1 * rng.standard_normal((n_samples, _GRAVITY_DIM)).astype(np.float32)
    )
    force = torch.from_numpy(rng.standard_normal((n_samples, _FORCE_DIM)).astype(np.float32))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(audio, out_dir / "audio.pt")
    torch.save(vision.reshape(n_samples, *_VISION_SHAPE), out_dir / "vision.pt")
    torch.save(tactile, out_dir / "tactile.pt")
    torch.save(gravity, out_dir / "gravity.pt")
    torch.save(force, out_dir / "force.pt")
    torch.save(labels, out_dir / "labels.pt")
    return n_samples, "offline surrogate (AR(1) scene-latent mock) — NOT scientific data"


def write_manifest(out_dir: Path, n: int, source: str, provenance: list[str]) -> None:
    files = ["audio.pt", "vision.pt", "tactile.pt", "gravity.pt", "force.pt", "labels.pt"]
    lines = [
        "# Studyforrest sample manifest",
        "",
        f"- Source: {source}",
        f"- Frames: {n} at {TARGET_HZ} Hz",
        "",
        "## Per-channel provenance",
        "",
        *[f"- {p}" for p in provenance],
        "",
        "## SHA256",
        "",
    ]
    for fname in files:
        fpath = out_dir / fname
        if fpath.exists():
            lines.append(f"- `{fname}`: `{_sha256(fpath)}`")
    lines.append("")
    (out_dir / "MANIFEST.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/studyforrest_sample"))
    parser.add_argument(
        "--source",
        choices=["openneuro", "surrogate"],
        default="openneuro",
        help="openneuro = real ds000113 S3; surrogate = offline mock (NOT scientific data)",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=900,
        help="Seconds of timeline to build (default 900 s = 9000 frames @ 10 Hz)",
    )
    parser.add_argument("--n-samples", type=int, default=4096, help="Surrogate frame count")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Re-run even if MANIFEST.md exists")
    args = parser.parse_args()

    manifest = args.out / "MANIFEST.md"
    if manifest.exists() and not args.force:
        print(f"[fetch] manifest already at {manifest}; use --force to re-run.")
        return

    if args.source == "openneuro":
        real = _try_openneuro_fetch(args.out, args.duration_seconds)
        if real is None:
            print(
                "[fetch] FATAL: --source openneuro failed and surrogate not requested.",
                file=sys.stderr,
            )
            print(
                "[fetch]        re-run with --source surrogate for an offline (NON-scientific) cache.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        n, urls = real
        provenance = [
            "`force.pt`  : REAL ds000113 cardiac(pulse-ox)+respiration physio, 500 Hz "
            "[cardiac,d,d2,resp,d,d2]",
            "`gravity.pt`: REAL ds000113 derivatives/motion *_bold_moco.txt rotation regressors",
            "`tactile.pt`: REAL ds000113 scenes.csv (scene/DAY-NIGHT/INT-EXT) one-hot -> 32-D proj",
            "`labels.pt` : REAL ds000113 scenes.csv scene index, quantised to 4",
            "`audio.pt`  : DERIVED PROXY (NOT soundtrack; copyrighted/absent from ds000113) "
            "from REAL german_audio_description.csv narration timing -> 128-D",
            "`vision.pt` : DERIVED PROXY (NOT movie; copyrighted/absent from ds000113) "
            "from REAL scenes.csv cut structure -> (16,16)",
            "",
            "### Real S3 object URLs",
            *[f"- `{k}`: {v}" for k, v in urls.items()],
        ]
        source = f"OpenNeuro {_DS} (Studyforrest, Hanke et al. 2014) via S3 REST"
    else:
        n, source = _build_offline_surrogate(args.out, args.n_samples, args.seed)
        provenance = [
            "ALL channels are offline surrogate — NOT scientific data.",
            "audio/vision: AR(1) scene-latent mock; tactile/gravity/force: seeded noise.",
        ]

    write_manifest(args.out, n, source, provenance)
    print(
        f"[fetch] wrote {n} frames to {args.out}/ (source: {source}); "
        f"export BOUBA_SENS_STUDYFORREST_DATA={args.out}"
    )


if __name__ == "__main__":
    main()
