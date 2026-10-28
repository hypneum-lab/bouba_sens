"""Unit tests for scripts/extract_actionsense.py on a synthetic recording."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np
import pytest
import soundfile as sf

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
T0 = 1000.0
SR = 16000
CHUNK = 2048


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


extract = _load("extract_actionsense")


def _stream(grp: h5py.Group, dev: str, name: str, t: np.ndarray, data: np.ndarray) -> None:
    g = grp.require_group(dev).require_group(name)
    g.create_dataset("data", data=data)
    g.create_dataset("time_s", data=t[:, None])


def _write_subject(raw: Path, subject: str = "S00") -> Path:
    sub = raw / subject
    sub.mkdir(parents=True)
    rng = np.random.default_rng(0)
    path = sub / f"2022-06-07_17-18-46_streamLog_actionNet-wearables_{subject}.hdf5"
    with h5py.File(path, "w") as h5:
        for side in ("left", "right"):
            t = T0 + np.arange(60) / 30.0
            _stream(h5, f"tactile-glove-{side}", "tactile_data", t, rng.random((60, 32, 32)))
            te = T0 + np.arange(400) / 200.0
            emg = rng.integers(-100, 100, (400, 8)).astype(np.int32)
            _stream(h5, f"myo-{side}", "emg", te, emg)
        ta = T0 + np.arange(100) / 50.0
        _stream(h5, "myo-right", "acceleration_g", ta, np.tile([0.0, 0.0, 1.0], (100, 1)))
        rows = np.array(
            [
                [b"Slice a cucumber", b"Start", b"Good", b""],
                [b"Slice a cucumber", b"Stop", b"Good", b""],
            ],
            dtype="S40",
        )
        _stream(h5, "experiment-activities", "activities", np.array([T0 + 0.2, T0 + 1.2]), rows)
    # Microphone: 2 s sine, chunk timestamps = time of the last sample of each chunk.
    n = 16 * CHUNK
    wav = (0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / SR)).astype(np.float32)
    stem = f"2022-06-07_17-13-41_streamLog_actionNet-microphone_{subject}"
    sf.write(sub / f"{stem}_sink_chunked_data.wav", wav, SR)
    n_chunks = 16
    ts = T0 + (np.arange(n_chunks) + 1) * CHUNK / SR
    with h5py.File(sub / f"{stem}.hdf5", "w") as h5:
        _stream(h5, "sink", "chunk_timestamp", ts, ts[:, None])
    return sub


def test_extract_subject_synthetic(tmp_path: Path) -> None:
    _write_subject(tmp_path)
    out = extract.extract_subject(tmp_path, "S00", vision=False)
    assert out["tactile"].shape == (10, 32)
    assert out["force"].shape == (10, 6)
    assert out["gravity"].shape == (10, 3)
    assert out["audio"].shape == (10, 128)
    assert out["labels"].shape == (10,) and (out["labels"] == 0).all()
    assert all(np.isfinite(a).all() for a in out.values())
    np.testing.assert_allclose(out["gravity"], [[0.0, 0.0, 1.0]] * 10, atol=1e-6)


def test_extract_aborts_on_misalignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_subject(tmp_path)
    real = extract.extract_subject_with_counts

    def bad(*args: object, **kwargs: object) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        arrays, counts = real(*args, **kwargs)
        arrays["audio"] = arrays["audio"][:-1]
        return arrays, counts

    monkeypatch.setattr(extract, "extract_subject_with_counts", bad)
    with pytest.raises(SystemExit) as exc:
        extract.main(
            [
                "--raw",
                str(tmp_path),
                "--out",
                str(tmp_path / "o"),
                "--subjects",
                "S00",
                "--no-vision",
            ]
        )
    assert exc.value.code not in (0, None)


def test_main_writes_tensors_and_manifest(tmp_path: Path) -> None:
    _write_subject(tmp_path)
    out = tmp_path / "o"
    extract.main(["--raw", str(tmp_path), "--out", str(out), "--subjects", "S00", "--no-vision"])
    import torch

    assert torch.load(out / "tactile.pt").shape == (10, 32)
    assert torch.load(out / "subject.pt").tolist() == [0] * 10
    assert torch.load(out / "labels.pt").dtype == torch.long
    manifest = (out / "MANIFEST.md").read_text()
    assert "CC BY-NC-SA 4.0 — ActionSense, DelPreto et al., NeurIPS 2022" in manifest
    assert "Slice a cucumber" in manifest


def test_nearest_frame_tolerance() -> None:
    ft = np.array([0.0, 0.033, 0.066, 0.5])
    idx, ok = extract.nearest_frame(ft, np.array([0.04, 0.3, 0.9]))
    assert idx.tolist() == [1, 3, 3]
    assert ok.tolist() == [True, False, False]


def test_fetch_resolve_urls_filters_kinds_and_subjects() -> None:
    fetch = _load("fetch_actionsense")
    base = "https://data.csail.mit.edu/ActionNet"
    w = f"{base}/wearable_data/2022-06-07_experiment_S00/2022-06-07_17-18-17_actionNet-wearables_S00"
    m = f"{base}/external_data/2022-06-07_experiment_S00/2022-06-07_17-13-37_actionNet-microphone_S00"
    files = [
        f"{w}/2022-06-07_17-18-46_streamLog_actionNet-wearables_S00.hdf5",
        f"{w}/2022-06-07_17-18-46_S00_eye-tracking-video-world_frame.mp4",
        f"{w}/2022-06-07_17-18-46_S00_eye-tracking-video-worldGaze_frame.avi",
        f"{m}/2022-06-07_17-13-41_streamLog_actionNet-microphone_S00_sink_chunked_data.wav",
        f"{m}/2022-06-07_17-13-41_streamLog_actionNet-microphone_S00_overhead_chunked_data.wav",
        f"{m}/2022-06-07_17-13-41_streamLog_actionNet-microphone_S00.hdf5",
        f"{base}/external_data/2022-06-07_experiment_S00/PG1_2022-06-07-17-13-26.avi",
    ]
    html = "".join(f'<a href="{u}">x</a><a href="{u}">y</a>' for u in files)
    got = fetch.resolve_urls(html, ["S00", "S02"])
    assert len(got["S00"]) == 4 and got["S02"] == []
    assert not any("worldGaze" in u or "overhead" in u or "PG1" in u for u in got["S00"])


def test_audio_frames_chunk_timestamp_alignment(tmp_path: Path) -> None:
    """A burst confined to one 0.1 s window must land in exactly that frame."""
    n_chunks = 16
    n = n_chunks * CHUNK
    win = SR // 10
    target = 5  # frame start = T0 + 0.5 s
    wav = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(1)
    wav[target * win : (target + 1) * win] = rng.uniform(-0.8, 0.8, win)
    wav_path = tmp_path / "sink.wav"
    sf.write(wav_path, wav, SR, subtype="FLOAT")
    first = 3
    starts = T0 + 0.1 * np.arange(first, 13)  # frames 3..12

    def peak(shift_s: float) -> int:
        # Real convention: timestamp of the LAST sample of each chunk.
        ts = T0 + shift_s + (np.arange(n_chunks) + 1) * CHUNK / SR
        h5_path = tmp_path / f"mic_{shift_s}.hdf5"
        with h5py.File(h5_path, "w") as h5:
            _stream(h5, "sink", "chunk_timestamp", ts, ts[:, None])
        mel, covered, _ = extract.audio_frames(wav_path, h5_path, starts)
        assert covered.all()
        return int(np.argmax(mel.mean(axis=1))) + first

    assert peak(0.0) == target
    assert peak(CHUNK / SR) != target  # timestamps one chunk late
    assert peak(-CHUNK / SR) != target  # timestamps one chunk early
