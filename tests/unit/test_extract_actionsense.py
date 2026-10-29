"""Unit tests for scripts/extract_actionsense.py on a synthetic recording."""

from __future__ import annotations

import importlib.util
import shutil
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
    ts = T0 + ((np.arange(n_chunks) + 1) * CHUNK - 1) / SR
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

    def energies(shift_s: float) -> np.ndarray:
        # Exact convention: ts[k] is the time of sample (k+1)*CHUNK-1, so sample j
        # of the WAV lies at T0 + j / SR (upstream MicrophoneStreamer.py, chunk 2048).
        ts = T0 + shift_s + ((np.arange(n_chunks) + 1) * CHUNK - 1) / SR
        h5_path = tmp_path / f"mic_{shift_s}.hdf5"
        with h5py.File(h5_path, "w") as h5:
            _stream(h5, "sink", "chunk_timestamp", ts, ts[:, None])
        mel, covered = extract.audio_frames(wav_path, h5_path, starts)
        assert covered.all()
        return np.asarray(mel.mean(axis=1))

    def peak(shift_s: float) -> int:
        return int(np.argmax(energies(shift_s))) + first

    e0 = energies(0.0)
    assert int(np.argmax(e0)) + first == target
    # Every other frame is exactly silent (log(1e-6)); catches in-chunk offset errors.
    quiet = np.delete(e0, target - first)
    np.testing.assert_allclose(quiet, np.log(1e-6), atol=1e-9)
    assert peak(CHUNK / SR) != target  # timestamps one chunk late
    assert peak(-CHUNK / SR) != target  # timestamps one chunk early


def test_audio_frames_rejects_off_by_one_timestamp_count(tmp_path: Path) -> None:
    n_chunks = 16
    wav = np.zeros(n_chunks * CHUNK, dtype=np.float32)
    wav_path = tmp_path / "sink.wav"
    sf.write(wav_path, wav, SR, subtype="FLOAT")
    starts = T0 + 0.1 * np.arange(3, 13)
    for count in (n_chunks - 1, n_chunks + 1):
        ts = T0 + ((np.arange(count) + 1) * CHUNK - 1) / SR
        h5_path = tmp_path / f"mic_{count}.hdf5"
        with h5py.File(h5_path, "w") as h5:
            _stream(h5, "sink", "chunk_timestamp", ts, ts[:, None])
        info: dict[str, object] = {}
        _mel, covered = extract.audio_frames(wav_path, h5_path, starts, info)
        assert not covered.any()
        assert "samples vs" in str(info["error"])


def _bursty_stamps(n_chunks: int, sr: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """True last-sample times and burst-stamped times centred on them.

    Chunks are stamped in groups of 1-3 sharing the mean true time of the
    group, as in the S00 microphone files (steps ~0, ~43, ~85 ms at 48 kHz).
    """
    true = T0 + ((np.arange(n_chunks) + 1) * CHUNK - 1) / sr
    rng = np.random.default_rng(seed)
    stamps = np.empty(n_chunks)
    k = 0
    while k < n_chunks:
        g = min(int(rng.integers(1, 4)), n_chunks - k)
        stamps[k : k + g] = true[k : k + g].mean()
        k += g
    return true, stamps


def test_audio_bursty_timestamps_kept(tmp_path: Path) -> None:
    """Burst-stamped chunks: no frame dropped, the burst lands in its frame."""
    sr = 48000
    n_chunks = 200  # 8.5 s: long enough for the fitted rate to settle
    win = sr // 10
    target = 42
    wav = np.zeros(n_chunks * CHUNK, dtype=np.float32)
    rng = np.random.default_rng(2)
    # Burst in the central 50 ms of the target window: a placement error of
    # 25 ms (the old per-chunk mapping was off by up to ~85 ms) would leak.
    lo, hi = target * win + win // 4, (target + 1) * win - win // 4
    wav[lo:hi] = rng.uniform(-0.8, 0.8, hi - lo)
    wav_path = tmp_path / "sink.wav"
    sf.write(wav_path, wav, sr, subtype="FLOAT")
    _true, ts = _bursty_stamps(n_chunks, sr, seed=3)
    steps = np.diff(ts)
    assert (steps == 0).any() and (steps > 1.5 * CHUNK / sr).any()  # really bursty
    h5_path = tmp_path / "mic.hdf5"
    with h5py.File(h5_path, "w") as h5:
        _stream(h5, "sink", "chunk_timestamp", ts, ts[:, None])
    first = 1
    starts = T0 + 0.1 * np.arange(first, 84)
    info: dict[str, object] = {}
    mel, covered = extract.audio_frames(wav_path, h5_path, starts, info)
    assert "error" not in info
    assert covered.all()  # nothing dropped despite 0 / 85 ms steps
    e = np.asarray(mel.mean(axis=1))
    assert int(np.argmax(e)) + first == target
    np.testing.assert_allclose(np.delete(e, target - first), np.log(1e-6), atol=1e-9)
    fit = info["fit"]
    assert isinstance(fit, dict)
    assert fit["rate_dev"] <= extract.MIC_RATE_TOL
    assert CHUNK / sr / 2 < fit["res_max"] < 2 * CHUNK / sr  # jitter is seen, not fatal


def test_audio_bad_rate_rejected(tmp_path: Path) -> None:
    """Timestamps implying 44.1 kHz against a 48 kHz WAV: file rejected."""
    sr = 48000
    n_chunks = 64
    wav_path = tmp_path / "sink.wav"
    sf.write(wav_path, np.zeros(n_chunks * CHUNK, dtype=np.float32), sr, subtype="FLOAT")
    ts = T0 + ((np.arange(n_chunks) + 1) * CHUNK - 1) / 44100
    h5_path = tmp_path / "mic.hdf5"
    with h5py.File(h5_path, "w") as h5:
        _stream(h5, "sink", "chunk_timestamp", ts, ts[:, None])
    info: dict[str, object] = {}
    _mel, covered = extract.audio_frames(wav_path, h5_path, T0 + 0.1 * np.arange(1, 20), info)
    assert not covered.any() and info.get("fit_rejected") is True
    assert "implied rate 44100.0 Hz" in str(info["error"])

    # Same rate error through a session: the counter and the manifest line.
    sub = _write_subject(tmp_path / "raw")
    mic = next(sub.glob("*microphone_S00.hdf5"))
    with h5py.File(mic, "a") as h5:
        ds = h5["sink"]["chunk_timestamp"]["time_s"]
        t = np.asarray(ds)
        ds[...] = T0 + (t - T0) * (48000 / 44100)
    counts: dict[str, int] = {}
    report: list[dict[str, object]] = []
    wear = next(sub.glob("*wearables*"))
    sess = extract._extract_session(wear, sub, "S00", False, counts, {}, report)
    assert sess is not None and len(sess["labels"]) == 0
    assert counts["audio_file_bad_clock_fit"] == 1
    assert "audio_file_inconsistent" not in counts
    assert "REJECTED" in "\n".join(extract._span_lines(report))


def test_audio_clock_jump_rejected(tmp_path: Path) -> None:
    """A 2 s clock jump mid-file breaks the residual bound, not the rate one."""
    sr = 1000  # low rate keeps a long (8192 s) file small
    n_chunks = 4000
    wav_path = tmp_path / "sink.wav"
    sf.write(wav_path, np.zeros(n_chunks * CHUNK, dtype=np.float32), sr, subtype="PCM_16")
    ts = T0 + ((np.arange(n_chunks) + 1) * CHUNK - 1) / sr
    ts[n_chunks // 2 :] += 2.0
    h5_path = tmp_path / "mic.hdf5"
    with h5py.File(h5_path, "w") as h5:
        _stream(h5, "sink", "chunk_timestamp", ts, ts[:, None])
    info: dict[str, object] = {}
    _mel, covered = extract.audio_frames(wav_path, h5_path, T0 + 0.1 * np.arange(1, 20), info)
    assert not covered.any() and info.get("fit_rejected") is True
    assert "residual" in str(info["error"])
    fit = info["fit"]
    assert isinstance(fit, dict) and fit["rate_dev"] <= extract.MIC_RATE_TOL


def test_session_counts_inconsistent_audio_file(tmp_path: Path) -> None:
    sub = _write_subject(tmp_path)
    mic = next(sub.glob("*microphone_S00.hdf5"))
    with h5py.File(mic, "a") as h5:
        grp = h5["sink"]["chunk_timestamp"]
        t = np.asarray(grp["time_s"])[:-1]
        del grp["time_s"], grp["data"]
        grp.create_dataset("time_s", data=t)
        grp.create_dataset("data", data=t)
    counts: dict[str, int] = {}
    wear = next(sub.glob("*wearables*"))
    sess = extract._extract_session(wear, sub, "S00", False, counts, {}, [])
    assert sess is not None and len(sess["labels"]) == 0
    assert counts["audio_file_inconsistent"] == 1
    assert counts["drop:audio:outside_span"] == 10


def test_nonmonotonic_wearable_time_skips_session(tmp_path: Path) -> None:
    sub = _write_subject(tmp_path)
    wear = next(sub.glob("*wearables*"))
    with h5py.File(wear, "a") as h5:
        ds = h5["myo-right"]["acceleration_g"]["time_s"]
        t = np.asarray(ds)
        t[50, 0] = t[10, 0]
        ds[...] = t
    counts: dict[str, int] = {}
    report: list[dict[str, object]] = []
    assert extract._extract_session(wear, sub, "S00", False, counts, {}, report) is None
    assert counts["session_skipped:nonmonotonic"] == 1
    assert "backwards steps" in str(report[0]["skipped"])


def test_rows_side_file_and_manifest_spans(tmp_path: Path) -> None:
    import torch

    _write_subject(tmp_path)
    out = tmp_path / "o"
    extract.main(["--raw", str(tmp_path), "--out", str(out), "--subjects", "S00", "--no-vision"])
    rows = torch.load(out / "rows.pt")
    assert set(rows) == {"time_s", "segment_id", "raw_label_idx"}
    assert rows["time_s"].dtype == torch.float64
    assert rows["segment_id"].dtype == torch.long and rows["raw_label_idx"].dtype == torch.long
    assert all(len(v) == 10 for v in rows.values())
    np.testing.assert_allclose(np.diff(rows["time_s"].numpy()), 0.1)
    manifest = (out / "MANIFEST.md").read_text()
    assert "0. Slice a cucumber" in manifest
    assert "backwards steps=0" in manifest
    assert "microphone/" in manifest and "labelled segments" in manifest
    assert "microphone clock fit" in manifest and ": accepted" in manifest


def _file_hashes(out: Path) -> dict[str, str]:
    import hashlib

    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.pt"))}


def test_extraction_is_deterministic(tmp_path: Path) -> None:
    _write_subject(tmp_path)
    hashes = []
    for k in range(2):
        out = tmp_path / f"o{k}"
        extract.main(
            ["--raw", str(tmp_path), "--out", str(out), "--subjects", "S00", "--no-vision"]
        )
        hashes.append(_file_hashes(out))
    assert hashes[0] == hashes[1]
    assert {"audio.pt", "tactile.pt", "labels.pt", "rows.pt"} <= set(hashes[0])


def test_parse_ffmpeg_version_and_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    assert extract._parse_ffmpeg_version("ffmpeg version 6.1.1 Copyright") == (6, 1)
    assert extract._parse_ffmpeg_version("ffmpeg version n7.0 Copyright") == (7, 0)
    assert extract._parse_ffmpeg_version("ffmpeg version N-1234-gabc") is None

    class Res:
        stdout = "ffmpeg version 4.4.2 Copyright (c) 2000-2021\n"

    monkeypatch.setattr(extract.subprocess, "run", lambda *a, **k: Res())
    with pytest.raises(SystemExit, match="too old"):
        extract.check_ffmpeg()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_decode_grids_reports_ffmpeg_failure(tmp_path: Path) -> None:
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(extract.FfmpegError, match=r"broken\.mp4"):
        extract.decode_grids(bad, np.array([0]))
