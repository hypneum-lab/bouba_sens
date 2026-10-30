"""Extract the ActionSense real-data cache (7 tensors) from the raw files.

ActionSense (DelPreto et al., NeurIPS 2022) recordings, reduced at 10 Hz to
the bouba_sens world contract. Spec: docs/superpowers/specs/
2026-10-28-actionsense-world-design.md, section 6.

Raw layout (written by scripts/fetch_actionsense.py): ``<raw>/<subject>/<file>``
with, per subject, one or more of:

  *_streamLog_actionNet-wearables_<S>.hdf5       wearable streams + activities
  *_<S>_eye-tracking-video-world_frame.mp4       egocentric world video
  *_streamLog_actionNet-microphone_<S>.hdf5      microphone chunk timestamps
  *_streamLog_actionNet-microphone_<S>_sink_chunked_data.wav

Real stream names (ActionNet code, recording_data/sensor_streamers):
  tactile-glove-{left,right}/tactile_data   (T,32,32) float32
  myo-{left,right}/emg                      (T,8) int32, 200 Hz
  myo-right/acceleration_g                  (T,3) float64, 50 Hz
  experiment-activities/activities          (K,4) bytes: label, Start/Stop,
                                            validity, notes
  eye-tracking-video-world/frame_timestamp  one time_s per video frame
  sink/chunk_timestamp (microphone HDF5)    time_s of the LAST sample of each
                                            2048-sample chunk of the WAV
Every ``time_s`` is seconds since epoch on the same system clock.

Outputs ``<out>/{audio,vision,tactile,gravity,force,labels,subject}.pt``
(float32, labels and subject long), the side file ``rows.pt`` (time_s, segment_id,
raw_label_idx; not part of the world contract) and ``MANIFEST.md``. No raw frame and no
waveform leaves this script. Data licence: CC BY-NC-SA 4.0.

Usage:
    uv run python scripts/extract_actionsense.py --raw data/actionsense_raw \
        --out data/actionsense --subjects S00 S02 S03 S04 S05
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import soundfile as sf
import torch
from PIL import Image

from bouba_sens.world.actionsense_features import (
    FAMILY_RULES,
    activity_segments,
    check_aligned,
    emg_sectors,
    frame_grid,
    frame_to_grid,
    label_family,
    lowpass,
    mel_filterbank,
    monotonic_violations,
    pool_tactile,
    window_mean,
    window_rms,
    zscore,
)

LICENCE = "CC BY-NC-SA 4.0 — ActionSense, DelPreto et al., NeurIPS 2022"
HZ = 10.0
WIDTH = 0.1  # frame window [t, t + 0.1) seconds
MAX_GAP = 0.2  # interpolate gaps up to 200 ms, drop otherwise
N_MELS = 128
# Microphone chunk size in samples: fixed upstream, see
# ActionNet recording_data/sensor_streamers/MicrophoneStreamer.py
# (chunk_size = 2048; WAV and chunk_timestamp are appended together).
MIC_CHUNK = 2048
# Microphone clock fit (pilot S00, 2026-10-29: chunk stamps come in bursts,
# steps ~0/43/85 ms, max 141 ms; 3660.3 s of stamps vs 3658.1 s of samples
# over the 18-11-04 file). Samples are placed on a piecewise linear fit of
# chunk timestamp against sample index. Breakpoints are found on the
# residuals of a global fit: the median residual of the MIC_STEP_WINDOW
# chunks (~1 s at 48 kHz) after chunk k minus that of the window before
# moves by more than MIC_STEP_DETECT_S (burst jitter averages out, a clock
# step or dropout does not). Each segment is fitted on its own and rejected
# when its rate deviates from the WAV rate by more than MIC_RATE_TOL
# (relative; 44.1 vs 48 kHz is 8e-2) or the 99th percentile of its
# |residual| exceeds MIC_SEGMENT_RESIDUAL_MAX_S (issue #52: a host stall,
# ~7 chunks stamped together ~0.3 s late, is a spike the step detector
# ignores; on the max residual it rejected a whole 1 h file). Chunks with
# |residual| > MIC_SEGMENT_RESIDUAL_MAX_S are left out of a second fit of
# their segment, which then sets the residuals and the gate; in an
# accepted segment, frames whose window touches such a chunk are dropped
# as a stall. Frames within
# MIC_BREAKPOINT_MARGIN_S of a breakpoint are dropped. A constant host
# latency between the sound and its timestamp is absorbed into the
# intercept and cannot be detected.
MIC_STEP_WINDOW = 25
MIC_STEP_DETECT_S = 0.05
MIC_RATE_TOL = 2e-3
MIC_SEGMENT_RESIDUAL_MAX_S = 0.2
MIC_BREAKPOINT_MARGIN_S = 1.0
FFMPEG_MIN = (5, 1)  # `-fps_mode` needs ffmpeg >= 5.1
LOWPASS_CUTOFF_HZ = 0.5
LOWPASS_MIN_FRAMES = 10  # sosfiltfilt (order 2) needs more than 9 samples
VISION_DECODE_PX = 64
VISION_TOL_S = 0.1  # nearest video frame must lie within this of the tick
MODALITIES = ("audio", "vision", "tactile", "gravity", "force")
FAMILY_NAMES = {
    0: "cutting (peel, slice)",
    1: "spreading and jars",
    2: "cleaning (wipe, clean)",
    3: "pouring and tableware (pour, set table, dishwasher)",
}


# --------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------
def list_inputs(raw_dir: Path, subject: str) -> list[Path]:
    """All raw files of a subject that the extraction can read."""
    sub = raw_dir / subject
    pats = (
        f"*_streamLog_actionNet-wearables_{subject}.hdf5",
        f"*_{subject}_eye-tracking-video-world_frame.mp4",
        f"*_streamLog_actionNet-microphone_{subject}.hdf5",
        f"*_streamLog_actionNet-microphone_{subject}_sink_chunked_data.wav",
    )
    return sorted(p for pat in pats for p in sub.glob(pat))


# --------------------------------------------------------------------------
# HDF5 readers
# --------------------------------------------------------------------------
def _times(h5: h5py.File, dev: str, stream: str) -> np.ndarray:
    return np.asarray(h5[dev][stream]["time_s"], dtype=np.float64).reshape(-1)


class NonMonotonicTimeError(Exception):
    """A stream's time_s steps backwards: searchsorted lookups would be wrong."""


def _checked_times(
    h5: h5py.File, dev: str, stream: str, spans: dict[str, tuple[float, float, int, int]]
) -> np.ndarray:
    """``_times`` plus a monotonic check; records (first, last, n, steps back)."""
    t = _times(h5, dev, stream)
    bad = monotonic_violations(t)
    if len(t):
        spans[f"{dev}/{stream}"] = (float(t[0]), float(t[-1]), len(t), bad)
    if bad:
        raise NonMonotonicTimeError(f"{dev}/{stream}: {bad} backwards steps in time_s")
    return t


def _decode(x: object) -> str:
    return x.decode("utf-8") if isinstance(x, bytes) else str(x)


def _read_activities(h5: h5py.File) -> tuple[list[tuple[str, str, str]], np.ndarray]:
    grp = h5["experiment-activities"]["activities"]
    rows = [(_decode(r[0]), _decode(r[1]), _decode(r[2])) for r in np.asarray(grp["data"]).tolist()]
    return rows, np.asarray(grp["time_s"], dtype=np.float64).reshape(-1)


def _missing_streams(h5: h5py.File) -> list[str]:
    need = [
        ("tactile-glove-left", "tactile_data"),
        ("tactile-glove-right", "tactile_data"),
        ("myo-left", "emg"),
        ("myo-right", "emg"),
        ("myo-right", "acceleration_g"),
        ("experiment-activities", "activities"),
    ]
    return [f"{d}/{s}" for d, s in need if d not in h5 or s not in h5[d]]


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------
def _log_mel(frames: np.ndarray, fb: np.ndarray) -> np.ndarray:
    win = np.hanning(frames.shape[1])
    power = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
    return np.log(power @ fb.T + 1e-6)


def _line_fit(
    ts: np.ndarray,
    chunk: int,
    sr: float,
    k0: int,
    k1: int,
    keep: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    """Least-squares ``ts[k] = a + b * ((k + 1) * chunk - 1)`` on chunks k0..k1-1.

    Chunk ``k`` is stamped at its last sample, so sample ``j`` lies at
    ``a + b * j``. With ``keep`` (bool per chunk of k0..k1-1), only the kept
    chunks enter the regression; residuals and their statistics still cover
    every chunk. Returns the fit (``a``, ``b``, implied ``rate = 1 / b``,
    ``rate_dev = |rate - sr| / sr``, |residual| ``res_median``, ``res_p99``,
    ``res_max``) and the signed residuals.
    """
    idx = (np.arange(k0, k1, dtype=np.float64) + 1.0) * chunk - 1.0
    t_rel = ts[k0:k1] - ts[k0]  # keep the epoch offset out of the regression
    x, y = (idx, t_rel) if keep is None else (idx[keep], t_rel[keep])
    im, tm = x.mean(), y.mean()
    den = ((x - im) ** 2).sum()
    b = float(((x - im) * (y - tm)).sum() / den) if den > 0 else 0.0
    a_rel = float(tm - b * im)
    res = t_rel - (a_rel + b * idx)
    absr = np.abs(res)
    ok = b > 0
    return {
        "a": float(ts[k0]) + a_rel,
        "b": b,
        "rate": 1.0 / b if ok else float("nan"),
        "rate_dev": abs(1.0 / b - sr) / sr if ok else float("inf"),
        "res_median": float(np.median(absr)),
        "res_p99": float(np.percentile(absr, 99)),
        "res_max": float(absr.max()),
    }, res


def mic_breakpoints(
    res: np.ndarray, w: int = MIC_STEP_WINDOW, thr: float = MIC_STEP_DETECT_S
) -> list[tuple[int, float]]:
    """Clock steps in a residual series: ``(first chunk after the step, step s)``.

    ``d[k] = median(res[k:k+w]) - median(res[k-w:k])``; each run of
    ``|d| > thr`` gives one step, located at the split of the
    ``[k* - w, k* + w)`` neighbourhood of the run's peak ``k*`` that assigns
    every chunk to the nearer of the two far-side median levels.
    """
    n = len(res)
    if n < 2 * w + 1:
        return []
    med = np.median(np.lib.stride_tricks.sliding_window_view(res, w), axis=1)
    ks = np.arange(w, n - w + 1)
    d = med[ks] - med[ks - w]
    hot = np.abs(d) > thr
    out: list[tuple[int, float]] = []
    j = 0
    while j < len(ks):
        if not hot[j]:
            j += 1
            continue
        e = j
        while e + 1 < len(ks) and hot[e + 1]:
            e += 1
        pk = int(ks[j + int(np.argmax(np.abs(d[j : e + 1])))])
        lo, hi = max(pk - w, 1), min(pk + w, n - 1)
        m_left = float(np.median(res[max(lo - w, 0) : lo + 1]))
        m_right = float(np.median(res[hi - 1 : min(hi + w, n)]))
        cand = np.arange(lo, hi + 1)
        seg = res[lo:hi]
        dl = np.abs(seg - m_left)
        dr = np.abs(seg - m_right)
        # cost(s) = sum_{k<s} dl + sum_{k>=s} dr over the neighbourhood
        cost = np.concatenate([[0.0], np.cumsum(dl)]) + (
            dr.sum() - np.concatenate([[0.0], np.cumsum(dr)])
        )
        s = int(cand[int(np.argmin(cost))])
        if not out or s > out[-1][0]:
            out.append((s, m_right - m_left))
        j = e + 1
    return out


def mic_clock_segments(ts: np.ndarray, chunk: int, sr: float) -> list[dict[str, float]]:
    """Piecewise clock fit: one dict per segment between detected breakpoints.

    Each dict holds ``k0``, ``k1`` (chunk range), the ``_line_fit`` keys,
    ``step`` (clock step at its start from the two neighbouring fits, the
    detection estimate if a neighbour is too short, 0 for the first),
    ``t0``/``t1`` (time
    of its first/last sample on its own fit), ``n_stall`` (chunks with
    |residual| > ``MIC_SEGMENT_RESIDUAL_MAX_S``) and ``accepted`` (1.0 /
    0.0) after the ``MIC_RATE_TOL`` gate and the ``MIC_SEGMENT_RESIDUAL_MAX_S``
    gate on ``res_p99``.
    """
    return _mic_clock_fit(ts, chunk, sr)[0]


def _mic_clock_fit(
    ts: np.ndarray, chunk: int, sr: float
) -> tuple[list[dict[str, float]], np.ndarray]:
    """``mic_clock_segments`` plus a per-chunk stall mask.

    The mask flags the chunks of accepted segments whose |residual| on
    their own segment's fit exceeds ``MIC_SEGMENT_RESIDUAL_MAX_S``. When
    any chunk does, the segment is refitted once without those chunks (they
    would pull the line: an undetected jump of a few chunks at a file end
    moves it by tens of ms), and the residuals, the mask and the p99 gate
    are taken from that second fit; with fewer than 2 chunks left, the
    segment is rejected instead. Without such chunks the first fit is kept
    as is.
    """
    stall = np.zeros(len(ts), dtype=bool)
    _glob, res = _line_fit(ts, chunk, sr, 0, len(ts))
    bps = mic_breakpoints(res)
    edges = [0, *(k for k, _ in bps), len(ts)]
    steps = [0.0, *(st for _, st in bps)]
    segs: list[dict[str, float]] = []
    for i in range(len(edges) - 1):
        k0, k1 = edges[i], edges[i + 1]
        if k1 - k0 >= 2:
            fit, r = _line_fit(ts, chunk, sr, k0, k1)
            spike = np.abs(r) > MIC_SEGMENT_RESIDUAL_MAX_S
            # Fewer than 2 chunks left to refit on: rejected, the first fit
            # is kept for its statistics.
            fittable = (~spike).sum() >= 2
            if spike.any() and fittable:
                fit, r = _line_fit(ts, chunk, sr, k0, k1, keep=~spike)
                spike = np.abs(r) > MIC_SEGMENT_RESIDUAL_MAX_S
            good = (
                bool(fittable)
                and fit["rate_dev"] <= MIC_RATE_TOL
                and fit["res_p99"] <= MIC_SEGMENT_RESIDUAL_MAX_S
            )
            fit["n_stall"] = float(spike.sum())
            if good:
                stall[k0:k1] = spike
        else:
            fit = {"a": float(ts[k0]), "b": 1.0 / sr, "rate": float("nan")}
            fit |= {"rate_dev": float("inf"), "res_median": 0.0, "res_p99": 0.0}
            fit |= {"res_max": 0.0, "n_stall": 0.0}
            good = False
        fit |= {"k0": float(k0), "k1": float(k1), "step": steps[i]}
        fit["t0"] = fit["a"] + fit["b"] * k0 * chunk
        fit["t1"] = fit["a"] + fit["b"] * (k1 * chunk - 1)
        fit["accepted"] = 1.0 if good else 0.0
        if segs and k1 - k0 >= 2 and edges[i - 1] + 2 <= k0:
            # Step = this segment's time of its first sample minus the
            # previous segment's line extrapolated to that sample.
            prev = segs[-1]
            fit["step"] = fit["t0"] - (prev["a"] + prev["b"] * k0 * chunk)
        segs.append(fit)
    return segs, stall


def audio_frames(
    wav_path: Path,
    mic_h5_path: Path,
    starts: np.ndarray,
    info: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Log-mel (N, 128) for each frame start covered by this microphone file.

    Returns ``(mel, covered)``. The file is rejected (nothing covered,
    ``info["error"]`` set) unless ``len(wav) == MIC_CHUNK * len(ts)`` and the
    chunk timestamps are monotonic. Samples are then placed on the piecewise
    fit of ``mic_clock_segments``: in a segment fitted as ``a + b * j``, the
    window of frame start ``s`` begins at sample ``rint((s - a) / b)``. A
    frame is covered only if its window lies inside one accepted segment,
    at least ``MIC_BREAKPOINT_MARGIN_S`` from any breakpoint, in no other
    segment, and touches no stall chunk (|residual| on its segment's fit >
    ``MIC_SEGMENT_RESIDUAL_MAX_S``). ``info`` receives ``span`` (first, last
    chunk time), and, once fitted, ``segments``, ``breakpoint_drop`` (bool
    per frame: lost to the breakpoint margins and the clock gaps between
    segments), ``stall_drop`` (bool per frame: lost to a stall) and their
    sums ``n_breakpoint_drop`` and ``n_stall_drop``.
    """
    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    with h5py.File(mic_h5_path, "r") as h5:
        dev = next((k for k in h5 if "sink" in k), None)
        if dev is None:
            raise KeyError(f"no sink device in {mic_h5_path.name}: {list(h5)}")
        ts = _times(h5, dev, "chunk_timestamp")
    out = np.zeros((len(starts), N_MELS), dtype=np.float64)
    covered = np.zeros(len(starts), dtype=bool)
    chunk = MIC_CHUNK
    info = {} if info is None else info
    if len(ts) >= 2:
        # Nominal span, also for a rejected file: its frames are a gap, not
        # outside the recording.
        info["span"] = (float(ts[0] - (chunk - 1) / sr), float(ts[-1]))
    if len(ts) < 2 or len(wav) != chunk * len(ts):
        info["error"] = (
            f"{wav_path.name}: {len(wav)} samples vs {len(ts)} chunk timestamps "
            f"(expected {chunk} x {len(ts)} = {chunk * len(ts)})"
        )
        return out, covered
    if monotonic_violations(ts):
        info["error"] = f"{mic_h5_path.name}: chunk_timestamp not monotonic"
        return out, covered
    segs, stall = _mic_clock_fit(ts, chunk, float(sr))
    info["segments"] = segs
    win = round(WIDTH * sr)
    margin = round(MIC_BREAKPOINT_MARGIN_S * sr)
    fb = mel_filterbank(N_MELS, win, int(sr))
    i0_all = np.zeros(len(starts), dtype=np.int64)
    n_hit = np.zeros(len(starts), dtype=np.int64)
    usable = np.zeros(len(starts), dtype=bool)
    for i, seg in enumerate(segs):
        lo, hi = int(seg["k0"]) * chunk, int(seg["k1"]) * chunk
        i0 = np.rint((starts - seg["a"]) / seg["b"]).astype(np.int64)
        hit = (i0 >= lo) & (i0 + win <= hi)
        n_hit += hit
        if not seg["accepted"]:
            continue
        lo_m = lo + (margin if i > 0 else 0)
        hi_m = hi - (margin if i < len(segs) - 1 else 0)
        inner = (i0 >= lo_m) & (i0 + win <= hi_m)
        i0_all[inner] = i0[inner]
        usable |= inner
    ok = np.flatnonzero(usable & (n_hit == 1))
    # Stall: the window [i0, i0 + win) touches a flagged chunk.
    n_stall = np.concatenate([[0], np.cumsum(stall)])
    c0, c1 = i0_all[ok] // chunk, (i0_all[ok] + win - 1) // chunk + 1
    stalled = n_stall[c1] - n_stall[c0] > 0
    stall_drop = np.zeros(len(starts), dtype=bool)
    stall_drop[ok[stalled]] = True
    ok = ok[~stalled]
    for lo in range(0, len(ok), 2048):
        sel = ok[lo : lo + 2048]
        idx = i0_all[sel, None] + np.arange(win)[None, :]
        out[sel] = _log_mel(wav[idx].astype(np.float64), fb)
    covered[ok] = True
    # Frames lost to a breakpoint: window within the margin of a segment
    # boundary, in the clock gap between two segments, or claimed twice.
    near = np.zeros(len(starts), dtype=bool)
    for prev, nxt in itertools.pairwise(segs):
        z0 = min(prev["t1"], nxt["t0"]) - MIC_BREAKPOINT_MARGIN_S
        z1 = max(prev["t1"], nxt["t0"]) + MIC_BREAKPOINT_MARGIN_S
        near |= (starts + WIDTH > z0) & (starts < z1)
    bp_drop = near & ~covered & ~stall_drop
    info["breakpoint_drop"] = bp_drop
    info["stall_drop"] = stall_drop
    info["n_breakpoint_drop"] = int(bp_drop.sum())
    info["n_stall_drop"] = int(stall_drop.sum())
    return out, covered


# --------------------------------------------------------------------------
# Vision
# --------------------------------------------------------------------------
def nearest_frame(
    frame_t: np.ndarray, starts: np.ndarray, tol: float = VISION_TOL_S
) -> tuple[np.ndarray, np.ndarray]:
    """Index of the video frame nearest to each tick, and whether within tol."""
    pos = np.clip(np.searchsorted(frame_t, starts), 1, len(frame_t) - 1)
    left, right = frame_t[pos - 1], frame_t[pos]
    idx = np.where(np.abs(starts - left) <= np.abs(right - starts), pos - 1, pos)
    ok = np.abs(frame_t[idx] - starts) <= tol
    return idx.astype(np.int64), ok


class FfmpegError(RuntimeError):
    """ffmpeg exited non-zero; the message carries the stderr tail."""


def _parse_ffmpeg_version(text: str) -> tuple[int, int] | None:
    m = re.search(r"ffmpeg version n?(\d+)\.(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def ffmpeg_exe() -> str:
    return shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def check_ffmpeg() -> str:
    """Fail fast unless ffmpeg >= 5.1; return its version line for the manifest."""
    try:
        res = subprocess.run([ffmpeg_exe(), "-version"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            f"ffmpeg not usable ({exc}); install ffmpeg >= 5.1 or use --no-vision"
        ) from exc
    line = res.stdout.splitlines()[0] if res.stdout else ""
    ver = _parse_ffmpeg_version(line)
    if ver is not None and ver < FFMPEG_MIN:
        raise SystemExit(
            f"ffmpeg {ver[0]}.{ver[1]} is too old: `-fps_mode` needs >= "
            f"{FFMPEG_MIN[0]}.{FFMPEG_MIN[1]} ({line})"
        )
    return line


def decode_grids(mp4: Path, wanted: np.ndarray) -> tuple[dict[int, np.ndarray], int]:
    """Stream-decode the video once and keep the (16,16) grid of wanted frames.

    ffmpeg writes 64x64 gray frames to a pipe: nothing but the reduced grid is
    ever kept. Returns ``(grids, n_decoded_frames)``. Raises ``FfmpegError``
    (with the stderr tail) when ffmpeg exits non-zero.
    """
    cmd = [
        ffmpeg_exe(),
        "-v",
        "error",
        "-i",
        str(mp4),
        "-vf",
        f"scale={VISION_DECODE_PX}:{VISION_DECODE_PX}:flags=area,format=gray",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    want = set(int(i) for i in wanted)
    size = VISION_DECODE_PX * VISION_DECODE_PX
    grids: dict[int, np.ndarray] = {}
    n = 0
    with (
        tempfile.TemporaryFile() as err,
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err) as proc,
    ):
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            if n in want:
                img = Image.frombytes("L", (VISION_DECODE_PX, VISION_DECODE_PX), buf)
                grids[n] = frame_to_grid(img)
            n += 1
        rc = proc.wait()
        err.seek(0)
        tail = err.read().decode("utf-8", "replace").strip()[-500:]
    if rc != 0:
        raise FfmpegError(f"ffmpeg exit {rc} on {mp4.name}: {tail or 'no stderr'}")
    return grids, n


# --------------------------------------------------------------------------
# Per-session / per-subject extraction
# --------------------------------------------------------------------------
def _bump(counts: dict[str, int], key: str, n: int = 1) -> None:
    counts[key] = counts.get(key, 0) + n


def _lowpass_runs(
    x: np.ndarray, valid: np.ndarray, run: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Low-pass each maximal contiguous valid run; too-short runs are dropped."""
    out = x.copy()
    keep = valid.copy()
    n, i = len(x), 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and valid[j + 1] and run[j + 1] == run[i]:
            j += 1
        if j - i + 1 >= LOWPASS_MIN_FRAMES:
            out[i : j + 1] = lowpass(x[i : j + 1], hz=HZ, cutoff=LOWPASS_CUTOFF_HZ)
        else:
            keep[i : j + 1] = False
        i = j + 1
    return out, keep


def _extract_session(
    wear: Path,
    raw_sub: Path,
    subject: str,
    vision: bool,
    counts: dict[str, int],
    label_index: dict[str, int],
    report: list[dict[str, object]],
) -> dict[str, np.ndarray] | None:
    """One wearable file -> kept frames; appends a span/monotonicity report row."""
    entry: dict[str, object] = {"subject": subject, "file": wear.name, "streams": {}}
    report.append(entry)
    try:
        return _session_body(wear, raw_sub, subject, vision, counts, label_index, entry)
    except NonMonotonicTimeError as exc:
        _bump(counts, "session_skipped:nonmonotonic")
        entry["skipped"] = str(exc)
        print(f"  skip {wear.name}: {exc}", flush=True)
        return None


def _session_body(
    wear: Path,
    raw_sub: Path,
    subject: str,
    vision: bool,
    counts: dict[str, int],
    label_index: dict[str, int],
    entry: dict[str, object],
) -> dict[str, np.ndarray] | None:
    spans: dict[str, tuple[float, float, int, int]] = {}
    entry["streams"] = spans
    inspan: dict[str, np.ndarray] = {}
    prefix = wear.name.split("_streamLog_")[0]
    with h5py.File(wear, "r") as h5:
        missing = _missing_streams(h5)
        if missing:
            _bump(counts, "session_skipped:missing_stream")
            print(f"  skip {wear.name}: missing {missing}", flush=True)
            return None
        rows, rtimes = _read_activities(h5)
        if monotonic_violations(rtimes):
            raise NonMonotonicTimeError("experiment-activities/activities: time_s steps back")
        for label, kind, _validity in rows:
            if kind == "Start":
                _bump(counts, f"seen:{label}")
        seg_raw: list[str] = []
        segments, seg_counts = activity_segments(rows, rtimes, seg_raw)
        if segments:
            entry["labelled"] = (min(g[0] for g in segments), max(g[1] for g in segments))
        for k, v in seg_counts.items():
            _bump(counts, k, v)
        # Frames, one run id per segment so filters never cross segments.
        starts_l, labels_l, run_l, raw_l = [], [], [], []
        for si, seg in enumerate(segments):
            s, lab = frame_grid([seg], hz=HZ)
            starts_l.append(s)
            labels_l.append(lab)
            run_l.append(np.full(len(s), si, dtype=np.int64))
            ridx = label_index.setdefault(seg_raw[si], len(label_index))
            raw_l.append(np.full(len(s), ridx, dtype=np.int64))
        starts = np.concatenate(starts_l) if starts_l else np.zeros(0)
        labels = np.concatenate(labels_l) if labels_l else np.zeros(0, dtype=np.int64)
        run = np.concatenate(run_l) if run_l else np.zeros(0, dtype=np.int64)
        raw_idx = np.concatenate(raw_l) if raw_l else np.zeros(0, dtype=np.int64)
        n = len(starts)
        _bump(counts, "frames_total", n)
        if n == 0:
            return None
        valid = np.ones(n, dtype=bool)
        feats: dict[str, np.ndarray] = {}
        reasons: dict[str, np.ndarray] = {}

        parts, oks = [], []
        for side in ("left", "right"):
            dev = f"tactile-glove-{side}"
            data = np.asarray(h5[dev]["tactile_data"]["data"], dtype=np.float32)
            pooled = pool_tactile(data.reshape(-1, 32, 32))  # pool first: mean is linear
            tt = _checked_times(h5, dev, "tactile_data", spans)
            m, ok = window_mean(tt, pooled, starts, WIDTH, MAX_GAP, counts)
            parts.append(m)
            oks.append(ok)
        feats["tactile"] = np.concatenate(parts, axis=1)
        reasons["tactile"] = oks[0] & oks[1]

        parts, oks = [], []
        for side in ("left", "right"):
            dev = f"myo-{side}"
            emg = np.asarray(h5[dev]["emg"]["data"], dtype=np.float64)
            te = _checked_times(h5, dev, "emg", spans)
            rms, ok = window_rms(te, emg, starts, WIDTH, MAX_GAP, counts)
            parts.append(emg_sectors(rms))
            oks.append(ok)
        feats["force"] = np.concatenate(parts, axis=1)
        reasons["force"] = oks[0] & oks[1]

        acc = np.asarray(h5["myo-right"]["acceleration_g"]["data"], dtype=np.float64)
        ta = _checked_times(h5, "myo-right", "acceleration_g", spans)
        grav, gok = window_mean(ta, acc, starts, WIDTH, MAX_GAP, counts)
        feats["gravity"] = grav
        reasons["gravity"] = gok

        if vision:
            vdev = "eye-tracking-video-world"
            mp4 = raw_sub / f"{prefix}_{subject}_eye-tracking-video-world_frame.mp4"
            vok = np.zeros(n, dtype=bool)
            vis = np.zeros((n, 16, 16), dtype=np.float32)
            if mp4.exists() and vdev in h5 and "frame_timestamp" in h5[vdev]:
                ft = _checked_times(h5, vdev, "frame_timestamp", spans)
                if len(ft) >= 2:
                    inspan["vision"] = (starts >= ft[0] - VISION_TOL_S) & (
                        starts <= ft[-1] + VISION_TOL_S
                    )
                    idx, near = nearest_frame(ft, starts)
                    try:
                        grids, n_dec = decode_grids(mp4, np.unique(idx[near]))
                    except FfmpegError as exc:
                        _bump(counts, "session_vision_ffmpeg_failed")
                        print(f"  {exc}", flush=True)
                        grids, n_dec = {}, -1
                    if n_dec < 0:
                        pass
                    elif n_dec != len(ft):
                        _bump(counts, "session_vision_frame_count_mismatch")
                        print(f"  {mp4.name}: {n_dec} decoded vs {len(ft)} timestamps", flush=True)
                    else:
                        for i in np.flatnonzero(near):
                            vis[i] = grids[int(idx[i])]
                        vok = near
            else:
                _bump(counts, "session_without_video")
            feats["vision"] = vis
            reasons["vision"] = vok

        aud = np.zeros((n, N_MELS))
        aok = np.zeros(n, dtype=bool)
        # Frames lost to a breakpoint or a stall in some file, counted once
        # per frame and only if no other file covers them.
        bp_lost = np.zeros(n, dtype=bool)
        stall_lost = np.zeros(n, dtype=bool)
        fitted = False
        for mic_h5 in sorted(raw_sub.glob(f"*_streamLog_actionNet-microphone_{subject}.hdf5")):
            wav = mic_h5.with_name(mic_h5.stem + "_sink_chunked_data.wav")
            if not wav.exists():
                continue
            info: dict[str, object] = {}
            mel, cov = audio_frames(wav, mic_h5, starts, info)
            segs = info.get("segments")
            if isinstance(segs, list):
                fits = entry.setdefault("mic_fits", {})
                assert isinstance(fits, dict)
                fits[mic_h5.name] = segs
                for seg in segs:
                    if not seg["accepted"]:
                        _bump(counts, "audio_segment_bad_clock_fit")
                        print(
                            f"  audio_segment_bad_clock_fit: {mic_h5.name} chunks "
                            f"{int(seg['k0'])}..{int(seg['k1'])}, rate {seg['rate']:.1f} Hz, "
                            f"p99 |residual| {seg['res_p99']:.3f} s, "
                            f"max |residual| {seg['res_max']:.3f} s",
                            flush=True,
                        )
                _bump(counts, "audio_breakpoints", len(segs) - 1)
                fitted = True
                for key, acc in (("breakpoint_drop", bp_lost), ("stall_drop", stall_lost)):
                    mask = info.get(key)
                    if isinstance(mask, np.ndarray):
                        acc |= mask
            if "error" in info:
                _bump(counts, "audio_file_inconsistent")
                print(f"  audio_file_inconsistent: {info['error']}", flush=True)
            span = info.get("span")
            if isinstance(span, tuple):
                spans[f"microphone/{mic_h5.name}"] = (span[0], span[1], 0, 0)
                inside = (starts >= span[0]) & (starts + WIDTH <= span[1])
                inspan["audio"] = inspan.get("audio", np.zeros(n, dtype=bool)) | inside
            take = cov & ~aok
            aud[take] = mel[take]
            aok |= cov
        if fitted:
            _bump(counts, "audio_breakpoint_drop", int((bp_lost & ~aok).sum()))
            _bump(counts, "audio_stall_drop", int((stall_lost & ~aok & ~bp_lost).sum()))
        feats["audio"] = aud
        reasons["audio"] = aok

    # Low-pass gravity inside each segment, then drop invalid frames.
    for name, ok in reasons.items():
        ok = ok & np.isfinite(feats[name].reshape(n, -1)).all(axis=1)
        reasons[name] = ok
    gvalid = reasons["gravity"]
    feats["gravity"], gkeep = _lowpass_runs(feats["gravity"], gvalid, run)
    reasons["gravity"] = gkeep
    for name in ("tactile", "force", "gravity", "vision", "audio"):
        if name not in reasons:
            continue
        bad = valid & ~reasons[name]
        _bump(counts, f"drop:{name}", int(bad.sum()))
        if name in ("audio", "vision"):
            ins = inspan.get(name, np.zeros(n, dtype=bool))
            _bump(counts, f"drop:{name}:outside_span", int((bad & ~ins).sum()))
            _bump(counts, f"drop:{name}:gap", int((bad & ins).sum()))
        valid &= reasons[name]
    _bump(counts, "frames_kept", int(valid.sum()))
    out = {k: v[valid].astype(np.float32) for k, v in feats.items()}
    out["labels"] = labels[valid].astype(np.int64)
    # Side-file columns (rows.pt); not part of the world contract.
    out["time_s"] = starts[valid].astype(np.float64)
    out["segment_id"] = run[valid].astype(np.int64)
    out["raw_label_idx"] = raw_idx[valid].astype(np.int64)
    return out


def extract_subject_with_counts(
    raw_dir: Path,
    subject: str,
    vision: bool = True,
    label_index: dict[str, int] | None = None,
    report: list[dict[str, object]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Per-subject raw (not z-scored) tensors and the drop counters.

    The arrays also carry the side columns ``time_s``, ``segment_id`` (unique
    within the subject) and ``raw_label_idx`` (into ``label_index``, filled in
    first-seen order). ``report`` collects one span/monotonicity row per file.
    """
    label_index = {} if label_index is None else label_index
    report = [] if report is None else report
    raw_sub = raw_dir / subject
    wears = sorted(raw_sub.glob(f"*_streamLog_actionNet-wearables_{subject}.hdf5"))
    if not wears:
        raise FileNotFoundError(f"no wearable HDF5 for {subject} in {raw_sub}")
    counts: dict[str, int] = {}
    sessions: list[dict[str, np.ndarray]] = []
    seg_offset = 0
    for wear in wears:
        print(f"{subject}: {wear.name}", flush=True)
        sess = _extract_session(wear, raw_sub, subject, vision, counts, label_index, report)
        if sess is not None and len(sess["labels"]) > 0:
            sess["segment_id"] = sess["segment_id"] + seg_offset
            seg_offset = int(sess["segment_id"].max()) + 1
            sessions.append(sess)
    if not sessions:
        keys = ["audio", "tactile", "gravity", "force", "labels"] + (["vision"] if vision else [])
        shapes = {"audio": (0, N_MELS), "tactile": (0, 32), "gravity": (0, 3), "force": (0, 6)}
        empty = {k: np.zeros(shapes.get(k, (0,)), dtype=np.float32) for k in keys}
        empty["labels"] = np.zeros(0, dtype=np.int64)
        empty["time_s"] = np.zeros(0, dtype=np.float64)
        empty["segment_id"] = np.zeros(0, dtype=np.int64)
        empty["raw_label_idx"] = np.zeros(0, dtype=np.int64)
        if vision:
            empty["vision"] = np.zeros((0, 16, 16), dtype=np.float32)
        return empty, counts
    merged = {k: np.concatenate([s[k] for s in sessions], axis=0) for k in sessions[0]}
    return merged, counts


def extract_subject(raw_dir: Path, subject: str, vision: bool = True) -> dict[str, np.ndarray]:
    """Importable entry point: tensors only (see ``extract_subject_with_counts``)."""
    return extract_subject_with_counts(raw_dir, subject, vision)[0]


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _input_sha(raw_dir: Path, path: Path) -> str:
    """sha256 from the fetch MANIFEST.json when it matches, else hash the file."""
    man = raw_dir / "MANIFEST.json"
    if man.exists():
        rel = path.relative_to(raw_dir).as_posix()
        entry = json.loads(man.read_text()).get(rel)
        if entry and entry.get("bytes") == path.stat().st_size and entry.get("sha256"):
            return str(entry["sha256"])
    return _sha256(path)


def _span_lines(report: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for ent in report:
        lines.append(f"### {ent['subject']} `{ent['file']}`")
        lines.append("")
        if "skipped" in ent:
            lines += [f"SKIPPED: {ent['skipped']}", ""]
            continue
        lab = ent.get("labelled")
        ref = lab[0] if isinstance(lab, tuple) else 0.0
        if isinstance(lab, tuple):
            lines.append(f"- labelled segments: {lab[0] - ref:.1f} .. {lab[1] - ref:.1f} s")
        spans = ent["streams"]
        assert isinstance(spans, dict)
        for name in sorted(spans):
            first, last, n, bad = spans[name]
            lines.append(
                f"- `{name}`: {first - ref:.1f} .. {last - ref:.1f} s, n={n}, backwards steps={bad}"
            )
        fits = ent.get("mic_fits", {})
        assert isinstance(fits, dict)
        for name in sorted(fits):
            segs = fits[name]
            bps = [f"{sg['t0'] - ref:.1f} s (step {sg['step']:+.3f} s)" for sg in segs[1:]]
            lines.append(
                f"- microphone clock fit `{name}`: {len(segs)} segment(s); breakpoints: "
                + (", ".join(bps) or "none")
            )
            for sg in segs:
                lines.append(
                    f"  - chunks {int(sg['k0'])}..{int(sg['k1'])} "
                    f"({sg['t0'] - ref:.1f} .. {sg['t1'] - ref:.1f} s): implied rate "
                    f"{sg['rate']:.3f} Hz (|dev| {sg['rate_dev']:.2e}), |residual| median "
                    f"{sg['res_median']:.4f} s, p99 {sg['res_p99']:.4f} s, max "
                    f"{sg['res_max']:.4f} s: " + ("accepted" if sg["accepted"] else "REJECTED")
                )
        lines.append("")
    return lines


def _manifest(
    raw_dir: Path,
    out: Path,
    subjects: Sequence[str],
    tensors: dict[str, torch.Tensor],
    stats: dict[str, dict[str, dict[str, list[float]]]],
    counts: dict[str, dict[str, int]],
    n_sub: dict[str, int],
    vision: bool,
    report: list[dict[str, object]],
    label_list: list[str],
    ffmpeg_line: str | None,
) -> str:
    labels = tensors["labels"].numpy()
    n_all = len(labels)
    freq = np.bincount(labels, minlength=4)
    lines = [
        "# ActionSense real-data cache manifest",
        "",
        f"- Licence of the derived tensors: **{LICENCE}**",
        "- Code: Apache-2.0. The tensors are NOT covered by it and must not be used "
        "commercially. Attribution: DelPreto et al., ActionSense, NeurIPS 2022 "
        "(Datasets and Benchmarks), MIT CSAIL.",
        f"- Subjects (concatenated in this order): {', '.join(subjects)}",
        f"- Rows: {n_all} at {HZ:g} Hz; per subject: "
        + ", ".join(f"{s}={n_sub[s]}" for s in subjects),
        "",
        "## Reduction parameters",
        "",
        f"- Frame window [t, t+{WIDTH:g} s), non-overlapping, only full frames inside "
        "Good-validity Start/Stop segments; gaps <= "
        f"{MAX_GAP:g} s interpolated, otherwise the frame is dropped.",
        "- audio: sink microphone WAV, window of 0.1 s (= hop), Hann, power spectrum "
        f"of n_fft = window samples, {N_MELS}-bin HTK mel bank, log(x + 1e-6); frames "
        "located through a piecewise least-squares line of the microphone HDF5 "
        "`chunk_timestamp` (time of the last sample of each chunk) against sample index. "
        f"Breakpoints: the {MIC_STEP_WINDOW}-chunk median residual of the global fit "
        f"steps by more than {MIC_STEP_DETECT_S:g} s; frames within "
        f"{MIC_BREAKPOINT_MARGIN_S:g} s of a breakpoint are dropped. A segment is "
        f"rejected if its rate deviates from the WAV rate by more than {MIC_RATE_TOL:g} "
        f"(relative) or its p99 |residual| exceeds {MIC_SEGMENT_RESIDUAL_MAX_S:g} s, on a "
        f"refit without the chunks whose |residual| exceeds {MIC_SEGMENT_RESIDUAL_MAX_S:g} s "
        "if there are any; in an accepted segment, frames touching a chunk whose "
        f"|residual| on that fit exceeds {MIC_SEGMENT_RESIDUAL_MAX_S:g} s are dropped (stall, "
        "`audio_stall_drop`); a file is "
        "rejected if its WAV length is not 2048 x the chunk count or its timestamps are "
        "not monotonic. A constant host latency is absorbed into the intercept "
        "(fits per file under Stream spans).",
        "- vision: world video frame nearest to the tick (|dt| <= "
        f"{VISION_TOL_S:g} s), ffmpeg gray {VISION_DECODE_PX}x{VISION_DECODE_PX} "
        "area-scaled, then 16x16 mean intensity (`frame_to_grid`). "
        + ("" if vision else "**NOT EXTRACTED in this run (--no-vision).**"),
        "- tactile: each glove 32x32 averaged in 8x8 blocks to 16, left then right = 32.",
        "- gravity: right Myo acceleration_g window mean, then 2nd-order zero-phase "
        f"Butterworth low-pass {LOWPASS_CUTOFF_HZ:g} Hz at {HZ:g} Hz, applied inside each "
        f"labelled segment (runs shorter than {LOWPASS_MIN_FRAMES} frames are dropped).",
        "- force: Myo EMG 100 ms RMS, per arm mean of channels 0-2, 3-5, 6-7; left then right = 6.",
        "- z-score: per subject, per channel, over that subject's kept frames; std < 1e-8 gives 0.",
        "",
        "## Label families",
        "",
        "Prefix rules (lower-cased label starts with; first match wins): "
        + ", ".join(f"`{n}`->{f}" for n, f in FAMILY_RULES),
        "",
        "| family | meaning | raw label strings seen |",
        "|---|---|---|",
    ]
    seen: dict[str, int] = {}
    for c in counts.values():
        for k, v in c.items():
            if k.startswith("seen:"):
                seen[k[5:]] = seen.get(k[5:], 0) + v
    for fam in range(4):
        raw = sorted(s for s in seen if label_family(s) == fam)
        lines.append(f"| {fam} | {FAMILY_NAMES[fam]} | {'; '.join(raw) or '-'} |")
    unmapped = sorted(s for s in seen if label_family(s) is None)
    lines += [
        "",
        "Unmapped labels (dropped, never guessed): " + ("; ".join(unmapped) or "none"),
        "",
    ]
    lines += ["## Class frequencies", "", "| family | rows | share |", "|---|---|---|"]
    for fam in range(4):
        lines.append(f"| {fam} | {int(freq[fam])} | {freq[fam] / max(n_all, 1):.4f} |")
    lines += [
        "",
        f"Majority-class baseline: {freq.max() / max(n_all, 1):.4f}",
        "",
        "## Counts per subject",
        "",
        "`unclosed`/`bad`/`unmapped:*` count activity pairs; `drop:*` and `frames_*` "
        "count 10 Hz frames; `seen:*` counts Start rows.",
        "",
    ]
    for s in subjects:
        lines.append(f"### {s}")
        lines.append("")
        for k in sorted(counts[s]):
            if not k.startswith("seen:"):
                lines.append(f"- `{k}`: {counts[s][k]}")
        lines.append("")
    lines += [
        "## Stream spans and clock checks",
        "",
        "Per wearable file: first/last `time_s` of every stream, relative to the "
        "first labelled segment start (s), sample count and backwards steps (a file "
        "with any backwards step is skipped). Media spans (world video "
        "`frame_timestamp`, microphone sample times) are shown against the labelled "
        "span; `drop:<m>:outside_span` / `drop:<m>:gap` split the media losses.",
        "",
    ]
    lines += _span_lines(report)
    lines += ["## Raw label index (`raw_label_idx` in rows.pt)", ""]
    lines += [f"{i}. {name}" for i, name in enumerate(label_list)]
    lines += [
        "",
        "`rows.pt` is a side file (not part of the world contract, ignored by "
        "ActionSenseWorld): dict of `time_s` float64 (frame start, wearable clock), "
        "`segment_id` long (unique across subjects) and `raw_label_idx` long, "
        "aligned row for row with the tensors.",
        "",
        f"- ffmpeg: {ffmpeg_line or 'not used (--no-vision)'}",
        "",
    ]
    lines += ["## Per-subject z-score statistics", "", "```json", json.dumps(stats), "```", ""]
    lines += ["## SHA256 of the raw inputs", ""]
    for s in subjects:
        for p in list_inputs(raw_dir, s):
            lines.append(f"- `{p.relative_to(raw_dir).as_posix()}`: `{_input_sha(raw_dir, p)}`")
    lines += ["", "## SHA256 of the tensors", ""]
    for name in (*tensors, "rows"):
        lines.append(f"- `{name}.pt`: `{_sha256(out / (name + '.pt'))}`")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/actionsense_raw"))
    ap.add_argument("--out", type=Path, default=Path("data/actionsense"))
    ap.add_argument("--subjects", nargs="+", default=["S00", "S02", "S03", "S04", "S05"])
    ap.add_argument(
        "--no-vision",
        action="store_true",
        help="debug only: skip ffmpeg, write no vision.pt",
    )
    args = ap.parse_args(argv)
    vision = not args.no_vision

    ffmpeg_line = check_ffmpeg() if vision else None
    label_index: dict[str, int] = {}
    report: list[dict[str, object]] = []
    side: dict[str, list[torch.Tensor]] = {"time_s": [], "segment_id": [], "raw_label_idx": []}
    seg_base = 0
    cols: dict[str, list[torch.Tensor]] = {}
    stats: dict[str, dict[str, dict[str, list[float]]]] = {}
    counts: dict[str, dict[str, int]] = {}
    n_sub: dict[str, int] = {}
    subj_ids: list[torch.Tensor] = []
    for sid in args.subjects:
        arrays, cnt = extract_subject_with_counts(args.raw, sid, vision, label_index, report)
        try:
            n = check_aligned(arrays)
        except ValueError as exc:
            print(f"ABORT {sid}: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        if n == 0:
            print(f"ABORT {sid}: no usable frames, counts={cnt}", file=sys.stderr)
            raise SystemExit(2)
        counts[sid] = cnt
        n_sub[sid] = n
        seg = arrays["segment_id"]
        side["time_s"].append(torch.from_numpy(arrays["time_s"]))
        side["segment_id"].append(torch.from_numpy(seg + seg_base))
        side["raw_label_idx"].append(torch.from_numpy(arrays["raw_label_idx"]))
        seg_base += int(seg.max()) + 1
        stats[sid] = {}
        for name in MODALITIES:
            if name not in arrays:
                continue
            flat = arrays[name].reshape(n, -1).astype(np.float64)
            z, mean, std = zscore(flat)
            stats[sid][name] = {"mean": mean.round(6).tolist(), "std": std.round(6).tolist()}
            cols.setdefault(name, []).append(
                torch.from_numpy(z.astype(np.float32).reshape(arrays[name].shape))
            )
        cols.setdefault("labels", []).append(torch.from_numpy(arrays["labels"]))
        subj_ids.append(torch.full((n,), int(sid.lstrip("S")), dtype=torch.long))
        print(f"{sid}: {n} frames kept of {cnt.get('frames_total', 0)}", flush=True)

    tensors = {k: torch.cat(v, dim=0) for k, v in cols.items()}
    tensors["subject"] = torch.cat(subj_ids)
    rows = {k: torch.cat(v, dim=0) for k, v in side.items()}
    try:
        check_aligned({k: v.numpy() for k, v in {**tensors, **rows}.items()})
    except ValueError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    args.out.mkdir(parents=True, exist_ok=True)
    for name, t in tensors.items():
        torch.save(t, args.out / f"{name}.pt")
        print(f"wrote {name}.pt {tuple(t.shape)} {t.dtype}", flush=True)
    torch.save(rows, args.out / "rows.pt")
    print(f"wrote rows.pt {[(k, tuple(v.shape)) for k, v in rows.items()]}", flush=True)
    text = _manifest(
        args.raw,
        args.out,
        args.subjects,
        tensors,
        stats,
        counts,
        n_sub,
        vision,
        report,
        sorted(label_index, key=label_index.__getitem__),
        ffmpeg_line,
    )
    (args.out / "MANIFEST.md").write_text(text)
    print(f"wrote {args.out}/MANIFEST.md", flush=True)


if __name__ == "__main__":
    main()
