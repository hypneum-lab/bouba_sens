"""Pure, network-free reduction helpers for the ActionSense world."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt  # type: ignore[import-untyped]

FAMILY_RULES: tuple[tuple[str, int], ...] = (
    ("peel", 0),
    ("slic", 0),
    ("spread", 1),
    ("jar", 1),
    ("clean", 2),
    ("wipe", 2),
    ("pour", 3),
    ("set table", 3),
    ("dishwasher", 3),
)

_EPS = 1e-9


def label_family(label: str) -> int | None:
    """Map an activity label to a family id, or None if unmapped."""
    low = label.lower()
    for needle, fam in FAMILY_RULES:
        if needle in low:
            return fam
    return None


def activity_segments(
    rows: list[tuple[str, str, str]], times: np.ndarray
) -> tuple[list[tuple[float, float, int]], dict[str, int]]:
    """Pair Start/Stop rows into (t_start, t_end, family) segments.

    ``times[i]`` is the timestamp of ``rows[i]``. Counts keys: ``unclosed``,
    ``bad``, ``unmapped:<label>``.
    """
    segments: list[tuple[float, float, int]] = []
    counts: dict[str, int] = {"unclosed": 0, "bad": 0}
    open_rows: dict[str, tuple[float, str]] = {}
    for (label, kind, validity), t in zip(rows, times, strict=True):
        if kind == "Start":
            if label in open_rows:
                counts["unclosed"] += 1
            open_rows[label] = (float(t), validity)
        elif kind == "Stop" and label in open_rows:
            t0, start_validity = open_rows.pop(label)
            fam = label_family(label)
            if fam is None:
                key = f"unmapped:{label}"
                counts[key] = counts.get(key, 0) + 1
            elif start_validity != "Good" or validity != "Good":
                counts["bad"] += 1
            else:
                segments.append((t0, float(t), fam))
    counts["unclosed"] += len(open_rows)
    return segments, counts


def frame_grid(
    segments: list[tuple[float, float, int]], hz: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Full non-overlapping frames of 1/hz s inside each segment."""
    width = 1.0 / hz
    starts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for t0, t1, fam in segments:
        n = int(np.floor((t1 - t0) * hz + _EPS))
        if n <= 0:
            continue
        starts.append(t0 + width * np.arange(n))
        labels.append(np.full(n, fam, dtype=np.int64))
    if not starts:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    return np.concatenate(starts), np.concatenate(labels)


def _window_reduce(
    t: np.ndarray,
    x: np.ndarray,
    starts: np.ndarray,
    width: float,
    max_gap: float,
    square: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    lo = np.searchsorted(t, starts, side="left")
    hi = np.searchsorted(t, starts + width, side="left")
    src = x * x if square else x
    csum = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(src, axis=0)])
    cnt = hi - lo
    out = np.zeros((len(starts), x.shape[1]))
    valid = cnt > 0
    ok = np.flatnonzero(valid)
    out[ok] = (csum[hi[ok]] - csum[lo[ok]]) / cnt[ok, None]
    for i in np.flatnonzero(~valid):
        b, a = lo[i] - 1, hi[i]
        if b < 0 or a >= len(t) or t[a] - t[b] > max_gap + _EPS:
            continue
        centre = starts[i] + width / 2.0
        w = (centre - t[b]) / (t[a] - t[b]) if t[a] > t[b] else 0.0
        val = x[b] + w * (x[a] - x[b])
        out[i] = val * val if square else val
        valid[i] = True
    if square:
        out = np.sqrt(out)
    return out, valid


def window_mean(
    t: np.ndarray,
    x: np.ndarray,
    starts: np.ndarray,
    width: float = 0.1,
    max_gap: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame mean of x over [start, start+width) with gap interpolation."""
    return _window_reduce(t, x, starts, width, max_gap, square=False)


def window_rms(
    t: np.ndarray,
    x: np.ndarray,
    starts: np.ndarray,
    width: float = 0.1,
    max_gap: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame RMS of x over [start, start+width) with gap interpolation."""
    return _window_reduce(t, x, starts, width, max_gap, square=True)


def pool_tactile(frames: np.ndarray) -> np.ndarray:
    """Average 8x8 blocks of (N, 32, 32) frames into (N, 16)."""
    n = frames.shape[0]
    pooled: np.ndarray = frames.reshape(n, 4, 8, 4, 8).mean(axis=(2, 4))
    return pooled.reshape(n, 16)


def emg_sectors(rms: np.ndarray) -> np.ndarray:
    """Reduce (N, 8) EMG RMS to (N, 3) sector means."""
    return np.stack(
        [rms[:, 0:3].mean(axis=1), rms[:, 3:6].mean(axis=1), rms[:, 6:8].mean(axis=1)],
        axis=1,
    )


def lowpass(x: np.ndarray, hz: float = 10.0, cutoff: float = 0.5) -> np.ndarray:
    """2nd-order zero-phase Butterworth low-pass along axis 0."""
    sos = butter(2, cutoff, btype="low", fs=hz, output="sos")
    return np.asarray(sosfiltfilt(sos, x, axis=0))


def zscore(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z, mean, std); z is 0 where std < 1e-8."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    safe = np.where(std < 1e-8, 1.0, std)
    z = np.where(std < 1e-8, 0.0, (x - mean) / safe)
    return z, mean, std


def check_aligned(tensors: dict[str, np.ndarray]) -> int:
    """Return the common first dimension or raise naming every length."""
    lengths = {name: int(a.shape[0]) for name, a in tensors.items()}
    if len(set(lengths.values())) != 1:
        desc = ", ".join(f"{k}={v}" for k, v in lengths.items())
        raise ValueError(f"modalities not aligned: {desc}")
    return next(iter(lengths.values()))
