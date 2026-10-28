"""Pure, network-free reduction helpers for the ActionSense world."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import butter, sosfiltfilt  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from PIL import Image

VISION_SHAPE = (16, 16)

# Prefix rules on the lower-cased label (spec section 6). Matching the leading
# words, not a substring, keeps "Spread ... on a bread slice" out of cutting.
FAMILY_RULES: tuple[tuple[str, int], ...] = (
    ("peel", 0),
    ("slice", 0),
    ("spread", 1),
    ("open/close a jar", 1),
    ("clean", 2),
    ("wipe", 2),
    ("pour", 3),
    ("set table", 3),
    ("load dishwasher", 3),
    ("unload dishwasher", 3),
)

_EPS = 1e-9


def label_family(label: str) -> int | None:
    """Map an activity label to a family id, or None if unmapped."""
    low = label.strip().lower()
    for prefix, fam in FAMILY_RULES:
        if low.startswith(prefix):
            return fam
    return None


def activity_segments(
    rows: list[tuple[str, str, str]],
    times: np.ndarray,
    raw_labels: list[str] | None = None,
) -> tuple[list[tuple[float, float, int]], dict[str, int]]:
    """Pair Start/Stop rows into (t_start, t_end, family) segments.

    ``times[i]`` is the timestamp of ``rows[i]``. Counts keys: ``unclosed``,
    ``bad``, ``unmapped:<label>``. When ``raw_labels`` is given, the raw label
    string of each returned segment is appended to it, in segment order.
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
                if raw_labels is not None:
                    raw_labels.append(label)
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
    counts: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    # Non-finite samples are masked out (treated like missing samples) so one
    # NaN cannot poison every later cumulative sum.
    finite = np.isfinite(x).all(axis=1) & np.isfinite(t)
    n_bad = int((~finite).sum())
    if n_bad:
        if counts is not None:
            counts["nonfinite_samples"] = counts.get("nonfinite_samples", 0) + n_bad
        t, x = t[finite], x[finite]
    if len(t) == 0:
        return np.zeros((len(starts), x.shape[1])), np.zeros(len(starts), dtype=bool)
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
    counts: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame mean of x over [start, start+width) with gap interpolation.

    Non-finite samples are masked and counted in ``counts["nonfinite_samples"]``.
    """
    return _window_reduce(t, x, starts, width, max_gap, square=False, counts=counts)


def window_rms(
    t: np.ndarray,
    x: np.ndarray,
    starts: np.ndarray,
    width: float = 0.1,
    max_gap: float = 0.2,
    counts: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame RMS of x over [start, start+width) with gap interpolation.

    Non-finite samples are masked and counted in ``counts["nonfinite_samples"]``.
    """
    return _window_reduce(t, x, starts, width, max_gap, square=True, counts=counts)


def monotonic_violations(t: np.ndarray) -> int:
    """Number of strictly backwards steps in a timestamp array."""
    t = np.asarray(t, dtype=np.float64)
    return int((np.diff(t) < 0).sum()) if len(t) > 1 else 0


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


# --------------------------------------------------------------------------
# Shared with scripts/extract_ccfilm_features.py (moved here, behaviour kept).
# --------------------------------------------------------------------------
def hz_to_mel(f: np.ndarray) -> np.ndarray:
    """HTK mel scale."""
    out: np.ndarray = 2595.0 * np.log10(1.0 + f / 700.0)
    return out


def mel_to_hz(m: np.ndarray) -> np.ndarray:
    """Inverse HTK mel scale."""
    out: np.ndarray = 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    return out


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


def frame_to_grid(img: Image.Image) -> np.ndarray:
    """Decoded frame -> (16, 16) mean-intensity patch grid in [0, 1]."""
    g = img.convert("L").resize((VISION_SHAPE[1] * 4, VISION_SHAPE[0] * 4))
    arr = np.asarray(g, dtype=np.float32) / 255.0
    h, w = arr.shape
    ph, pw = h // VISION_SHAPE[0], w // VISION_SHAPE[1]
    grid = arr[: ph * VISION_SHAPE[0], : pw * VISION_SHAPE[1]]
    grid = grid.reshape(VISION_SHAPE[0], ph, VISION_SHAPE[1], pw).mean(axis=(1, 3))
    return np.asarray(grid.astype(np.float32))
