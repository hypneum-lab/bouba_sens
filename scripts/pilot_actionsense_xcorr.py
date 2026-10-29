"""Pilot check 3: tactile-pressure onsets vs audio-energy onsets (slicing).

Spec docs/superpowers/specs/2026-10-28-actionsense-world-design.md, section 7,
check 3. Reads an extraction cache (``tactile.pt``, ``audio.pt``, ``rows.pt``,
``MANIFEST.md``) and, on the frames whose raw label starts with "slice":

- tactile signal = ``tactile.sum(1)`` (sum of the 32 z-scored taxel blocks);
- audio energy = mean over the 128 z-scored log-mel bands;
- onset = positive part of the first difference between two consecutive
  10 Hz frames of the same segment (a dropped frame breaks the pair);
- each onset series is standardised per segment, then the cross-correlation
  is pooled over all slicing segments for lags of -1 s .. +1 s.

Positive lag = audio onsets FOLLOW tactile onsets. The frame rate is 10 Hz,
so the lag resolution is 100 ms. Gate: |peak lag| <= 200 ms.

Usage:
    uv run python scripts/pilot_actionsense_xcorr.py --data out_a \
        --fig docs/pilot/actionsense-s00-xcorr.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HZ = 10.0
MAX_LAG = 10  # frames, = 1 s
GATE_MS = 200.0


def raw_labels(manifest: Path) -> list[str]:
    """The ``raw_label_idx`` -> label list of MANIFEST.md."""
    text = manifest.read_text()
    sec = text.split("## Raw label index", 1)[1].split("\n## ", 1)[0]
    return [m.group(1) for m in re.finditer(r"^\d+\. (.+)$", sec, flags=re.M)]


def onsets(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Onset series on the segment's 10 Hz grid; NaN where undefined."""
    k = np.rint((t - t[0]) * HZ).astype(np.int64)
    grid = np.full(int(k.max()) + 1, np.nan)
    grid[k] = x
    d = np.diff(grid)
    on = np.where(np.isnan(d), np.nan, np.maximum(d, 0.0))
    ok = ~np.isnan(on)
    if ok.sum() > 1 and np.nanstd(on) > 0:
        on[ok] = (on[ok] - on[ok].mean()) / on[ok].std()
    else:
        on[:] = np.nan
    return on, ok


def pooled_xcorr(
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled mean product x[k] * y[k + lag] over valid pairs, and pair counts."""
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    num = np.zeros(len(lags))
    cnt = np.zeros(len(lags))
    for x, y in pairs:
        n = len(x)
        for i, lag in enumerate(int(v) for v in lags):
            if abs(lag) >= n:
                continue
            a = x[max(0, -lag) : n - max(0, lag)]
            b = y[max(0, lag) : n - max(0, -lag)]
            ok = ~np.isnan(a) & ~np.isnan(b)
            num[i] += float((a[ok] * b[ok]).sum())
            cnt[i] += int(ok.sum())
    xc = np.divide(num, cnt, out=np.full(len(lags), np.nan), where=cnt > 0)
    return lags, xc, cnt.astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None, help="optional result file")
    args = ap.parse_args()

    tactile = torch.load(args.data / "tactile.pt").numpy().astype(np.float64)
    audio = torch.load(args.data / "audio.pt").numpy().astype(np.float64)
    rows = torch.load(args.data / "rows.pt")
    t = rows["time_s"].numpy()
    seg = rows["segment_id"].numpy()
    ridx = rows["raw_label_idx"].numpy()
    names = raw_labels(args.data / "MANIFEST.md")
    slicing = [i for i, s in enumerate(names) if s.strip().lower().startswith("slice")]

    tac = tactile.sum(axis=1)
    aud = audio.mean(axis=1)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    n_frames = 0
    seg_ids = np.unique(seg[np.isin(ridx, slicing)])
    for s in seg_ids:
        sel = np.flatnonzero(seg == s)
        sel = sel[np.argsort(t[sel])]
        if len(sel) < 2 * MAX_LAG:
            continue
        x, _ = onsets(t[sel], tac[sel])
        y, _ = onsets(t[sel], aud[sel])
        if np.isnan(x).all() or np.isnan(y).all():
            continue
        pairs.append((x, y))
        n_frames += len(sel)

    lags, xc, cnt = pooled_xcorr(pairs)
    n_onsets = [int((~np.isnan(x)).sum()) for x, _ in pairs]
    i_peak = int(np.nanargmax(xc))
    peak_ms = float(lags[i_peak] * 1000.0 / HZ)
    outside = np.abs(lags * 1000.0 / HZ) > GATE_MS
    res = {
        "slicing_labels": [names[i] for i in slicing],
        "segments": len(pairs),
        "frames": n_frames,
        "lags_ms": (lags * 1000.0 / HZ).tolist(),
        "xcorr": [round(float(v), 6) for v in xc],
        "pairs_per_lag": cnt.tolist(),
        "onset_samples_per_segment": n_onsets,
        "peak_lag_ms": peak_ms,
        "peak_value": round(float(xc[i_peak]), 6),
        "max_outside_gate": round(float(np.nanmax(xc[outside])), 6),
        "pass": abs(peak_ms) <= GATE_MS,
    }

    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    ax.axvspan(-GATE_MS, GATE_MS, color="0.9", label="gate ±200 ms")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.plot(lags * 1000.0 / HZ, xc, marker="o", color="#1f4e79", lw=1.5)
    ax.axvline(peak_ms, color="#c0504d", ls="--", lw=1.0, label=f"peak {peak_ms:+.0f} ms")
    for lag, v, c in zip(lags * 1000.0 / HZ, xc, cnt, strict=True):
        if c > 0:
            ax.annotate(
                f"n={c}",
                (lag, v),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=6,
                color="0.35",
            )
    ax.set_xlabel("lag (ms), positive = audio onset after tactile onset")
    ax.set_ylabel("pooled onset cross-correlation")
    ax.set_title(f"S00 slicing: {len(pairs)} segments, {n_frames} frames at 10 Hz")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(args.fig)

    text = json.dumps(res, indent=2)
    if args.json is not None:
        args.json.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
