"""Pilot check 3 null test: circular-shift null for the onset cross-correlation.

Rule (PI decision 2026-10-29, fixed before any computation):

- Statistic S = the maximum cross-correlation over lags in [-200, +200] ms
  between tactile.sum(1) onsets and audio-energy onsets on the slicing
  frames, computed exactly as in scripts/pilot_actionsense_xcorr.py (same
  onset definition, same lag grid, same frame selection by segment).
- Null: for each of N = 1000 draws (seed 0, numpy default_rng), circularly
  shift the audio-onset series independently within each labelled segment
  by a uniform random offset of at least 2 s and at most (segment length -
  2 s); segments shorter than 4 s are left unshifted and counted. Compute S
  on each draw.
- Decision: GO if S_observed > the 95th percentile of the null; otherwise
  NO-GO. Report also the empirical p-value (1 + #null >= S_obs)/(N + 1), the
  null mean and 95th percentile, and the number of unshifted short segments.

Implementation notes (no freedom left to the numbers):

- Segments, onset series and the pooled cross-correlation come from the
  functions of ``pilot_actionsense_xcorr.py`` (``raw_labels``, ``onsets``,
  ``pooled_xcorr``) with the same selection (raw label starting with
  "slice", segments of at least 20 kept frames).
- Segment length L = length of the segment's onset series on its 10 Hz grid
  (frames). Offsets are integers in frames, drawn with
  ``rng.integers(20, L - 20, endpoint=True)`` (2 s = 20 frames); segments
  with L < 40 frames (4 s) are left unshifted. The shift is
  ``np.roll(y, offset)`` of the whole series, NaN (undefined onset) entries
  included; the tactile series is never shifted.
- Draws are taken in a fixed order: draw 0..N-1, and within a draw one
  offset per shifted segment in increasing segment id.
- The 95th percentile is ``np.percentile(null, 95)`` (linear interpolation).

Usage:
    uv run python scripts/pilot_actionsense_null.py --data out_a \
        --fig docs/pilot/actionsense-s00-run2-null.png --json null.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pilot_actionsense_xcorr import HZ, MAX_LAG, onsets, pooled_xcorr, raw_labels

N_DRAWS = 1000
SEED = 0
GATE_MS = 200.0
MIN_SHIFT = 20  # frames = 2 s at 10 Hz
PERCENTILE = 95.0


def slicing_pairs(data: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """(tactile onsets, audio onsets) per slicing segment, as in the xcorr helper."""
    tactile = torch.load(data / "tactile.pt").numpy().astype(np.float64)
    audio = torch.load(data / "audio.pt").numpy().astype(np.float64)
    rows = torch.load(data / "rows.pt")
    t = rows["time_s"].numpy()
    seg = rows["segment_id"].numpy()
    ridx = rows["raw_label_idx"].numpy()
    names = raw_labels(data / "MANIFEST.md")
    slicing = [i for i, s in enumerate(names) if s.strip().lower().startswith("slice")]
    tac = tactile.sum(axis=1)
    aud = audio.mean(axis=1)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for s in np.unique(seg[np.isin(ridx, slicing)]):
        sel = np.flatnonzero(seg == s)
        sel = sel[np.argsort(t[sel])]
        if len(sel) < 2 * MAX_LAG:
            continue
        x, _ = onsets(t[sel], tac[sel])
        y, _ = onsets(t[sel], aud[sel])
        if np.isnan(x).all() or np.isnan(y).all():
            continue
        pairs.append((x, y))
    return pairs


def statistic(pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """S: max pooled cross-correlation over lags within +/-200 ms."""
    lags, xc, _ = pooled_xcorr(pairs)
    inside = np.abs(lags * 1000.0 / HZ) <= GATE_MS
    return float(np.nanmax(xc[inside]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None, help="optional result file")
    args = ap.parse_args()

    pairs = slicing_pairs(args.data)
    s_obs = statistic(pairs)
    lengths = [len(y) for _, y in pairs]
    shifted = [i for i, n in enumerate(lengths) if n >= 2 * MIN_SHIFT]
    n_short = len(pairs) - len(shifted)

    rng = np.random.default_rng(SEED)
    null = np.zeros(N_DRAWS)
    for d in range(N_DRAWS):
        draw = list(pairs)
        for i in shifted:
            x, y = pairs[i]
            off = int(rng.integers(MIN_SHIFT, lengths[i] - MIN_SHIFT, endpoint=True))
            draw[i] = (x, np.roll(y, off))
        null[d] = statistic(draw)

    p95 = float(np.percentile(null, PERCENTILE))
    n_ge = int((null >= s_obs).sum())
    res = {
        "rule": "GO if S_obs > p95 of the circular-shift null, else NO-GO",
        "n_draws": N_DRAWS,
        "seed": SEED,
        "segments": len(pairs),
        "segment_lengths_frames": lengths,
        "unshifted_short_segments": n_short,
        "s_obs": round(s_obs, 6),
        "null_mean": round(float(null.mean()), 6),
        "null_p95": round(p95, 6),
        "null_max": round(float(null.max()), 6),
        "n_null_ge_obs": n_ge,
        "p_value": round((1 + n_ge) / (N_DRAWS + 1), 6),
        "decision": "GO" if s_obs > p95 else "NO-GO",
    }

    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    ax.hist(null, bins=40, color="#9db4cc", edgecolor="white")
    ax.axvline(p95, color="0.3", ls=":", lw=1.2, label=f"null p95 = {p95:.3f}")
    ax.axvline(s_obs, color="#c0504d", lw=1.8, label=f"observed S = {s_obs:.3f}")
    ax.set_xlabel("S = max onset cross-correlation over lags in [-200, +200] ms")
    ax.set_ylabel("null draws")
    ax.set_title(
        f"S00 slicing, circular-shift null (N={N_DRAWS}, p={res['p_value']:.3f}): "
        f"{res['decision']}",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(args.fig)

    text = json.dumps(res, indent=2)
    if args.json is not None:
        args.json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
