"""Raw-feature feasibility for the v0.9 label/world proposals (2026-10-28).

No network is trained and no pilot seed is used: a linear probe on the RAW
modality features, 10 contiguous time folds (block CV), balanced accuracy.
Answers: does a single raw modality already decode the label (ceiling), and
does the concatenation exceed the best single modality (designed headroom)?

Usage: python scripts/pilots/v09_revc_label_feasibility.py \
    <studyforrest_real dir> <ccfilm_real dir> > out.json
Run on macM1, 2026-10-28; output reports/pilots/
v09_revc_label_feasibility.json. Exploratory, for the rev. C proposal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODS = ("audio", "vision", "tactile", "gravity", "force")


def load(d: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    x = {}
    for m in MODS:
        t = torch.load(d / f"{m}.pt")
        x[m] = t.reshape(t.shape[0], -1).double().numpy()
    return x, torch.load(d / "labels.pt").long().numpy()


def block_cv(x: np.ndarray, y: np.ndarray, buf: int = 16, folds: int = 10) -> float:
    n = len(y)
    edges = np.linspace(0, n, folds + 1).astype(int)
    pred = np.full(n, -1)
    for i in range(folds):
        te = np.arange(edges[i], edges[i + 1])
        tr = np.r_[0 : max(0, edges[i] - buf), min(n, edges[i + 1] + buf) : n]
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        )
        clf.fit(x[tr], y[tr])
        pred[te] = clf.predict(x[te])
    ok = pred >= 0
    return float(balanced_accuracy_score(y[ok], pred[ok]))


def probe_all(x: dict[str, np.ndarray], y: np.ndarray, buf: int = 16) -> dict[str, float]:
    """Balanced accuracy per raw modality and for the concatenation.

    ``buf`` frames are dropped on each side of every test fold.
    """
    out = {m: block_cv(x[m], y, buf) for m in MODS if np.var(x[m]) > 0}
    out["concat"] = block_cv(np.concatenate([x[m] for m in MODS], 1), y, buf)
    return out


def main() -> None:
    sf, cc = Path(sys.argv[1]), Path(sys.argv[2])
    res: dict[str, object] = {}
    x, y = load(sf)
    res["studyforrest_native"] = {
        "n": len(y),
        "counts": np.bincount(y).tolist(),
        "bacc": probe_all(x, y),
    }
    # Two-channel label, exactly as proposed (rev. C, D.2 (a)):
    # a = narration-on flag. The audio proxy is base @ proj + 0.05 noise with
    # base = [on, into, density, to_next] and proj = default_rng(1) (4, 128)
    # (scripts/fetch_studyforrest_sample.py, _build_audio_proxy); the base is
    # recovered by least squares and the flag by rounding its first column.
    proj = np.random.default_rng(1).standard_normal((4, 128)).astype(np.float32)
    base = x["audio"] @ np.linalg.pinv(proj.astype(np.float64))
    on_hat = base[:, 0]
    a = (on_hat > 0.5).astype(int)
    res["narration_flag_recovery"] = {
        "max_abs_dev_from_0_or_1": float(np.max(np.minimum(np.abs(on_hat), np.abs(on_hat - 1)))),
        "proxy_residual_rms": float(np.sqrt(np.mean((x["audio"] - base @ proj) ** 2))),
    }
    # b = inhalation: sign of the respiration first derivative (force column 4).
    b = (x["force"][:, 4] > 0).astype(int)
    y2 = 2 * a + b
    res["two_channel"] = {
        "prevalence_a": float(a.mean()),
        "prevalence_b": float(b.mean()),
        "phi_ab": float(np.corrcoef(a, b)[0, 1]),
        "counts": np.bincount(y2, minlength=4).tolist(),
        "mean_run_length_frames": {
            "a": float(len(a) / (1 + np.count_nonzero(np.diff(a)))),
            "b": float(len(b) / (1 + np.count_nonzero(np.diff(b)))),
        },
    }
    for buf in (16, 64):  # 64 frames (6.4 s) exceeds the +-5 s density window
        res["two_channel"][f"buffer_{buf}"] = {
            "bacc_label": probe_all(x, y2, buf),
            "bacc_a": probe_all(x, a, buf),
            "bacc_b": probe_all(x, b, buf),
        }
    x, y = load(cc)
    res["ccfilm_native"] = {
        "n": len(y),
        "counts": np.bincount(y).tolist(),
        "bacc": probe_all(x, y),
    }
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
