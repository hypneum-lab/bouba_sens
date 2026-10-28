"""Raw-feature feasibility for the v0.9 label/world proposals (2026-10-28).

No network is trained and no pilot seed is used: a linear probe on the RAW
modality features, 10 contiguous time folds (block CV), balanced accuracy.
Answers: does a single raw modality already decode the label (ceiling), and
does the concatenation exceed the best single modality (designed headroom)?

Usage: python scripts/pilots/v09_revc_label_feasibility.py \
    <studyforrest_real dir> <ccfilm_real dir> > out.json
Run on macM1, 2026-10-28 (47 s); output reports/pilots/
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


def block_cv(x: np.ndarray, y: np.ndarray, folds: int = 10) -> float:
    n = len(y)
    edges = np.linspace(0, n, folds + 1).astype(int)
    pred = np.full(n, -1)
    for i in range(folds):
        te = np.arange(edges[i], edges[i + 1])
        buf = 16  # frames dropped on each side of the test block
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


def probe_all(x: dict[str, np.ndarray], y: np.ndarray) -> dict[str, float]:
    out = {m: block_cv(x[m], y) for m in MODS if np.var(x[m]) > 0}
    out["concat"] = block_cv(np.concatenate([x[m] for m in MODS], 1), y)
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
    # Two-channel label: a = narration-on (audio proxy, column found by
    # matching a binary indicator), b = inhalation (force resp derivative > 0).
    a_src = x["audio"]
    # the narration flag is one base feature projected linearly: recover it as
    # the sign of the leading principal component restricted to two levels
    _, _, vt = np.linalg.svd(a_src - a_src.mean(0), full_matrices=False)
    pcs = (a_src - a_src.mean(0)) @ vt[:4].T
    levels = [len(np.unique(np.round(pcs[:, k], 6))) for k in range(4)]
    res["audio_pc_levels"] = levels
    force = x["force"]
    b = (force[:, 4] > np.median(force[:, 4])).astype(int)  # resp first derivative
    two = {}
    for k in range(4):
        a = (pcs[:, k] > np.median(pcs[:, k])).astype(int)
        two[k] = a
    res["candidate_a_balance"] = {k: float(v.mean()) for k, v in two.items()}
    best_k = int(
        np.argmin(
            [
                abs(v.mean() - 0.5) if lv <= 50 else 9
                for (k, v), lv in zip(two.items(), levels, strict=True)
            ]
        )
    )
    a = two[best_k]
    y2 = 2 * a + b
    res["two_channel"] = {
        "a_pc": best_k,
        "phi_ab": float(np.corrcoef(a, b)[0, 1]),
        "counts": np.bincount(y2, minlength=4).tolist(),
        "bacc_label": probe_all(x, y2),
        "bacc_a": probe_all(x, a),
        "bacc_b": probe_all(x, b),
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
