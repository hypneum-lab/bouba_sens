"""v0.9 rev. B outcome: linear-probe fusion gain on frozen representations.

PI decision 2026-10-25 (19:47), after the v0.9 pilot (zero-masked
fusion gain without headroom; ADR-0022: zero-masked queries at chance).
At the end of Phase 1 the network is frozen and read on **real** inputs:

* h_all: the fused code with the deterministic soft readout
  (``fuse(softmax_demod=True)``), flattened (896-d), reduced to 128
  dimensions by PCA fitted on the probe-training rows only;
* h_m: each modality's own representation ``core(input_proj(x_m))``
  (128-d); no channel is zeroed.

Probe-training rows (``probe_training_rows``, frozen): label-run blocks of
16 frames in the original index space (``probe.label_run_blocks``); a
block is eligible if it holds no held-out row and lies at least
``BUFFER`` (16) frames from every held-out row; per class c = 0 ... 3 in
order, one ``torch.randperm`` (generator seed ``TRAIN_SEED``) over the
eligible blocks and the first min(16, available) are kept (class 0 has
180 frames in all: it keeps what is available, no redistribution); the
classifier then weights classes by inverse frequency
(``class_weight="balanced"``).

Classifier (frozen): standardise on the probe-training rows, multinomial
logistic regression, L2, C = 1, lbfgs, 2,000 iterations,
``class_weight="balanced"``. Score: balanced accuracy on the 512
held-out rows (chance 0.25 for any class mix).

Best modality m*: chosen **on the probe-training rows only**, by 4-fold
cross-validation grouped by block (``GroupKFold``), highest mean balanced
accuracy (ties: modality order); never on held-out rows.

FG_lin = bacc_heldout(h_all) - bacc_heldout(h_m*). Positive control
(headroom gate): the concatenation of the five h_m (640-d, PCA 128 on the
probe-training rows), FG_ctrl = bacc(concat) - bacc(h_m*): a
representation that has every modality by construction.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from bouba_sens.probe import PROBE_N_CLASSES, PROBE_WINDOW, label_run_blocks
from bouba_sens.sensory import MODALITIES

BLOCKS_PER_CLASS = 16
BUFFER = 16
TRAIN_SEED = 20261025 + 9
PCA_DIMS = 128
C_REG = 1.0
MAX_ITER = 2000
CV_FOLDS = 4


def probe_training_rows(
    labels: Tensor,
    heldout: Tensor,
    *,
    blocks_per_class: int = BLOCKS_PER_CLASS,
    window: int = PROBE_WINDOW,
    buffer: int = BUFFER,
    seed: int = TRAIN_SEED,
    n_classes: int = PROBE_N_CLASSES,
) -> tuple[Tensor, Tensor, dict[int, int]]:
    """Rows, block ids and blocks per class of the probe-training set (module docstring)."""
    y = labels.long().flatten()
    held = {int(i) for i in heldout.tolist()}
    near = np.zeros(len(y), dtype=bool)
    for i in held:
        near[max(0, i - buffer) : min(len(y), i + buffer + 1)] = True
    blocks = label_run_blocks(y, window)
    gen = torch.Generator().manual_seed(seed)
    rows: list[int] = []
    ids: list[int] = []
    per_class: dict[int, int] = {}
    for c in range(n_classes):
        starts = [t for t in sorted(blocks.get(c, [])) if not near[t : t + window].any()]
        take = min(blocks_per_class, len(starts))
        per_class[c] = take
        if take == 0:
            raise ValueError(f"class {c} has no eligible probe-training block")
        for i in torch.randperm(len(starts), generator=gen)[:take].tolist():
            rows.extend(range(starts[i], starts[i] + window))
            ids.extend([starts[i]] * window)
    r = torch.tensor(rows, dtype=torch.long)
    if set(r.tolist()) & held:
        raise AssertionError("probe-training rows overlap the held-out rows")
    return r, torch.tensor(ids, dtype=torch.long), per_class


@torch.no_grad()
def representations(loop: Any, sample: Any) -> dict[str, np.ndarray]:
    """h_all (soft deterministic fused code) and h_m per modality, on real inputs."""
    out: dict[str, np.ndarray] = {}
    carriers = {m: loop.sensories[m].step(getattr(sample, m)) for m in MODALITIES}
    out["all"] = loop.nerve.fuse(carriers, softmax_demod=True).flatten(1).double().numpy()
    for m in MODALITIES:
        s = loop.sensories[m]
        out[m] = s.core(s.input_proj(getattr(sample, m))).double().numpy()
    return out


def _clf() -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C_REG, max_iter=MAX_ITER, class_weight="balanced"),
    )


def _fit_score(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray
) -> tuple[float, list[float], bool]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import balanced_accuracy_score, recall_score

    clf = _clf()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    labels = list(range(PROBE_N_CLASSES))
    recalls = recall_score(y_te, pred, labels=labels, average=None, zero_division=0)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return float(balanced_accuracy_score(y_te, pred)), [float(r) for r in recalls], converged


def _pca(x_tr: np.ndarray, x_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA

    n = min(PCA_DIMS, x_tr.shape[0] - 1, x_tr.shape[1])
    pca = PCA(n_components=n, svd_solver="full").fit(x_tr)
    return pca.transform(x_tr), pca.transform(x_te)


def _cv_bacc(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    from sklearn.model_selection import GroupKFold

    scores = []
    for tr, te in GroupKFold(n_splits=CV_FOLDS).split(x, y, groups):
        scores.append(_fit_score(x[tr], y[tr], x[te], y[te])[0])
    return float(np.mean(scores))


@dataclass(frozen=True)
class LinearFG:
    fg: float
    bacc_all: float
    best_modality: str
    bacc_best: float
    bacc_by_modality: dict[str, float]
    cv_bacc_by_modality: dict[str, float]
    fg_control: float
    bacc_control: float
    recalls_all: list[float]
    converged: bool
    per_class_training_blocks: dict[int, int]


def fg_from_representations(
    r_tr: dict[str, np.ndarray],
    y_tr: np.ndarray,
    groups: np.ndarray,
    r_te: dict[str, np.ndarray],
    y_te: np.ndarray,
    per_class: dict[int, int] | None = None,
) -> LinearFG:
    """FG_lin and the positive control from probe-training / held-out representations."""
    a_tr, a_te = _pca(r_tr["all"], r_te["all"])
    bacc_all, rec_all, conv = _fit_score(a_tr, y_tr, a_te, y_te)
    cv: dict[str, float] = {m: _cv_bacc(r_tr[m], y_tr, groups) for m in MODALITIES}
    best = max(MODALITIES, key=lambda m: (cv[m], -MODALITIES.index(m)))
    held: dict[str, float] = {}
    for m in MODALITIES:
        b, _, c = _fit_score(r_tr[m], y_tr, r_te[m], y_te)
        held[m] = b
        conv = conv and c
    c_tr, c_te = _pca(
        np.concatenate([r_tr[m] for m in MODALITIES], 1),
        np.concatenate([r_te[m] for m in MODALITIES], 1),
    )
    bacc_ctrl, _, c = _fit_score(c_tr, y_tr, c_te, y_te)
    return LinearFG(
        fg=bacc_all - held[best],
        bacc_all=bacc_all,
        best_modality=best,
        bacc_best=held[best],
        bacc_by_modality=held,
        cv_bacc_by_modality=cv,
        fg_control=bacc_ctrl - held[best],
        bacc_control=bacc_ctrl,
        recalls_all=rec_all,
        converged=conv and c,
        per_class_training_blocks=dict(per_class or {}),
    )


def linear_probe_fg(loop: Any, world: Any) -> LinearFG:
    """FG_lin and the positive control for one frozen network (module docstring)."""
    heldout_sample, heldout = world.probe_sample_with_indices(world.n_probe, 0)
    # dataset_labels marks held-out rows -1, so no label run crosses them.
    train_rows, block_ids, per_class = probe_training_rows(world.dataset_labels, heldout)
    train_sample = world.gather(train_rows)
    return fg_from_representations(
        representations(loop, train_sample),
        train_sample.label.numpy(),
        block_ids.numpy(),
        representations(loop, heldout_sample),
        heldout_sample.label.numpy(),
        per_class,
    )
