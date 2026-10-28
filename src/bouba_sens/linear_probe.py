"""Linear read-out probe on frozen per-modality representations (ADR-0023).

The B-3 probe v1 (`AdaptationLoop.query_successes`) zeroes the four
non-query modalities and reads the network's own head: the network is fed
inputs it never saw in training and its single-modality accuracy sits at
chance (ADR-0022 §Verdict). The v2 probe instead asks what is linearly
decodable from ONE modality's frozen representation, computed by that
modality's own sensory pathway on a clean, in-distribution input — no
other modality is involved, so nothing is pushed out of distribution.

This module holds the model-agnostic part: a multinomial logistic
regression (scikit-learn, already a project dependency) on standardised
features, with the L2 strength chosen by K-fold cross-validation on the
training pool only, then refit on the whole pool and scored item by item
on a held-out test set. Everything is deterministic (lbfgs, fixed fold
seed), and nothing touches the global torch / numpy RNGs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Inverse L2 strengths scanned by cross-validation. Fixed a priori (ADR-0023).
DEFAULT_CS: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
DEFAULT_FOLDS = 5
# Fold shuffling seed: fixed so that a probe is a pure function of its data.
FOLD_SEED = 0
MAX_ITER = 2000


@dataclass(frozen=True)
class ProbeFit:
    """Outcome of one linear probe.

    `successes[i]` is 1 when held-out item `i` is classified correctly.
    `baseline` is the accuracy of predicting the training-pool majority
    class on the held-out items — what an uninformative representation
    yields (the probe's own chance level, which is not 1 / n_classes on
    worlds with unbalanced labels).
    """

    successes: list[int]
    accuracy: float
    cv_accuracy: float
    chosen_c: float
    baseline: float
    n_train: int
    n_test: int


def _make_classifier(c: float) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=MAX_ITER),
    )


def _fit_predict(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, c: float) -> np.ndarray:
    """Fit on (x_tr, y_tr) and predict x_te; a single-class pool predicts that class."""
    classes = np.unique(y_tr)
    if classes.size < 2:
        return np.full(x_te.shape[0], classes[0], dtype=y_tr.dtype)
    clf = _make_classifier(c)
    clf.fit(x_tr, y_tr)  # type: ignore[attr-defined]
    return np.asarray(clf.predict(x_te))  # type: ignore[attr-defined]


def fit_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    cs: Sequence[float] = DEFAULT_CS,
    folds: int = DEFAULT_FOLDS,
) -> ProbeFit:
    """Cross-validated logistic-regression probe, scored on a held-out set.

    1. For each C in `cs`, mean accuracy over `folds` shuffled K-fold splits
       of the TRAINING pool (fold seed `FOLD_SEED`).
    2. Pick the C with the highest CV accuracy (ties -> the smallest C,
       i.e. the most regularised).
    3. Refit on the whole training pool, predict the held-out test items.

    The test set never enters steps 1-2.
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[1] != x_test.shape[1]:
        raise ValueError(
            f"expected 2-D features of equal width; got {x_train.shape}, {x_test.shape}"
        )
    if x_train.shape[0] != y_train.shape[0] or x_test.shape[0] != y_test.shape[0]:
        raise ValueError("features and labels must have the same number of rows")
    if x_train.shape[0] < folds:
        raise ValueError(f"need at least {folds} training items; got {x_train.shape[0]}")
    if not cs:
        raise ValueError("cs must be non-empty")

    splitter = KFold(n_splits=folds, shuffle=True, random_state=FOLD_SEED)
    splits = list(splitter.split(x_train))
    best_c, best_cv = float(cs[0]), -1.0
    for c in sorted(float(v) for v in cs):
        scores = [
            float((_fit_predict(x_train[tr], y_train[tr], x_train[va], c) == y_train[va]).mean())
            for tr, va in splits
        ]
        cv = float(np.mean(scores))
        if cv > best_cv + 1e-12:
            best_c, best_cv = c, cv

    preds = _fit_predict(x_train, y_train, x_test, best_c)
    successes = [int(v) for v in (preds == y_test)]
    values, counts = np.unique(y_train, return_counts=True)
    majority = values[int(np.argmax(counts))]
    return ProbeFit(
        successes=successes,
        accuracy=float(np.mean(successes)),
        cv_accuracy=best_cv,
        chosen_c=best_c,
        baseline=float((y_test == majority).mean()),
        n_train=int(x_train.shape[0]),
        n_test=int(x_test.shape[0]),
    )
