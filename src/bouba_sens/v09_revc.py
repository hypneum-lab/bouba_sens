"""v0.9 rev. C: two-channel labels, purged segment split, gates (PI decision 2026-10-28).

PI decision 2026-10-28 (about 19:05): CC-film is the PRIMARY world (cache
re-extracted at 10 fps), the Studyforrest two-channel label the SECONDARY
(replication); straight-through emission with the learning gate (P1).

**Labels** (y = 2a + b, 4 classes; ``two_channel_labels``):

* Studyforrest: a = narration-on flag, recovered from the audio proxy
  (``[on, into, density, to_next] @ proj + 0.05 noise``, ``proj`` from
  ``numpy.random.default_rng(1)``, ``scripts/fetch_studyforrest_sample.py``)
  by least squares and rounding; b = sign of force column 4 (respiration
  first derivative) > 0.
* CC-film: a = audio energy high (mean over the 128 standardised log-mel
  bands above its median); b = vision brightness high (mean of the 16 x 16
  intensity grid above its median).

**Split** (``SegmentSplitWorld``, frozen): the frames are cut into
``N_SEGMENTS`` = 30 contiguous segments (``numpy.array_split``);
``N_HELD`` = 5 are held out, chosen among ``SPLIT_TRIES`` = 2,000 random
5-subsets (generator ``SPLIT_SEED``) as the first that maximises the
smallest per-class held-out share. (The proposal's "3 segments, every class
>= 15 % held out" is infeasible: 3 / 30 of the frames cannot give every
class 15 %; corrected to 5 segments and reported.) ``PURGE`` = 64 frames
on each side of every held-out segment are excluded from training batches,
donor pools, check anchors and probe training. Held-out rows = the probe.

**Outcome** (``linear_probe_fg_revc``): rev. B's linear-probe fusion gain
with three changes: probe-training rows are every ``PROBE_STRIDE``-th
usable (non-held, non-purged) row; the best modality is chosen by grouped
4-fold CV over the remaining segments with a ``PURGE``-frame purge around
each test fold; the positive control is a logistic on the standardised
640-dim concatenation **without PCA**. h_all keeps PCA 128 (rev. B).
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from bouba_sens.sensory import MODALITIES
from bouba_sens.world.base import WorldSample

N_SEGMENTS = 30
N_HELD = 5
PURGE = 64
SPLIT_SEED = 20261028 + 9
SPLIT_TRIES = 2000
PROBE_STRIDE = 4
CV_FOLDS = 4
EMISSION = "straight-through"
PILOT_SEEDS: tuple[int, ...] = (3206, 3207, 3208, 3209, 3210)
ONSETS: tuple[int, ...] = (0, 50, 100, 150, 200)
ARMS: tuple[str, ...] = ("decor", "joint", "plain", "label_break")
BREAK_MODALITY = {"ccfilm": "vision", "studyforrest": "force"}  # carries bit b
# gates (proposal D.2 (a), adopted for both worlds)
ISOLATION_MAX = 0.65
CONCAT_OVER_BEST = 0.20
CONTROL_MIN = 0.10
BREAK_DROP = 0.10
PLAIN_BACC_MIN = 0.35
ZERO_D_MAX = 0.5

_FILES = ("audio", "vision", "tactile", "gravity", "force")


# --- labels ----------------------------------------------------------------------


def narration_flag(audio: np.ndarray) -> tuple[np.ndarray, float]:
    """Narration-on flag of the Studyforrest audio proxy, and the max rounding error."""
    proj = np.random.default_rng(1).standard_normal((4, audio.shape[1])).astype(np.float32)
    base = audio.astype(np.float64) @ np.linalg.pinv(proj.astype(np.float64))
    on = base[:, 0]
    dev = float(np.max(np.minimum(np.abs(on), np.abs(on - 1.0))))
    return (on > 0.5).astype(np.int64), dev


def two_channel_labels(source: str, x: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    """y = 2a + b and its provenance (module docstring)."""
    if source == "studyforrest":
        a, dev = narration_flag(x["audio"])
        if dev > 0.1:
            raise ValueError(f"narration flag not recoverable (max deviation {dev:.3f})")
        b = (x["force"][:, 4] > 0).astype(np.int64)
        info: dict[str, Any] = {
            "a": "narration-on flag (audio proxy, least squares)",
            "a_max_rounding_dev": dev,
            "b": "force column 4 (respiration first derivative) > 0",
        }
    elif source == "ccfilm":
        energy = x["audio"].reshape(len(x["audio"]), -1).mean(1)
        bright = x["vision"].reshape(len(x["vision"]), -1).mean(1)
        a = (energy > np.median(energy)).astype(np.int64)
        b = (bright > np.median(bright)).astype(np.int64)
        info = {
            "a": "audio energy (mean standardised log-mel) > median",
            "b": "vision brightness (mean 16x16 intensity) > median",
        }
    else:
        raise ValueError(f"unknown source {source!r}")
    y = 2 * a + b
    info.update(
        {
            "prevalence_a": float(a.mean()),
            "prevalence_b": float(b.mean()),
            "phi_ab": float(np.corrcoef(a, b)[0, 1]),
            "counts": np.bincount(y, minlength=4).tolist(),
        }
    )
    return y, info


# --- split -----------------------------------------------------------------------


def choose_heldout_segments(
    labels: np.ndarray, *, n_segments: int = N_SEGMENTS, n_held: int = N_HELD
) -> tuple[list[np.ndarray], list[int], float]:
    """Segments, the held-out segment ids and the smallest per-class held-out share."""
    segs = np.array_split(np.arange(len(labels)), n_segments)
    counts = np.array([np.bincount(labels[s], minlength=4) for s in segs])
    total = counts.sum(0)
    if (total == 0).any():
        raise ValueError("a class is empty")
    rng = np.random.default_rng(SPLIT_SEED)
    best: tuple[float, list[int]] | None = None
    for _ in range(SPLIT_TRIES):
        pick = sorted(rng.choice(n_segments, size=n_held, replace=False).tolist())
        share = float((counts[pick].sum(0) / total).min())
        if best is None or share > best[0]:
            best = (share, pick)
    assert best is not None
    return segs, best[1], best[0]


class SegmentSplitWorld:
    """Two-channel-label world with the frozen purged segment split (module docstring)."""

    def __init__(self, source: str, data_dir: Path) -> None:
        d = Path(data_dir)
        self.source = source
        t = {m: torch.load(d / f"{m}.pt") for m in _FILES}
        if t["vision"].ndim == 2:
            t["vision"] = t["vision"].reshape(t["vision"].shape[0], 16, 16)
        self._t = t
        raw = {m: t[m].reshape(t[m].shape[0], -1).double().numpy() for m in _FILES}
        y, self.label_info = two_channel_labels(source, raw)
        self._labels = torch.from_numpy(y).long()
        n = len(y)
        self._n = n
        segs, held_ids, share = choose_heldout_segments(y)
        self.segment_of = np.empty(n, dtype=np.int64)
        for i, s in enumerate(segs):
            self.segment_of[s] = i
        held = np.concatenate([segs[i] for i in held_ids])
        held_mask = np.zeros(n, dtype=bool)
        held_mask[held] = True
        purge = np.zeros(n, dtype=bool)
        for i in held_ids:
            lo, hi = int(segs[i][0]), int(segs[i][-1]) + 1
            purge[max(0, lo - PURGE) : lo] = True
            purge[hi : min(n, hi + PURGE)] = True
        purge &= ~held_mask
        self.held_segments = held_ids
        self.min_class_heldout_share = share
        self._heldout = torch.from_numpy(np.sort(held)).long()
        self._held_mask = torch.from_numpy(held_mask)
        self._excluded = torch.from_numpy(held_mask | purge)
        self.n_probe = len(held)
        self.n_purged = int(purge.sum())
        self.n_train = n - self.n_probe
        self._valid_starts: dict[int, Tensor] = {}

    stratified = True

    @property
    def mode(self) -> str:
        return f"segment_split_{self.source}"

    @property
    def usable_mask(self) -> np.ndarray:
        """Rows usable for training / donors / probe training (not held, not purged)."""
        return (~self._excluded).numpy()

    def _training_starts(self, batch_size: int) -> Tensor:
        if batch_size not in self._valid_starts:
            bad = torch.cat([torch.zeros(1), self._excluded.double().cumsum(0)])
            t = torch.arange(0, self._n - batch_size + 1)
            starts = t[(bad[t + batch_size] - bad[t]) == 0]
            if len(starts) == 0:
                raise ValueError(f"no training window of {batch_size} frames")
            self._valid_starts[batch_size] = starts
        return self._valid_starts[batch_size]

    def sample(self, batch_size: int, seed: int) -> WorldSample:
        return self.sample_with_indices(batch_size, seed)[0]

    def sample_with_indices(self, batch_size: int, seed: int) -> tuple[WorldSample, Tensor]:
        gen = torch.Generator().manual_seed(seed)
        starts = self._training_starts(batch_size)
        j = int(torch.randint(0, len(starts), (1,), generator=gen).item())
        idx = torch.arange(int(starts[j]), int(starts[j]) + batch_size)
        return self.gather(idx), idx

    @property
    def dataset_labels(self) -> Tensor:
        """Labels with held-out and purged rows set to -1 (never donors)."""
        masked = self._labels.clone()
        masked[self._excluded] = -1
        return masked

    def probe_sample_with_indices(self, batch_size: int, seed: int) -> tuple[WorldSample, Tensor]:
        if batch_size != self.n_probe:
            raise ValueError(f"the held-out probe has {self.n_probe} rows; got {batch_size}")
        return self.gather(self._heldout), self._heldout

    def gather(self, idx: Tensor) -> WorldSample:
        return WorldSample(
            z=torch.zeros(len(idx), 1),
            audio=self._t["audio"][idx],
            vision=self._t["vision"][idx],
            tactile=self._t["tactile"][idx],
            gravity=self._t["gravity"][idx],
            force=self._t["force"][idx],
            label=self._labels[idx],
        )

    def modality_dims(self) -> dict[str, tuple[int, ...]]:
        return {m: tuple(self._t[m].shape[1:]) for m in _FILES}

    def describe(self) -> dict[str, Any]:
        y = self._labels.numpy()
        held = self._held_mask.numpy()
        use = self.usable_mask
        return {
            "source": self.source,
            "n_frames": self._n,
            "n_heldout": self.n_probe,
            "n_purged": self.n_purged,
            "held_segments": self.held_segments,
            "min_class_heldout_share": self.min_class_heldout_share,
            "heldout_counts": np.bincount(y[held], minlength=4).tolist(),
            "usable_counts": np.bincount(y[use], minlength=4).tolist(),
            "label": self.label_info,
        }


# --- label-breaking arm ------------------------------------------------------------


class LabelBreakSchedule:
    """Inside the window, modality ``modality`` comes from a uniform usable row.

    The positive treatment control of rev. C: the donor of the modality
    that carries bit b is drawn regardless of the label (b destroyed); the
    other modalities stay the anchor's. Donor generator: the namespaced
    donor stream of the seed (as the registered arms).
    """

    arm = "label_break"

    def __init__(
        self,
        window: Any,
        *,
        seed: int,
        world: SegmentSplitWorld,
        modality: str,
        seed_namespace: int,
    ) -> None:
        self.window = window
        self.seed = seed
        self.modality = modality
        self.seed_namespace = seed_namespace
        self.pool = torch.from_numpy(np.flatnonzero(world.usable_mask)).long()
        self.n_self_donor = 0

    def sample(self, world: Any, *, batch_size: int, seed: int, step: int) -> WorldSample:
        if not self.window.active(step):
            return world.sample(batch_size, seed)  # type: ignore[no-any-return]
        from bouba_sens.seeds import STREAMS, ns_seed

        base, _ = world.sample_with_indices(batch_size, seed)
        gen = torch.Generator().manual_seed(
            ns_seed(self.seed_namespace, self.seed, STREAMS["donor"], step)
        )
        rows = self.pool[torch.randint(0, len(self.pool), (batch_size,), generator=gen)]
        out: WorldSample = replace(
            base, **{self.modality: getattr(world.gather(rows), self.modality)}
        )
        return out


# --- outcome -------------------------------------------------------------------------


def _purged_group_folds(
    rows: np.ndarray, groups: np.ndarray, n_folds: int = CV_FOLDS
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped folds over segments; training rows within PURGE of a test row are dropped."""
    uniq = np.unique(groups)
    folds = np.array_split(uniq, n_folds)
    out = []
    for f in folds:
        te = np.isin(groups, f)
        # a fold's segments need not be adjacent: purge around every test row
        test_set = np.zeros(int(rows.max()) + 1, dtype=bool)
        test_set[rows[te]] = True
        csum = np.concatenate([[0], np.cumsum(test_set)])
        lo = np.clip(rows - PURGE, 0, len(test_set))
        hi = np.clip(rows + PURGE + 1, 0, len(test_set))
        near = (csum[hi] - csum[lo]) > 0
        tr = ~te & ~near
        out.append((np.flatnonzero(tr), np.flatnonzero(te)))
    return out


def _fit_score(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray
) -> tuple[float, list[float], bool]:
    from bouba_sens.fusion_probe import _fit_score as fs

    return fs(x_tr, y_tr, x_te, y_te)


@dataclass(frozen=True)
class LinearFGRevC:
    fg: float
    bacc_all: float
    best_modality: str
    bacc_best: float
    bacc_by_modality: dict[str, float]
    cv_bacc_by_modality: dict[str, float]
    fg_control: float
    bacc_control: float
    recalls_all: list[float]
    recalls_control: list[float]
    converged: bool
    n_train_rows: int
    n_heldout_rows: int


def linear_probe_fg_revc(loop: Any, world: SegmentSplitWorld) -> LinearFGRevC:
    from bouba_sens.fusion_probe import _pca, representations

    held_sample, _ = world.probe_sample_with_indices(world.n_probe, 0)
    rows = np.flatnonzero(world.usable_mask)[::PROBE_STRIDE]
    tr_sample = world.gather(torch.from_numpy(rows).long())
    r_tr = representations(loop, tr_sample)
    r_te = representations(loop, held_sample)
    y_tr, y_te = tr_sample.label.numpy(), held_sample.label.numpy()
    groups = world.segment_of[rows]
    folds = _purged_group_folds(rows, groups)
    a_tr, a_te = _pca(r_tr["all"], r_te["all"])
    bacc_all, rec_all, conv = _fit_score(a_tr, y_tr, a_te, y_te)
    cv: dict[str, float] = {}
    held: dict[str, float] = {}
    for m in MODALITIES:
        scores = []
        for tr, te in folds:
            scores.append(_fit_score(r_tr[m][tr], y_tr[tr], r_tr[m][te], y_tr[te])[0])
        cv[m] = float(np.mean(scores))
        b, _, c = _fit_score(r_tr[m], y_tr, r_te[m], y_te)
        held[m] = b
        conv = conv and c
    best = max(MODALITIES, key=lambda m: (cv[m], -MODALITIES.index(m)))
    c_tr = np.concatenate([r_tr[m] for m in MODALITIES], 1)
    c_te = np.concatenate([r_te[m] for m in MODALITIES], 1)
    bacc_ctrl, rec_ctrl, c = _fit_score(c_tr, y_tr, c_te, y_te)
    return LinearFGRevC(
        fg=bacc_all - held[best],
        bacc_all=bacc_all,
        best_modality=best,
        bacc_best=held[best],
        bacc_by_modality=held,
        cv_bacc_by_modality=cv,
        fg_control=bacc_ctrl - held[best],
        bacc_control=bacc_ctrl,
        recalls_all=rec_all,
        recalls_control=rec_ctrl,
        converged=conv and c,
        n_train_rows=len(rows),
        n_heldout_rows=len(y_te),
    )


# --- one cell --------------------------------------------------------------------------


def run_cell_revc(
    world: SegmentSplitWorld,
    seed: int,
    onset: int,
    arm: str,
    *,
    seed_namespace: int,
    width: int = 50,
    recovery: int = 200,
    batch: int = 16,
) -> dict[str, Any]:
    """One Phase-1 run of (seed, onset, arm): outcome and learning-gate metrics."""
    from bouba_sens.builders import build_loop
    from bouba_sens.decorrelation import DecorrelationSchedule, DecorrelationWindow
    from bouba_sens.learning_gate import heldout_task_metrics
    from bouba_sens.v09_driver import check_learning_rate

    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("call torch.use_deterministic_algorithms(True) first")
    t0 = time.monotonic()
    torch.manual_seed(seed)
    loop = build_loop(
        world, seed, lock_after=None, seed_namespace=seed_namespace, emission=EMISSION
    )
    check_learning_rate(loop)
    window = DecorrelationWindow(onset, width)
    sched: Any
    if arm == "label_break":
        sched = LabelBreakSchedule(
            window,
            seed=seed,
            world=world,
            modality=BREAK_MODALITY[world.source],
            seed_namespace=seed_namespace,
        )
    else:
        sched = DecorrelationSchedule(
            window,
            arm,
            seed=seed,
            world=world,
            seed_namespace=seed_namespace,
        )
    steps = onset + width + recovery
    loop.pretrain(steps, batch_size=batch, seed=seed, decorrelation=sched)
    check_learning_rate(loop)
    held, _ = world.probe_sample_with_indices(world.n_probe, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lin = linear_probe_fg_revc(loop, world)
    return {
        "seed": seed,
        "onset": onset,
        "arm": arm,
        "steps": steps,
        "n_self_donor": sched.n_self_donor,
        "task": heldout_task_metrics(loop, held),
        "lin": asdict(lin),
        "wall_s": time.monotonic() - t0,
    }


# --- summary ---------------------------------------------------------------------------


def summarise(cells: Sequence[dict[str, Any]], seeds: Sequence[int]) -> dict[str, Any]:
    """Gates 1-5 and the D matrix of the pilot grid (proposal D.2 (a))."""
    from bouba_sens.learning_gate import learning_gate
    from bouba_sens.v09_driver import variance_components

    by = {(c["seed"], c["onset"], c["arm"]): c for c in cells}
    plain = [c for c in cells if c["arm"] == "plain"]
    gate1 = learning_gate(
        {s: [c["task"] for c in plain if c["seed"] == s] for s in seeds}, need_over_single=True
    )
    per_mod = {
        m: float(np.mean([c["lin"]["bacc_by_modality"][m] for c in plain])) for m in MODALITIES
    }
    concat_over = float(
        np.mean(
            [c["lin"]["bacc_control"] - max(c["lin"]["bacc_by_modality"].values()) for c in plain]
        )
    )
    fg_ctrl = float(np.mean([c["lin"]["fg_control"] for c in plain]))
    drops = [
        by[(s, o, "joint")]["lin"]["bacc_all"] - by[(s, o, "label_break")]["lin"]["bacc_all"]
        for s in seeds
        for o in ONSETS
        if (s, o, "label_break") in by
    ]
    d = np.array(
        [
            [by[(s, o, "joint")]["lin"]["fg"] - by[(s, o, "decor")]["lin"]["fg"] for o in ONSETS]
            for s in seeds
        ]
    )
    plain_bacc = float(np.mean([c["lin"]["bacc_all"] for c in plain]))
    zero_d = float(np.mean(d == 0.0))
    gates = {
        "gate1_learning": gate1["passed"],
        "gate2_isolation": max(per_mod.values()) <= ISOLATION_MAX
        and concat_over >= CONCAT_OVER_BEST,
        "gate3_positive_control": fg_ctrl >= CONTROL_MIN,
        "gate4_label_break": bool(drops) and float(np.mean(drops)) >= BREAK_DROP,
        "gate5_headroom": plain_bacc >= PLAIN_BACC_MIN and zero_d < ZERO_D_MAX,
    }
    return {
        "learning_gate": gate1,
        "plain_bacc_by_modality_mean": per_mod,
        "plain_concat_over_best_single_mean": concat_over,
        "plain_fg_control_mean": fg_ctrl,
        "label_break_drop_mean": float(np.mean(drops)) if drops else None,
        "label_break_drops": drops,
        "plain_bacc_all_mean": plain_bacc,
        "D_zero_fraction": zero_d,
        "onsets": list(ONSETS),
        "D": d.tolist(),
        "variance_components": variance_components(d),
        "all_converged": all(c["lin"]["converged"] for c in cells),
        "gates": gates,
    }
