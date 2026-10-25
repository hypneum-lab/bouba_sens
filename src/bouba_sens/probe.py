"""Class-balanced multi-window probe for amendment v0.7 (section 4.2).

PI decision 2026-10-25, after the B-2 pilot: one contiguous window of
512 frames (``world.sample(512, seed = s - 99_999)``) was single-label on
3 of 5 pilot seeds, because the Studyforrest cache has only 10 label
runs. The E-2 probe is therefore built from short windows that cover
every class equally:

* Candidate windows: every label run is tiled from its first frame into
  consecutive, non-overlapping blocks of ``window`` frames; a trailing
  remainder shorter than ``window`` is not used. A block therefore lies
  wholly inside one run and carries a single label.
* Draw: one ``torch.Generator`` seeded with ``seed`` (for v0.7,
  ``s - 99_999``, the legacy probe seed). For each class c = 0, 1, ...,
  ``n_classes - 1`` in that order, one ``torch.randperm`` over the
  class's blocks (in increasing start order) is drawn and its first
  ``windows_per_class`` entries are taken, in permutation order.
* Probe rows: the frames of the chosen blocks, class 0 first, each
  block in increasing frame order. With the v0.7 constants (4 classes,
  8 windows of 16) the probe has exactly 512 rows, 128 per class, and no
  frame twice.
* A class with fewer than ``windows_per_class`` blocks raises
  ``ValueError`` (a property of the cache, checked before any seed).

The draw uses only its own generator, so it does not touch the global
torch RNG and leaves training and the Gumbel readout unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor

PROBE_WINDOW = 16
PROBE_WINDOWS_PER_CLASS = 8
PROBE_N_CLASSES = 4


def label_run_blocks(labels: Tensor, window: int = PROBE_WINDOW) -> dict[int, list[int]]:
    """Start frames of the non-overlapping ``window`` blocks tiled in each label run."""
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")
    y = labels.long().flatten().tolist()
    blocks: dict[int, list[int]] = {}
    start = 0
    for t in range(1, len(y) + 1):
        if t == len(y) or y[t] != y[start]:
            length = t - start
            for b in range(length // window):
                blocks.setdefault(int(y[start]), []).append(start + b * window)
            start = t
    return blocks


def class_balanced_probe_indices(
    labels: Tensor,
    *,
    seed: int,
    n_classes: int = PROBE_N_CLASSES,
    windows_per_class: int = PROBE_WINDOWS_PER_CLASS,
    window: int = PROBE_WINDOW,
) -> Tensor:
    """Dataset rows of the class-balanced probe (module docstring)."""
    blocks = label_run_blocks(labels, window)
    gen = torch.Generator().manual_seed(seed)
    rows: list[int] = []
    for c in range(n_classes):
        starts = sorted(blocks.get(c, []))
        if len(starts) < windows_per_class:
            raise ValueError(
                f"class {c} has {len(starts)} blocks of {window} frames; "
                f"{windows_per_class} are needed"
            )
        perm = torch.randperm(len(starts), generator=gen)[:windows_per_class]
        for i in perm.tolist():
            rows.extend(range(starts[i], starts[i] + window))
    return torch.tensor(rows, dtype=torch.long)
