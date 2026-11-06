"""Class-stratified probe draw (ADR-0019, amendment 2026-11-06).

The contiguous probe slice of the Studyforrest bridge can hold a single
label class (issue #77). This module draws a probe of ``n`` rows from the
whole cache, stratified by class, without replacement, following the
allocation rule written in the amendment:

* equal share ``n // C`` for the ``C`` classes present in the cache;
* a class with fewer rows than its share gives all its rows, and the
  shortfall is shared equally among the classes that still have rows,
  repeated until ``n`` rows are allocated or the cache is exhausted;
* an integer remainder goes to the classes in increasing label order.

Inside each class the rows are picked by a uniform permutation from a
``torch.Generator`` seeded with ``seed``; the returned indices are sorted.
"""

from __future__ import annotations

import torch


def stratified_allocation(class_counts: dict[int, int], n: int) -> dict[int, int]:
    """Rows to draw from each class (water-filling, remainder by label order)."""
    alloc = dict.fromkeys(class_counts, 0)
    remaining = min(n, sum(class_counts.values()))
    while remaining > 0:
        open_classes = sorted(c for c in class_counts if alloc[c] < class_counts[c])
        if not open_classes:
            break
        share, extra = divmod(remaining, len(open_classes))
        given = 0
        for i, c in enumerate(open_classes):
            want = share + (1 if i < extra else 0)
            take = min(want, class_counts[c] - alloc[c])
            alloc[c] += take
            given += take
        remaining -= given
        if given == 0:
            break
    return alloc


def stratified_indices(labels: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """Sorted row indices of a class-stratified draw of ``n`` rows."""
    labels = labels.long().flatten()
    classes = sorted(int(c) for c in torch.unique(labels).tolist())
    counts = {c: int((labels == c).sum()) for c in classes}
    alloc = stratified_allocation(counts, n)
    gen = torch.Generator().manual_seed(seed)
    picked = []
    for c in classes:
        rows = torch.nonzero(labels == c, as_tuple=False).flatten()
        perm = torch.randperm(rows.numel(), generator=gen)
        picked.append(rows[perm[: alloc[c]]])
    idx = torch.cat(picked) if picked else torch.empty(0, dtype=torch.long)
    return torch.sort(idx).values


def sample_probe(world, batch_size: int, seed: int, mode: str = "contiguous"):
    """Probe batch: the world's contiguous ``sample`` or a stratified draw."""
    if mode == "contiguous":
        return world.sample(batch_size=batch_size, seed=seed)
    if mode != "stratified":
        raise ValueError(f"probe sampling must be 'contiguous' or 'stratified'; got {mode!r}")
    if not (hasattr(world, "dataset_labels") and hasattr(world, "gather")):
        raise ValueError(
            "stratified probe sampling needs a world with `dataset_labels` and `gather`"
        )
    idx = stratified_indices(world.dataset_labels, batch_size, seed)
    return world.gather(idx)
