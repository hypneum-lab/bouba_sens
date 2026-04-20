"""Modality-partition helpers for B-3 null-model control (Sprint 7 Task 7.1).

The pre-registered perceptive / proprioceptive split is the only
partition used in v0.1 - v0.3. Sprint 7 adds random 3+2 partitions to
check whether Me6 passes because of the partition or because of the
labelling.
"""

from __future__ import annotations

import itertools
import random

__all__ = ["PERCEPTIVE_PROPRIOCEPTIVE", "generate_random_3_2_partitions"]

MODALITIES: frozenset[str] = frozenset({"audio", "vision", "tactile", "gravity", "force"})

PERCEPTIVE_PROPRIOCEPTIVE: tuple[frozenset[str], frozenset[str]] = (
    frozenset({"audio", "vision", "tactile"}),
    frozenset({"gravity", "force"}),
)


def _all_3_2_partitions() -> list[tuple[frozenset[str], frozenset[str]]]:
    out: list[tuple[frozenset[str], frozenset[str]]] = []
    for big in itertools.combinations(sorted(MODALITIES), 3):
        big_f = frozenset(big)
        small_f = frozenset(MODALITIES - big_f)
        out.append((big_f, small_f))
    return out


def generate_random_3_2_partitions(
    *, n: int, seed: int, unique: bool = False
) -> list[tuple[frozenset[str], frozenset[str]]]:
    """Return ``n`` random 3+2 partitions of the 5 modalities.

    Always excludes the pre-registered ``PERCEPTIVE_PROPRIOCEPTIVE``.
    If ``unique`` is True the result is deduplicated and capped at the
    9 distinct alternatives that exist.
    """
    all_parts = [p for p in _all_3_2_partitions() if p != PERCEPTIVE_PROPRIOCEPTIVE]
    rng = random.Random(seed)
    if unique:
        rng.shuffle(all_parts)
        return all_parts[: min(n, len(all_parts))]
    return [rng.choice(all_parts) for _ in range(n)]
