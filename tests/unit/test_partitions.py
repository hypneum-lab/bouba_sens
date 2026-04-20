from __future__ import annotations

from bouba_sens.metrics.partitions import (
    PERCEPTIVE_PROPRIOCEPTIVE,
    generate_random_3_2_partitions,
)


def test_preregistered_partition_constant() -> None:
    assert (
        frozenset({"audio", "vision", "tactile"}),
        frozenset({"gravity", "force"}),
    ) == PERCEPTIVE_PROPRIOCEPTIVE


def test_generate_10_excludes_prereg() -> None:
    parts = generate_random_3_2_partitions(n=10, seed=0)
    assert len(parts) == 10
    pre_big, pre_small = PERCEPTIVE_PROPRIOCEPTIVE
    for big, small in parts:
        assert len(big) == 3 and len(small) == 2
        assert big | small == {"audio", "vision", "tactile", "gravity", "force"}
        assert (big, small) != (pre_big, pre_small)


def test_generate_is_deterministic_on_seed() -> None:
    a = generate_random_3_2_partitions(n=5, seed=42)
    b = generate_random_3_2_partitions(n=5, seed=42)
    assert a == b


def test_only_nine_distinct_non_prereg_partitions_exist() -> None:
    # C(5,3) = 10; minus the one pre-reg = 9 distinct alternatives
    parts = generate_random_3_2_partitions(n=20, seed=0, unique=True)
    assert len(parts) == 9
