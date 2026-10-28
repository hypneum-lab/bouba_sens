"""Per-campaign seed namespaces: no derived seed is shared between pools (PI 2026-10-25)."""

from __future__ import annotations

import itertools

import pytest
import torch

from bouba_sens.readout import GUMBEL_DRAWS, gumbel_draw_seed
from bouba_sens.seeds import (
    BY_NAME,
    CAMPAIGNS,
    STREAMS,
    campaign_code,
    derived_seeds,
    ns_seed,
    resolve_campaign,
)
from bouba_sens.world.gaussian import GaussianWorld


def test_ns_seed_is_deterministic_63_bit_and_key_sensitive() -> None:
    a = ns_seed(8101, 2201, STREAMS["phase1_batch"], 0)
    assert a == ns_seed(8101, 2201, 0, 0) and 0 <= a < 2**63
    keys = [(8101, 2201, 0, 0), (8102, 2201, 0, 0), (8101, 2202, 0, 0), (8101, 2201, 1, 0),
            (8101, 2201, 0, 1)]  # fmt: skip
    assert len({ns_seed(*k) for k in keys}) == len(keys)
    with pytest.raises(ValueError):
        ns_seed(-1, 0, 0, 0)


def test_codes_are_unique_and_pools_resolve() -> None:
    assert len({c.code for c in CAMPAIGNS}) == len(CAMPAIGNS)
    assert resolve_campaign("v08", 2201).name == "v08-pilot"
    assert resolve_campaign("v08", 2001).name == "v08-confirmatory"
    assert resolve_campaign("v07-b2", 1101).code == 7201
    assert resolve_campaign("v07-b2", 4001).code == 7202
    assert campaign_code("v07-b1-pilot", 4201) == 7101
    with pytest.raises(ValueError):
        resolve_campaign("v08", 2101)  # first pilot, legacy seeds, not a campaign
    with pytest.raises(ValueError):
        campaign_code("v09-pilot", 3001)


# Draw counts per stream, as the frozen protocols execute them.
_PHASE2 = {"phase2_batch": 100, "replay_init": 16, "replay_fifo": 100, "phase2_probe": 1}
RANGES = {
    "v07-b2": {"phase1_batch": 200, **_PHASE2, "b2_probe": 1},
    "v08": {"phase1_batch": 200, **_PHASE2},
    "v09": {"phase1_batch": 450, "donor": 450, "check_anchor": 32},
}
N_CELLS = 30  # run_grid.sh: with a namespace, cell seed = seed * 10000 + 1 ... 30


def _cells(seeds: tuple[int, ...]) -> list[int]:
    return [s * 10000 + k for s in seeds for k in range(1, N_CELLS + 1)]


def _namespaced(name: str) -> set[int]:
    """Every namespaced seed the campaign's runners draw (no legacy arithmetic)."""
    c = BY_NAME[name]
    prefix = name.rsplit("-", 1)[0]
    if prefix == "v07-b1":
        out = derived_seeds(c.code, c.seeds, {"phase1_batch": 200})
        out |= derived_seeds(c.code, _cells(c.seeds), _PHASE2)
        # lesion and eval both draw the per-query probe with seed = cell + 777
        out |= derived_seeds(c.code, [x + 777 for x in _cells(c.seeds)], {"query": 1})
        return out
    out = derived_seeds(c.code, c.seeds, RANGES[prefix])
    if prefix == "v09":  # manipulation-check draws b = 1 ... 200
        out |= {
            ns_seed(c.code, s, STREAMS["check_draw"], b) for s in c.seeds for b in range(1, 201)
        }
    return out


def _legacy_ints(name: str) -> set[int]:
    """Integer seeds still used directly (model init, global RNG, Gumbel draws)."""
    c = BY_NAME[name]
    prefix = name.rsplit("-", 1)[0]
    out = {s + i for s in c.seeds for i in range(6)}  # torch.manual_seed(s); modules s+1..s+5
    if prefix == "v07-b1":
        cells = _cells(c.seeds)
        out |= set(cells) | {x + 777 for x in cells}
        # main's ADR-0023 v2 linear-probe files: held-out seed + 777 and
        # training pool seed + 778, for train (base s) and lesion (cell)
        out |= {x + d for x in (*c.seeds, *cells) for d in (777, 778)}
    if prefix in ("v07-b2", "v08"):
        out |= {
            gumbel_draw_seed(s, p, r) for s in c.seeds for p in (0, 1) for r in range(GUMBEL_DRAWS)
        }
    return out


def test_no_namespaced_seed_is_shared_between_any_two_pools() -> None:
    sets = {c.name: _namespaced(c.name) for c in CAMPAIGNS}
    for a, b in itertools.combinations(sets, 2):
        assert not sets[a] & sets[b], (a, b)
    v08 = BY_NAME["v08-pilot"]
    n = len(v08.seeds) * sum(RANGES["v08"].values())
    assert len(derived_seeds(v08.code, v08.seeds, RANGES["v08"])) == n


# Deliberate exception: the v0.7 B-1 and B-2 confirmatory campaigns use the
# same base seeds 4001-4154, hence the same model-initialisation seeds
# (paired initial networks); every pilot pool is disjoint from every
# confirmatory pool, and all other pairs are disjoint.
_SHARED_INIT = {frozenset({"v07-b1-confirmatory", "v07-b2-confirmatory"})}


def test_legacy_integer_seeds_are_disjoint_between_pilot_and_confirmatory_pools() -> None:
    sets = {c.name: _legacy_ints(c.name) for c in CAMPAIGNS}
    for a, b in itertools.combinations(sets, 2):
        if frozenset({a, b}) in _SHARED_INIT:
            assert a.endswith("confirmatory") and b.endswith("confirmatory")
            continue
        assert not sets[a] & sets[b], (a, b)


class _SpyWorld(GaussianWorld):
    def __init__(self) -> None:
        super().__init__(seed=0)
        self.seeds: list[int] = []

    def sample(self, batch_size: int, seed: int):  # type: ignore[no-untyped-def,override]
        self.seeds.append(seed)
        return super().sample(batch_size=batch_size, seed=seed)


def _loop(world: GaussianWorld, ns: int | None):  # type: ignore[no-untyped-def]
    from bouba_sens.builders import build_loop

    return build_loop(world, 7, lock_after=None, seed_namespace=ns)


def test_loop_uses_legacy_seeds_without_namespace(deterministic_torch: None) -> None:
    w = _SpyWorld()
    loop = _loop(w, None)
    loop.pretrain(3, batch_size=4, seed=7)
    assert w.seeds == [7, 8, 9]
    w.seeds.clear()
    loop.lesion_phase(None, 2, batch_size=4, seed=50, replay_buffer_size=3)
    assert w.seeds[:4] == [50 - 99_999, 49, 48, 47]
    assert w.seeds[4:] == [50, 50 - 2 - 0 - 1, 51, 50 - 2 - 1 - 1]


def test_loop_uses_namespaced_seeds_with_a_namespace(deterministic_torch: None) -> None:
    w = _SpyWorld()
    loop = _loop(w, 8101)
    loop.pretrain(2, batch_size=4, seed=2201)
    assert w.seeds == [ns_seed(8101, 2201, STREAMS["phase1_batch"], t) for t in range(2)]
    w.seeds.clear()
    loop.lesion_phase(None, 1, batch_size=4, seed=2201, replay_buffer_size=2)
    expected = [
        ns_seed(8101, 2201, STREAMS["phase2_probe"], 0),
        ns_seed(8101, 2201, STREAMS["replay_init"], 0),
        ns_seed(8101, 2201, STREAMS["replay_init"], 1),
        ns_seed(8101, 2201, STREAMS["phase2_batch"], 0),
        ns_seed(8101, 2201, STREAMS["replay_fifo"], 0),
    ]
    assert w.seeds == expected
    with pytest.raises(ValueError, match="probe seed"):
        loop.lesion_phase(None, 1, batch_size=4, seed=2201, probe_seed=5)
    torch.manual_seed(0)
