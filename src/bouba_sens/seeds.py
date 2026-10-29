"""Per-campaign seed namespaces (PI decision 2026-10-25, 19:47, after the pilots).

Legacy runs derive their data seeds by arithmetic on the experiment seed
(Phase-1 batch t of seed s uses ``s + t``, replay ``s - 1 - i``, the
probe ``s - 99_999``, ...), so the streams of two seeds overlap (seed
4002, step 199 and seed 4201, step 0 both draw ``world.sample(seed =
4201)``) and pilot and confirmatory pools share training windows. With a
namespace, every derived seed is

    ns_seed(code, s, stream, t) = SeedSequence([code, s, stream, t])
                                  .generate_state(1, uint64)[0] mod 2**63

with ``code`` the campaign code below (pilot and confirmatory pools of
one amendment have different codes), ``s`` the experiment seed,
``stream`` the kind of draw (``STREAMS``) and ``t`` its index. Distinct
``(code, s, stream, t)`` give distinct SeedSequence entropy and hence
independent generator states; ``tests/unit/test_seeds.py`` enumerates
the namespaced seeds every runner draws (as executed) and checks that no
two pools share one.

The legacy arithmetic is unchanged when no namespace is set.

Seeds that stay plain integers (not namespaced), and why the test still
covers them (``test_legacy_integer_seeds_are_disjoint_between_pilot_and_
confirmatory_pools``):

* model initialisation and the global torch RNG: ``s`` and the module
  seeds ``s + 1 ... s + 5``; for B-1 lesion cells the cell seed and
  ``cell + 777`` (lesion / eval). They are disjoint between every pilot
  and every confirmatory pool; the B-1 and B-2 confirmatory campaigns
  share base seeds 4001-4154 and hence initial networks (a deliberate
  pairing, confirmed by the PI on 2026-10-28);
* K = 8 Gumbel draw seeds ``2e9 + 1000 s + 100 p + r`` (disjoint by s);
* design constants shared on purpose by every pool: the v0.9 held-out
  probe seed, the probe-training allocation seed, the permutation and
  simulation generators, and the fixed init seeds 0-5 of the CLI's T1
  (congenital) cells.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

STREAMS: dict[str, int] = {
    "phase1_batch": 0,  # Phase-1 batch, t = step
    "phase2_batch": 1,  # Phase-2 fresh batch, t = step
    "replay_init": 2,  # initial replay buffer, t = i
    "replay_fifo": 3,  # FIFO replay refill, t = step
    "phase2_probe": 4,  # Phase-2 probe, t = 0
    "donor": 5,  # v0.9 donor draws, t = step
    "query": 6,  # B-1 per-query Me1 probe, t = 0
    "b2_probe": 7,  # v0.7 B-2 balanced probe, t = 0
    "check_anchor": 8,  # v0.9 manipulation-check anchors, t = r
    "check_draw": 9,  # v0.9 manipulation-check donor draws, t = b
}

_MASK63 = (1 << 63) - 1


def ns_seed(code: int, seed: int, stream: int, index: int) -> int:
    """63-bit generator seed of draw ``index`` of ``stream`` for seed ``seed``."""
    if min(code, seed, stream, index) < 0:
        raise ValueError("code, seed, stream and index must be >= 0")
    state = np.random.SeedSequence([code, seed, stream, index]).generate_state(1, np.uint64)
    return int(state[0]) & _MASK63


@dataclass(frozen=True)
class Campaign:
    name: str
    code: int
    seeds: tuple[int, ...]


def _r(a: int, b: int) -> tuple[int, ...]:
    return tuple(range(a, b + 1))


# Campaign codes (frozen). Pilot and confirmatory pools are separate
# campaigns. The v0.8 and v0.9 pilots of the redesign use the new
# namespace; the earlier pilots (2101-2110, 3101-3105) ran on the legacy
# arithmetic and are superseded.
CAMPAIGNS: tuple[Campaign, ...] = (
    Campaign("v07-b1-pilot", 7101, (4201,)),
    Campaign("v07-b1-confirmatory", 7102, _r(4001, 4154)),
    Campaign("v07-b2-pilot", 7201, _r(1101, 1105)),
    Campaign("v07-b2-confirmatory", 7202, _r(4001, 4154)),
    # rev. B pilot 2201-2210, rev. C pilot 2211-2220 (PI 2026-10-28)
    Campaign("v08-pilot", 8101, _r(2201, 2220)),
    Campaign("v08-confirmatory", 8102, _r(2001, 2044)),
    # rev. B 3201-3205, rev. C 3206-3210 (PI 2026-10-28), rev. D 3211-3215
    # (PI 2026-10-29)
    Campaign("v09-pilot", 9101, _r(3201, 3215)),
    Campaign("v09-confirmatory", 9102, _r(3001, 3044)),
)
BY_NAME = {c.name: c for c in CAMPAIGNS}


def campaign_code(name: str, seed: int) -> int:
    """Code of campaign ``name``; refuses a seed outside its pool."""
    c = BY_NAME[name]
    if seed not in c.seeds:
        raise ValueError(f"seed {seed} is not in the {name} pool")
    return c.code


def resolve_campaign(prefix: str, seed: int) -> Campaign:
    """The pilot or confirmatory campaign of ``prefix`` (e.g. 'v08') holding ``seed``."""
    hits = [c for c in CAMPAIGNS if c.name.startswith(prefix + "-") and seed in c.seeds]
    if len(hits) != 1:
        raise ValueError(f"seed {seed} is in no (or several) {prefix} pools")
    return hits[0]


def derived_seeds(code: int, seeds: Iterable[int], ranges: dict[str, int]) -> set[int]:
    """Every namespaced seed of ``seeds`` for the given stream index ranges."""
    out: set[int] = set()
    for s in seeds:
        for stream, n in ranges.items():
            for t in range(n):
                out.add(ns_seed(code, s, STREAMS[stream], t))
    return out
