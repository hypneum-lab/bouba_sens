"""Structural check for the v0.8 hard-demodulation probe (descriptive).

Not a v0.8 pilot run (the v0.8 driver does not exist yet) and it uses
no v0.8 pilot or confirmatory seed: seed 0 is a historical default
grid seed outside every amendment pool. For each lock, train Phase 1
(200 steps, batch 16, HARD gating, ``constellation_lock_after`` = lock)
as the E1 runner builds the loop, then capture Z_pre on the v0.8 probe
(``world.sample(512, seed = s - 99_999)``) with ``fuse(hard_demod=True)``
and report distinct rows, the section 3.1 numerical rank and the plug-in
participation ratio. Usage:

  BOUBA_SENS_STUDYFORREST_DATA=... uv run python \
    scripts/pilots/diagnose_v08_hard_rank.py 0 25 175
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from bouba_sens.metrics.subspace import numerical_rank, participation_ratio

_RUNNER = Path(__file__).resolve().parents[1] / "spikes" / "e1_b2_dimensional_mine.py"
_spec = importlib.util.spec_from_file_location("e1_runner", _RUNNER)
e1 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(e1)


def main(seed: int, locks: list[int]) -> None:
    torch.use_deterministic_algorithms(True)
    for lock in locks:
        torch.manual_seed(seed)
        loop = e1.build_loop("studyforrest", seed, lock)
        loop.pretrain(200, batch_size=16, seed=seed)
        probe = loop.world.sample(batch_size=512, seed=seed - 99_999)
        z_hard = e1._full_fused(loop, probe, hard=True)
        z_soft = e1._full_fused(loop, probe, hard=False)
        rec = {"seed": seed, "lock": lock, "world_class": type(loop.world).__name__}
        for name, z in (("hard", z_hard), ("gumbel", z_soft)):
            zz = z.double()
            rec[name] = {
                "n_distinct_rows": int(np.unique(z.numpy(), axis=0).shape[0]),
                "numerical_rank": numerical_rank(zz),
                "participation_ratio": participation_ratio(zz) if numerical_rank(zz) else None,
            }
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), [int(a) for a in sys.argv[2:]])
