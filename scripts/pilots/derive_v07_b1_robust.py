"""Power of the robust B-1 rule (v0.7 E-1 -> E-1b) at N = 150 (PI 2026-10-25, 13:32).

The PI switched E-1 (and E-1b, E-1c) to the robust TOST (90 % Hodges-
Lehmann CI, ``analyse_v07_tost.hl_tost``) and fixed N = 150. B-1 has no
new pilot; its planning data are the Q3/Q3+ 10 seeds x 5 LOCK_AFTER of
``reports/v0.5_amedi_curve_multiseed_10seed_final.json`` (section 7).

Planning law (same construction as the robust B-2 power): a seed is a
5-vector (one Me7 grid median per lock) resampled with replacement from
the 10 planning seeds; its deviations from the per-lock pseudo-medians
are inflated by the df = 9 UCL80 factor (``sd_ucl(1, 10)``) and shifted
to per-lock centres (observed pseudo-medians, or 0); values are rounded
to the 0.00625 quantum. E-1 is the robust TOST of the per-seed mean
over the five locks; E-1b needs all five per-lock robust TOSTs
equivalent (IUT). E-1c has no planning data. Generator
``default_rng(20261025 + 1)``. Usage:

  uv run python scripts/pilots/derive_v07_b1_robust.py OUT_JSON
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_AN = _ROOT / "scripts" / "analyse_v07_tost.py"
_spec = importlib.util.spec_from_file_location("analyse_v07_tost", _AN)
an = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(an)
_PW = _ROOT / "scripts" / "power_prereg_2026_10.py"
_spec2 = importlib.util.spec_from_file_location("power_prereg", _PW)
pw = importlib.util.module_from_spec(_spec2)
assert _spec2.loader is not None
_spec2.loader.exec_module(pw)

LOCKS = ("50", "75", "100", "125", "150")
QUANTUM = 0.00625
BOUND = 0.025
N = 150
SIMS = 2_000
GRID = (150, 300, 500)
GRID_SIMS = 500
RNG_SEED = 20261025 + 1


def pseudo_median(v: np.ndarray) -> float:
    return float(np.median(((v[:, None] + v[None, :]) / 2).ravel()))


def planning_data() -> np.ndarray:
    rep = json.loads((_ROOT / "reports/v0.5_amedi_curve_multiseed_10seed_final.json").read_text())
    return np.array([[rep["matrix"][str(s)][lk] for lk in LOCKS] for s in rep["all_seeds"]])


def simulate(dev: np.ndarray, centres: np.ndarray, n: int, sims: int, rng) -> dict[str, float]:
    e1 = both = 0
    for _ in range(sims):
        x = np.round((centres + dev[rng.integers(0, len(dev), n)]) / QUANTUM) * QUANTUM
        _, lo, hi = an.hl_ci(x.mean(1), 0.90)
        ok1 = lo > -BOUND and hi < BOUND
        e1 += ok1
        if ok1:
            ok_b = True
            for j in range(x.shape[1]):
                _, a, b = an.hl_ci(x[:, j], 0.90)
                if not (a > -BOUND and b < BOUND):
                    ok_b = False
                    break
            both += ok_b
    return {"P(E-1 equivalent)": e1 / sims, "P(E-1 and E-1b)": both / sims}


def main(dst: Path) -> None:
    y = planning_data()
    infl = pw.sd_ucl(1.0, len(y))
    centres_obs = np.array([pseudo_median(y[:, j]) for j in range(y.shape[1])])
    dev = (y - centres_obs) * infl
    dev = dev - np.array([pseudo_median(dev[:, j]) for j in range(dev.shape[1])])
    rng = np.random.default_rng(RNG_SEED)
    out: dict = {
        "n_planning_seeds": len(y),
        "inflation_df9": infl,
        "per_lock_pseudo_medians": centres_obs.tolist(),
        "N": N,
        "sims": SIMS,
    }
    for name, c in (("observed per-lock centres", centres_obs), ("all centres 0", np.zeros(5))):
        rec = simulate(dev, c, N, SIMS, rng)
        grid = {}
        for n in GRID:
            grid[str(n)] = simulate(dev, c, n, GRID_SIMS, rng)["P(E-1 and E-1b)"]
        rec["P(E-1 and E-1b) on grid"] = grid
        out[name] = rec
        print(name, json.dumps(rec), flush=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
