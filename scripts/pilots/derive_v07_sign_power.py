"""Power of the exact sign-test sensitivity analysis at N = 150 (PI 2026-10-25, 19:04).

Same planning laws as the robust rules: B-2 from pilot round 3
(``derive_v07_pilot.robust_power``: pilot values resampled, deviations
from the pilot pseudo-median inflated by 1.558, shifted so that the
pseudo-median is mu) and B-1 from the Q3/Q3+ 10 seeds
(``derive_v07_b1_robust``). The sign test targets the median; under
these skewed laws the median differs from the pseudo-median, so the
law is also reported with its median shifted to mu. Reported: P(the
90 % sign-test CI lies inside the bounds) at N = 150, and the same
probability at the bounds (size). Generator ``default_rng(20261025 + 2)``.
Usage:

  uv run python scripts/pilots/derive_v07_sign_power.py R3_PILOT_JSON OUT_JSON
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


an = _load("analyse_v07_tost", _HERE.parent / "analyse_v07_tost.py")
pw = _load("power_prereg", _HERE.parent / "power_prereg_2026_10.py")
b1 = _load("derive_v07_b1_robust", _HERE / "derive_v07_b1_robust.py")

N = 150
SIMS = 20_000
B1_SIMS = 2_000
RNG_SEED = 20261025 + 2


def pm(v: np.ndarray) -> float:
    return float(np.median(((v[:, None] + v[None, :]) / 2).ravel()))


def b2_block(vals: list[float], rng) -> dict:  # type: ignore[no-untyped-def]
    v = np.asarray(vals, float)
    infl = pw.sd_ucl(1.0, len(v))
    dev = (v - pm(v)) * infl
    out: dict = {}
    for centring, dev_c in (
        ("pseudo-median at mu", dev - pm(dev)),
        ("median at mu", dev - float(np.median(dev))),
    ):
        rec = {}
        for mu in (0.0, -0.02, 0.05, -0.05):
            hits = 0
            for _ in range(SIMS):
                x = mu + dev_c[rng.integers(0, len(dev_c), N)]
                _, lo, hi = an.sign_ci(x, 0.90)
                hits += lo > -0.05 and hi < 0.05
            rec[f"mu={mu:+.2f}"] = hits / SIMS
        out[centring] = rec
    return out


def b1_block(rng) -> dict:  # type: ignore[no-untyped-def]
    y = b1.planning_data()
    infl = pw.sd_ucl(1.0, len(y))
    centres = np.array([b1.pseudo_median(y[:, j]) for j in range(5)])
    dev = (y - centres) * infl
    dev = dev - np.array([b1.pseudo_median(dev[:, j]) for j in range(5)])
    out = {}
    for name, c in (("observed per-lock centres", centres), ("all centres 0", np.zeros(5))):
        e1 = both = 0
        for _ in range(B1_SIMS):
            x = np.round((c + dev[rng.integers(0, len(dev), N)]) / b1.QUANTUM) * b1.QUANTUM
            _, lo, hi = an.sign_ci(x.mean(1), 0.90)
            ok = lo > -b1.BOUND and hi < b1.BOUND
            e1 += ok
            if ok:
                both += all(
                    (lambda t: t[1] > -b1.BOUND and t[2] < b1.BOUND)(an.sign_ci(x[:, j], 0.90))
                    for j in range(5)
                )
        out[name] = {"P(E-1 sign)": e1 / B1_SIMS, "P(E-1 and E-1b sign)": both / B1_SIMS}
    return out


def main(src: Path, dst: Path) -> None:
    d = json.loads(src.read_text())
    vals = [s["ross_full_delta_bits"] for s in d["per_seed"]]
    rng = np.random.default_rng(RNG_SEED)
    out = {"N": N, "sims_b2": SIMS, "sims_b1": B1_SIMS, "rng_seed": RNG_SEED}
    out["B-2 sign, P(equivalent)"] = b2_block(vals, rng)
    out["B-1 sign"] = b1_block(rng)
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
