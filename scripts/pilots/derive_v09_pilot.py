"""N for amendment v0.9 from the pilot variance (section 7, "N").

Reads ``scripts/run_v09_decorrelation.py`` output for the pilot seeds and
applies the frozen rule: N is the smallest N >= 20 (cap 40) for which
the simulated P(A+ and C+) at mu0 = 0.05 under the "early" profile with
rho = 0 (``v09_rule`` in ``scripts/power_prereg_2026_10.py``), minus two
Monte-Carlo SEs, is >= 0.80, with 100,000 draws per candidate, the
permutation-calibrated Page test and SD = sigma_plan (sigma-hat x 1.558,
df = 4). The generator is ``default_rng(20261023 + 909)`` (the draft
fixes the draws, not the generator seed; chosen here and recorded).
Usage:

  uv run python scripts/pilots/derive_v09_pilot.py PILOT_JSON OUT_JSON
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

_POWER = Path(__file__).resolve().parents[1] / "power_prereg_2026_10.py"
_spec = importlib.util.spec_from_file_location("power_prereg", _POWER)
pw = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(pw)

SIMS = 100_000
N_MIN, N_MAX = 20, 40
TARGET = 0.80
RNG_SEED = 20261023 + 909


def main(src: Path, dst: Path) -> None:
    d = json.loads(src.read_text())
    revc_fail = d.get("design") in ("rev. C", "rev. D") and not d.get("all_gates_passed")
    if "variance_components" not in d or d.get("stopped") or revc_fail:
        why = d.get("stopped") or ("a prerequisite gate failed" if revc_fail else "no D matrix")
        out0 = {"source": str(src), "N": None, "note": f"no N: {why}"}
        dst.write_text(json.dumps(out0, indent=2) + "\n")
        print(json.dumps(out0, indent=2))
        return
    sd = d["variance_components"]["sigma_plan"]
    rng = np.random.default_rng(RNG_SEED)
    rows, chosen = [], None
    for n in range(N_MIN, N_MAX + 1):
        p = float(pw.v09_rule(n, 0.05, "early", sd, 0.0, SIMS, rng)["headline D-1&D-2"])
        lower = p - 2 * math.sqrt(p * (1 - p) / SIMS)
        rows.append({"N": n, "p_headline": p, "lower_2se": lower})
        print(f"N={n}: P(A+ and C+)={p:.4f} lower={lower:.4f}", flush=True)
        if lower >= TARGET:
            chosen = n
            break
    out = {
        "source": str(src),
        "sims": SIMS,
        "rng_seed": RNG_SEED,
        "sigma_plan": sd,
        "candidates": rows,
        "N": chosen,
        "rule_outcome": (
            "N found" if chosen is not None else "no N <= 40 qualifies (section 7: PI decides)"
        ),
    }
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "candidates"}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
