"""N for amendment v0.8 from the pilot quantities (section 7, "Drafted rule for N").

Reads the output of ``scripts/run_v08_subspace.py pilot`` and applies the
frozen rule: N is the smallest N >= 20 (cap 40) for which the simulated
P(Supported) of the safeguarded rule (``v08_rule_selection`` in
``scripts/power_prereg_2026_10.py``), minus two Monte-Carlo SEs, is
>= 0.80, with 100,000 draws per candidate, the permutation-calibrated
Page test, a total trend of g-hat + 0.70 within-seed SD, guard g-hat,
exclusion probability q-hat and the exclusion model read from the pilot
("level", or "seed" = ``independent`` in the script). g-hat uses the
UCL80-inflated sigma-hat (the draft's sigma-hat). The simulation
generator is ``default_rng(20261023 + 808)`` (the draft fixes the draws,
not the generator seed; chosen here and recorded). Usage:

  uv run python scripts/pilots/derive_v08_pilot.py PILOT_JSON OUT_JSON
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
TREND_EXTRA = 0.70
RNG_SEED = 20261023 + 808
MODEL = {"level": "level", "seed": "independent"}


def main(src: Path, dst: Path) -> None:
    q = json.loads(src.read_text())
    out: dict = {"source": str(src), "sims": SIMS, "rng_seed": RNG_SEED}
    if not q.get("fileable_as_drafted"):
        out["N"] = None
        out["note"] = q.get("note", "pilot not fileable as drafted")
        dst.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return
    g, qhat, model = q["g_hat"], q["q_hat_cp80"], MODEL[q["exclusion_model"]]
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    chosen = None
    for n in range(N_MIN, N_MAX + 1):
        r = pw.v08_rule_selection(n, g + TREND_EXTRA, g, qhat, SIMS, rng, exclusion=model)
        p = r["safeguarded"]
        lower = p - 2 * math.sqrt(p * (1 - p) / SIMS)
        rows.append({"N": n, **r, "lower_2se": lower})
        print(f"N={n}: safeguarded={p:.4f} lower={lower:.4f} fail={r['fail']:.4f}", flush=True)
        if lower >= TARGET:
            chosen = n
            break
    out.update(
        {
            "g_hat": g,
            "q_hat": qhat,
            "exclusion_model": model,
            "trend_sd_units": g + TREND_EXTRA,
            "candidates": rows,
            "N": chosen,
            "rule_outcome": (
                "N found" if chosen is not None else "no N <= 40 qualifies (section 7: PI decides)"
            ),
        }
    )
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "candidates"}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
