"""Pilot-derived quantities for amendment v0.7 (sections 4.3 and 7).

Reads the B-2 pilot output of the E1 runner (``--preset v07-e2``,
seeds 1101-1105) and computes what section 7 asks for at filing: the
per-seed SD of the primary (hard readout) Ross Delta-MI-hat and of the
E-2g (Gumbel) value, the one-sided UCL80 of each SD (df = 4,
``sd_ucl(sd, 5)``), the exact TOST power at N = 95 for mu = 0 and
mu = +-0.02 bit (``tost_power_exact``), the N that would give 0.80 and
the informativeness label. Only the hard readout is the registered
power computation; the Gumbel numbers are descriptive. N is not changed
(section 7). Usage:

  uv run python scripts/pilots/derive_v07_pilot.py PILOT_JSON OUT_JSON
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_POWER = Path(__file__).resolve().parents[1] / "power_prereg_2026_10.py"
_spec = importlib.util.spec_from_file_location("power_prereg", _POWER)
pw = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(pw)

N_FIXED = 95
BOUND = 0.05
PILOT_SEEDS = [1101, 1102, 1103, 1104, 1105]


def label(power: float) -> str:
    if power >= 0.80:
        return "adequately powered"
    if power >= 0.50:
        return "under-powered"
    return "low informativeness"


def power_block(values: list[float]) -> dict:
    a = np.asarray(values, float)
    sd = float(a.std(ddof=1))
    rec: dict = {"per_seed": values, "mean": float(a.mean()), "sd": sd}
    if sd == 0.0:
        rec["degenerate"] = (
            "every per-seed value is identical; the SD is 0, its UCL80 is 0 and "
            "the normal planning model of section 7 is undefined (any TOST on "
            "such data has a zero-width CI)"
        )
        return rec
    sdu = pw.sd_ucl(sd, len(a))
    rec["sd_ucl80_df4"] = sdu
    for mu in (0.0, -0.02):
        p = pw.tost_power_exact(mu, sdu, N_FIXED, BOUND)
        n80 = next(
            (n for n in range(3, 20000) if pw.tost_power_exact(mu, sdu, n, BOUND) >= 0.80), None
        )
        rec[f"mu={mu:+.2f}"] = {"exact_power_N95": p, "N_for_0.80": n80}
    rec["label_planning_case"] = label(rec["mu=-0.02"]["exact_power_N95"])
    return rec


def main(src: Path, dst: Path) -> None:
    d = json.loads(src.read_text())
    seeds = [s["seed"] for s in d["per_seed"]]
    if seeds != PILOT_SEEDS or d.get("preset") != "v07-e2":
        raise SystemExit(f"not the v0.7 B-2 pilot: preset={d.get('preset')} seeds={seeds}")
    out = {
        "source": str(src),
        "N_fixed": N_FIXED,
        "bound_bits": BOUND,
        "primary_hard_ross": power_block([s["ross_full_delta_bits"] for s in d["per_seed"]]),
        "e2g_gumbel_ross_descriptive": power_block(
            [s["ross_full_delta_bits_gumbel"] for s in d["per_seed"]]
        ),
        "ross_zero_distance_points_hard_pre": [
            s["ross_n_zero_distance_pre"] for s in d["per_seed"]
        ],
        "ross_zero_distance_points_hard_post": [
            s["ross_n_zero_distance_post"] for s in d["per_seed"]
        ],
    }
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
