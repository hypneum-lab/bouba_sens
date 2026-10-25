"""Pilot-derived quantities for amendment v0.7 (sections 4.3 and 7).

Reads the B-2 pilot output of the E1 runner (``--preset v07-e2``,
seeds 1101-1105) and computes what section 7 asks for at filing: the
per-seed SD of the primary Ross Delta-MI-hat and of the other readout,
the one-sided UCL80 of each SD (df = 4, ``sd_ucl(sd, 5)``), the exact
TOST power at N = 95 for mu = 0 and mu = +-0.02 bit
(``tost_power_exact``), the N that would give 0.80 and the
informativeness label. Only the primary readout is the registered power
computation; the other readout is descriptive. N is not changed
(section 7).

The primary readout is read from the pilot file: ``hard`` for round 1
(the drafted protocol of 2026-10-24) and ``gumbel`` for round 2 (PI
decision 2026-10-25, after round 1: Gumbel primary, class-balanced
probe). Usage:

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
_AN = Path(__file__).resolve().parents[1] / "analyse_v07_tost.py"
_spec2 = importlib.util.spec_from_file_location("analyse_v07_tost", _AN)
an = importlib.util.module_from_spec(_spec2)
assert _spec2.loader is not None
_spec2.loader.exec_module(an)

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


ROBUST_SIMS = 20_000
ROBUST_GRID_SIMS = 2_000
ROBUST_N_GRID = (95, 150, 200, 300, 500, 800, 1200)
ROBUST_SEED = 20261025


def _pseudo_median_discrete(v: np.ndarray) -> float:
    """Pseudo-median of the uniform distribution on ``v``: median of (X1 + X2) / 2."""
    w = np.sort(((v[:, None] + v[None, :]) / 2).ravel())
    return float(np.median(w))


def robust_power(values: list[float]) -> dict:
    """Power of the robust E-2 rule (90 % Hodges-Lehmann CI inside +-0.05 bit).

    Planning model, from the pilot (PI decision 2026-10-25): seeds are
    drawn with replacement from the pilot values, their deviations from
    the pilot pseudo-median inflated by the UCL80 multiplier of section 7
    (sqrt(4 / chi2_0.20,4) = 1.558, the df = 4 rule used for the SD), and
    the distribution shifted so that its pseudo-median is mu. Power at
    N = 95 with 20,000 draws; the smallest N on a coarse grid that reaches
    0.80 with 2,000 draws per N. Generator ``default_rng(20261025)``.
    """
    v = np.asarray(values, float)
    infl = pw.sd_ucl(1.0, len(v))
    dev = (v - _pseudo_median_discrete(v)) * infl
    dev = dev - _pseudo_median_discrete(dev)
    rng = np.random.default_rng(ROBUST_SEED)

    def power(mu: float, n: int, sims: int) -> float:
        hits = 0
        for _ in range(sims):
            x = mu + dev[rng.integers(0, len(dev), n)]
            _, lo, hi = an.hl_ci(x, 0.90)
            hits += lo > -BOUND and hi < BOUND
        return hits / sims

    rec: dict = {"inflation": infl, "support": dev.tolist()}
    for mu in (0.0, -0.02):
        p95 = power(mu, N_FIXED, ROBUST_SIMS)
        n80 = next((n for n in ROBUST_N_GRID if power(mu, n, ROBUST_GRID_SIMS) >= 0.80), None)
        rec[f"mu={mu:+.2f}"] = {
            "power_N95": p95,
            "N_for_0.80_grid": n80 if n80 is not None else f"> {ROBUST_N_GRID[-1]}",
        }
    rec["label_planning_case"] = label(rec["mu=-0.02"]["power_N95"])
    return rec


def main(src: Path, dst: Path) -> None:
    d = json.loads(src.read_text())
    seeds = [s["seed"] for s in d["per_seed"]]
    if seeds != PILOT_SEEDS or d.get("preset") != "v07-e2":
        raise SystemExit(f"not the v0.7 B-2 pilot: preset={d.get('preset')} seeds={seeds}")
    primary = d.get("primary_readout")
    rows = d["per_seed"]
    if str(primary).startswith("gumbel_k"):
        main_k(src, dst, d, rows)
        return
    other = {"hard": "gumbel", "gumbel": "hard"}[primary]
    out = {
        "source": str(src),
        "primary_readout": primary,
        "probe_scheme": d.get("probe_scheme", "contiguous"),
        "N_fixed": N_FIXED,
        "bound_bits": BOUND,
        f"primary_{primary}_ross": power_block([s["ross_full_delta_bits"] for s in rows]),
        f"{other}_ross_descriptive": power_block(
            [s[f"ross_full_delta_bits_{other}"] for s in rows]
        ),
        f"ross_zero_distance_points_{primary}_pre": [s["ross_n_zero_distance_pre"] for s in rows],
        f"ross_zero_distance_points_{primary}_post": [s["ross_n_zero_distance_post"] for s in rows],
        f"ross_zero_distance_points_{other}_pre": [
            s[f"ross_n_zero_distance_pre_{other}"] for s in rows
        ],
        f"ross_zero_distance_points_{other}_post": [
            s[f"ross_n_zero_distance_post_{other}"] for s in rows
        ],
    }
    if "probe_label_counts" in rows[0]:
        out["probe_label_counts"] = {str(s["seed"]): s["probe_label_counts"] for s in rows}
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def main_k(src: Path, dst: Path, d: dict, rows: list) -> None:
    """Round 3 (PI decisions 2026-10-25, 13:12): K-draw Gumbel, robust E-2, switch."""
    from scipy import stats

    k = sum(bool(s["switch_post"]) for s in rows)
    ci = stats.binomtest(k, len(rows)).proportion_ci(confidence_level=0.95, method="exact")
    vals = [s["ross_full_delta_bits"] for s in rows]
    out = {
        "source": str(src),
        "primary_readout": d["primary_readout"],
        "probe_scheme": d.get("probe_scheme"),
        "gumbel_draws": d.get("gumbel_draws"),
        "N_fixed": N_FIXED,
        "bound_bits": BOUND,
        "primary_values": vals,
        "primary_hl": an.hl_ci(vals, 0.90)[0],
        "primary_median": float(np.median(vals)),
        "robust_primary_power": robust_power(vals),
        "mean_based_secondary": power_block(vals),
        "gumbel1_descriptive": power_block([s["ross_full_delta_bits_gumbel1"] for s in rows]),
        "hard_descriptive": power_block([s["ross_full_delta_bits_hard"] for s in rows]),
        "per_draw_delta_sd_within_seed": {
            str(s["seed"]): float(np.std(s["ross_full_delta_bits_per_draw"], ddof=1)) for s in rows
        },
        "switch": {
            "per_seed": {
                str(s["seed"]): {
                    "pre": s["switch_pre"],
                    "post": s["switch_post"],
                    "post_top1_share": s["post_top1_share"],
                    "post_between_class_share": s["post_between_class_share"],
                    "pre_top1_share": s["pre_top1_share"],
                    "pre_between_class_share": s["pre_between_class_share"],
                }
                for s in rows
            },
            "post_count": k,
            "proportion": k / len(rows),
            "ci95_exact": [float(ci.low), float(ci.high)],
        },
    }
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
