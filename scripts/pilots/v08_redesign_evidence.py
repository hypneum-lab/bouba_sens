"""Evidence for the v0.8 redesign proposal, from the existing pilot blocks only.

Read-only, descriptive analysis of the 70 stored pilot blocks (seeds
2101-2110, Gumbel K = 8 averaged codes; ``run_v08_subspace.py capture``).
No rule is applied and no new network is trained. For each block:

* the registered quantities at k* = 13: S(pre, post_c), C = S(pre,
  control), Y = median_c [S(pre, post_c) - C], S_split;
* the k-free candidate: d_c = 1 - CKA(post_c, control) (linear CKA on
  the paired probe rows), its block median Y_cka, and the drift
  1 - CKA(pre, control);
* the spectrum near k*: lambda_k / lambda_(k+1) for k = 1 ... 40 of
  Z_pre, the k with the largest ratio in [ceil(PR / 2), 2 PR], and that
  ratio.

Within-seed SDs use the two-way ANOVA residual (seeds x locks), as in
section 7. Usage:

  uv run python scripts/pilots/v08_redesign_evidence.py BLOCKS_DIR OUT_JSON
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

from bouba_sens.metrics.subspace import participation_ratio, split_half_noise_floor, subspace_shift
from bouba_sens.metrics.subspace_aggregate import LOCKS
from bouba_sens.v08_driver import PILOT_SEEDS, anova_residual_sd, block_path, load_block

K_STAR = 13


def linear_cka(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.double() - a.double().mean(0)
    y = b.double() - b.double().mean(0)
    num = float(torch.linalg.matrix_norm(y.T @ x) ** 2)
    den = float(torch.linalg.matrix_norm(x.T @ x) * torch.linalg.matrix_norm(y.T @ y))
    return num / den


def spectrum(z: torch.Tensor) -> np.ndarray:
    zc = z.double() - z.double().mean(0)
    return (torch.linalg.svdvals(zc) ** 2 / (z.shape[0] - 1)).numpy()


def main(blocks_dir: Path, dst: Path) -> None:
    rows = []
    for s in PILOT_SEEDS:
        for lk in LOCKS:
            b = load_block(block_path(blocks_dir, s, lk))
            c = subspace_shift(b.z_pre, b.z_control, K_STAR)
            ds = [subspace_shift(b.z_pre, z, K_STAR) - c for z in b.z_post.values()]
            d_cka = [1 - linear_cka(z, b.z_control) for z in b.z_post.values()]
            lam = spectrum(b.z_pre)
            ratios = lam[:40] / lam[1:41]
            pr = participation_ratio(b.z_pre)
            lo, hi = math.ceil(pr / 2), int(2 * pr)
            k_gap = int(np.argmax(ratios[lo - 1 : hi]) + lo)
            rows.append(
                {
                    "seed": s,
                    "lock": lk,
                    "Y": float(np.median(ds)),
                    "C": c,
                    "S_split": split_half_noise_floor(b.z_pre, K_STAR, seed=s),
                    "Y_cka": float(np.median(d_cka)),
                    "d_cka_min": float(min(d_cka)),
                    "d_cka_max": float(max(d_cka)),
                    "drift_cka": 1 - linear_cka(b.z_pre, b.z_control),
                    "pr": pr,
                    "eigengap_at_kstar": float(ratios[K_STAR - 1]),
                    "k_largest_gap_in_window": k_gap,
                    "largest_gap_in_window": float(ratios[k_gap - 1]),
                    "top_ratios_1_20": [float(r) for r in ratios[:20]],
                }
            )
            print(json.dumps({k: rows[-1][k] for k in ("seed", "lock", "Y", "Y_cka")}), flush=True)

    def mat(key: str) -> np.ndarray:
        return np.array([[r[key] for r in rows if r["seed"] == s] for s in PILOT_SEEDS])

    out: dict = {"k_star": K_STAR, "blocks": rows}
    for key in ("Y", "Y_cka", "drift_cka", "S_split"):
        m = mat(key)
        sd, df = anova_residual_sd(m)
        out[f"{key}_summary"] = {
            "median": float(np.median(m)),
            "min": float(m.min()),
            "max": float(m.max()),
            "within_seed_sd": sd,
            "df": df,
            "level_means": m.mean(0).tolist(),
            "endpoint_contrast_median": float(np.median(m[:, -1] - m[:, 0])),
        }
    g = [r["eigengap_at_kstar"] for r in rows]
    out["eigengap_at_kstar"] = {
        "median": statistics.median(g),
        "share_below_1_1": float(np.mean([x < 1.1 for x in g])),
    }
    out["largest_gap_in_window"] = {
        "median": statistics.median(r["largest_gap_in_window"] for r in rows),
        "share_below_1_1": float(np.mean([r["largest_gap_in_window"] < 1.1 for r in rows])),
        "k_values": sorted({r["k_largest_gap_in_window"] for r in rows}),
    }
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "blocks"}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
