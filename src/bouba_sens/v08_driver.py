"""Driver for the v0.8 subspace shift (amendment v0.8 sections 3, 3.1, 7).

Two stages, so that k* can be fixed on the pilot Z_pre before any
k-dependent quantity is computed (section 7, "Order"):

1. ``run_block`` trains the Phase-1 checkpoint of one (seed s, lock l)
   block and calls ``subspace_protocol.capture_block`` with the frozen
   defaults (probe 512, 100 lesion steps, batch 16, ``k_steps`` 50,
   Gumbel probe readout). ``save_block`` / ``load_block`` keep the raw
   code matrices on disk.
2. ``pilot_quantities`` takes the pilot blocks: k* by
   ``select_k_from_pilot`` on every Z_pre, then ``evaluate_block`` at k*,
   ``aggregate`` and the planning inputs of section 7 (sigma-hat from the
   two-way ANOVA without replication on complete blocks with its UCL80,
   the median S_split, g-hat, the Clopper-Pearson q-hat and the exclusion
   model). The N rule itself is simulated by
   ``scripts/pilots/derive_v08_pilot.py``.

Seed mapping (section 3 item 11): the Phase-1 checkpoint is trained with
seed s and ``LOCK_AFTER`` = l (batch 16, 200 steps), after re-seeding the
global torch RNG with s; every Phase-2 arm restores it into a fresh loop
with the same lock and uses Phase-2 seed s (``capture_block``).
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats  # type: ignore[import-untyped]

from bouba_sens.builders import build_loop
from bouba_sens.loop import AdaptationLoop, Checkpoint
from bouba_sens.metrics.subspace import select_k_from_pilot
from bouba_sens.metrics.subspace_aggregate import LOCKS, aggregate, evaluate_block
from bouba_sens.subspace_protocol import TRAIN_BATCH_SIZE, BlockCodes, capture_block
from bouba_sens.world.base import WorldSimulator

STEPS_TRAIN = 200
PILOT_SEEDS: tuple[int, ...] = tuple(range(2101, 2111))  # first pilot (legacy seeds)
# Rev. B (PI decision 2026-10-25, 19:47): fresh pilot seeds in the v08-pilot
# seed namespace, deterministic soft readout primary, K = 8 Gumbel secondary.
PILOT_SEEDS_REVB: tuple[int, ...] = tuple(range(2201, 2211))
REVB_PRIMARY_DEMOD = "softmax"
REVB_SECONDARY: tuple[tuple[str, int], ...] = (("soft", 8),)
REVB_GUARD_SD = 0.5
CONFIRMATORY_POOL: tuple[int, ...] = tuple(range(2001, 2045))
EXPECTED_WORLD = "StudyforrestRealWorld"
MIN_COMPLETE_PILOT_BLOCKS = 3
Q_LEVEL = 0.80  # one-sided Clopper-Pearson / chi-square UCL level
EIGENGAP_UNSTABLE = 1.1


def check_world(world: object) -> str:
    """Refuse anything but the real Studyforrest bridge (section 4, world check)."""
    name = type(world).__name__
    if name != EXPECTED_WORLD:
        raise RuntimeError(f"v0.8 needs {EXPECTED_WORLD}; got {name} (technical failure)")
    return name


def require_determinism() -> None:
    """Bit-identical arms need deterministic CPU kernels (``tests/conftest.py``).

    On Linux x86 ``torch.linalg.lstsq`` (called by the demodulator) can
    differ by 1 ulp between identical calls unless
    ``torch.use_deterministic_algorithms(True)`` is set; then Z_pre would
    not be shared across the arms of a block.
    """
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("call torch.use_deterministic_algorithms(True) first")


def train_checkpoint(
    world: WorldSimulator, seed: int, lock: int, *, seed_namespace: int | None = None
) -> Checkpoint:
    """Phase-1 checkpoint of block (seed, lock): 200 steps, batch 16, seed s."""
    torch.manual_seed(seed)
    loop = build_loop(world, seed, lock_after=lock, seed_namespace=seed_namespace)
    return loop.pretrain(STEPS_TRAIN, batch_size=TRAIN_BATCH_SIZE, seed=seed)


def revb_namespace(seed: int) -> int:
    """Seed-namespace code of a rev. B seed (v08-pilot or v08-confirmatory)."""
    from bouba_sens.seeds import resolve_campaign

    return resolve_campaign("v08", seed).code


@dataclass(frozen=True)
class BlockRun:
    block: BlockCodes
    wall_s: float


def run_block(
    world: WorldSimulator,
    seed: int,
    lock: int,
    *,
    design: str = "revb",
    seed_namespace: int | None = None,
    **capture_kw: Any,
) -> BlockRun:
    """Checkpoint + every Phase-2 arm of one block (section 3 item 11).

    ``design="revb"`` (default): the v08 seed namespace of the seed's pool
    (unless ``seed_namespace`` is given), the deterministic soft readout as
    the primary capture and K = 8 Gumbel as the secondary readout.
    ``design="legacy"``: the first pilot's settings (legacy derived seeds,
    K = 8 Gumbel primary).
    """
    require_determinism()
    ns: int | None
    if design == "revb":
        ns = revb_namespace(seed) if seed_namespace is None else seed_namespace
        capture_kw = {
            "probe_demod": REVB_PRIMARY_DEMOD,
            "probe_gumbel_draws": 0,
            "extra_readouts": REVB_SECONDARY,
            **capture_kw,
        }
    elif design == "legacy":
        ns = seed_namespace
    else:
        raise ValueError(f"unknown design {design!r}")
    t0 = time.monotonic()
    ckpt = train_checkpoint(world, seed, lock, seed_namespace=ns)

    def make_loop() -> AdaptationLoop:
        return build_loop(world, seed, lock_after=lock, seed_namespace=ns)

    block = capture_block(make_loop, ckpt, seed=seed, lock=lock, **capture_kw)
    return BlockRun(block, time.monotonic() - t0)


def _block_dict(block: BlockCodes) -> dict[str, Any]:
    return {
        "seed": block.seed,
        "lock": block.lock,
        "phase2_seed": block.phase2_seed,
        "probe_labels": block.probe_labels,
        "z_pre": block.z_pre,
        "z_control": block.z_control,
        "z_post": {f"{m}|{f}": z for (m, f), z in block.z_post.items()},
        "secondary": {name: _block_dict(b) for name, b in block.secondary.items()},
    }


def save_block(block: BlockCodes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_block_dict(block), path)


def _from_dict(d: dict[str, Any]) -> BlockCodes:
    z_post = {}
    for key, z in d["z_post"].items():
        m, f = key.split("|")
        z_post[(m, float(f))] = z
    return BlockCodes(
        seed=int(d["seed"]),
        lock=int(d["lock"]),
        phase2_seed=int(d["phase2_seed"]),
        probe_labels=d["probe_labels"],
        z_pre=d["z_pre"],
        z_control=d["z_control"],
        z_post=z_post,
        secondary={n: _from_dict(b) for n, b in d.get("secondary", {}).items()},
    )


def load_block(path: Path) -> BlockCodes:
    return _from_dict(torch.load(path, weights_only=True))


def block_path(out_dir: Path, seed: int, lock: int) -> Path:
    return out_dir / f"seed{seed}_lock{lock}.pt"


def anova_residual_sd(y: np.ndarray) -> tuple[float, int]:
    """Residual SD of the two-way ANOVA without replication (rows = seeds)."""
    n, m = y.shape
    resid = y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + y.mean()
    df = (n - 1) * (m - 1)
    return math.sqrt(float((resid**2).sum()) / df), df


def sd_ucl_df(sd: float, df: int, level: float = Q_LEVEL) -> float:
    """One-sided upper confidence limit of an SD with ``df`` degrees of freedom."""
    return sd * math.sqrt(df / stats.chi2.ppf(1 - level, df))


def clopper_pearson_upper(x: int, n: int, level: float = Q_LEVEL) -> float:
    """Exact one-sided upper ``level`` bound of a binomial proportion x / n."""
    if x >= n:
        return 1.0
    return float(stats.beta.ppf(level, x + 1, n - x))


def pilot_quantities(
    blocks: Sequence[BlockCodes], *, seeds: Sequence[int] = PILOT_SEEDS
) -> dict[str, Any]:
    """Section 3 item 3 (k*) and section 7 planning inputs from the pilot blocks."""
    seeds = list(seeds)
    if sorted({b.seed for b in blocks}) != sorted(seeds):
        raise ValueError("pilot blocks do not match the pilot seeds")
    if len(blocks) != len(seeds) * len(LOCKS):
        raise ValueError(f"expected {len(seeds) * len(LOCKS)} blocks; got {len(blocks)}")
    ordered = sorted(blocks, key=lambda b: (b.seed, b.lock))
    ksel = select_k_from_pilot([b.z_pre for b in ordered])
    k = ksel.k
    results = [evaluate_block(b, k=k) for b in ordered]
    agg = aggregate(results, n_confirmatory=len(seeds), expected_seeds=seeds)
    n_c = len(agg.seeds_complete)
    n_incomplete = len(seeds) - n_c
    undefined_per_seed = {s: int(np.isnan(agg.y_all[i]).sum()) for i, s in enumerate(agg.seeds_all)}
    out: dict[str, Any] = {
        "k_star": k,
        "median_pr": ksel.median_pr,
        "prs": {f"{b.seed}/{b.lock}": pr for b, pr in zip(ordered, ksel.prs, strict=True)},
        "n_seeds": len(seeds),
        "n_complete": n_c,
        "seeds_complete": agg.seeds_complete,
        "undefined_levels_per_seed": undefined_per_seed,
        "per_level": [vars(c) for c in agg.per_level],
        "y_all": agg.y_all.tolist(),
        "blocks": [
            {
                "seed": r.seed,
                "lock": r.lock,
                "n_valid": r.n_valid,
                "y": r.y,
                "s_split": r.s_split,
                "eigengap_pre": r.eigengap_pre,
            }
            for r in results
        ],
        "q_hat_cp80": clopper_pearson_upper(n_incomplete, len(seeds)),
        "exclusion_model": (
            "level" if all(v <= 1 for v in undefined_per_seed.values()) else "seed"
        ),
    }
    eig = [r.eigengap_pre for r in results] + [v for r in results for v in r.eigengap_post.values()]
    # Section 5 subspace stability (descriptive): Z_pre and every Z_post.
    out["eigengap_share_below_1_1"] = float(np.mean([e < EIGENGAP_UNSTABLE for e in eig]))
    if n_c < MIN_COMPLETE_PILOT_BLOCKS:
        out["sigma_hat"] = None
        out["fileable_as_drafted"] = False
        out["note"] = f"fewer than {MIN_COMPLETE_PILOT_BLOCKS} complete pilot blocks (section 7)"
        return out
    sd, df = anova_residual_sd(agg.y)
    sd_u = sd_ucl_df(sd, df)
    med_split = float(statistics.median(agg.s_split.flatten().tolist()))
    out.update(
        {
            "sigma_hat_point": sd,
            "sigma_hat_df": df,
            "sigma_hat_ucl80": sd_u,
            "median_s_split_complete": med_split,
            "g_hat": med_split / sd_u,
            "g_hat_point_sd": med_split / sd,
            "fileable_as_drafted": True,
        }
    )
    return out


def pilot_quantities_revb(
    blocks: Sequence[BlockCodes], *, seeds: Sequence[int] = PILOT_SEEDS_REVB
) -> dict[str, Any]:
    """Rev. B planning inputs (PI decision 2026-10-25, 19:47) from the fresh pilot.

    Primary: Y = median_c [1 - CKA(Z_post(c), Z_control)] on the
    deterministic soft codes (``metrics.cka_shift``); sigma-hat from the
    two-way ANOVA on complete blocks with its UCL80; guard 0.5 sigma-hat,
    so ``g_hat`` = 0.5 in SD units; q-hat and the exclusion model as
    before. Secondary (descriptive): k at the eigengap and the principal
    shift at that k; the same CKA outcome on the K = 8 Gumbel codes.
    Gate statistics: between-class share and CKA(Z_pre, Z_control) of the
    soft codes, CKA between the soft and the Gumbel Z_pre.
    """
    from bouba_sens.metrics.cka_shift import (
        evaluate_block_cka,
        k_gap_from_pilot,
        linear_cka,
        ratio_at,
    )
    from bouba_sens.readout import code_structure

    seeds = list(seeds)
    if sorted({b.seed for b in blocks}) != sorted(seeds):
        raise ValueError("pilot blocks do not match the pilot seeds")
    if len(blocks) != len(seeds) * len(LOCKS):
        raise ValueError(f"expected {len(seeds) * len(LOCKS)} blocks; got {len(blocks)}")
    ordered = sorted(blocks, key=lambda b: (b.seed, b.lock))
    results = [evaluate_block_cka(b) for b in ordered]
    agg = aggregate(results, n_confirmatory=len(seeds), expected_seeds=seeds)
    n_c = len(agg.seeds_complete)
    undefined = {s: int(np.isnan(agg.y_all[i]).sum()) for i, s in enumerate(agg.seeds_all)}
    kg = k_gap_from_pilot([b.z_pre for b in ordered])
    sec_s = [evaluate_block(b, k=kg.k) for b in ordered]
    sec_g = [
        evaluate_block_cka(b.secondary["soft_k8"]) for b in ordered if "soft_k8" in b.secondary
    ]

    def med(xs: list[float]) -> float | None:
        return float(statistics.median(xs)) if xs else None

    out: dict[str, Any] = {
        "design": "rev. B",
        "n_seeds": len(seeds),
        "n_complete": n_c,
        "seeds_complete": agg.seeds_complete,
        "undefined_levels_per_seed": undefined,
        "per_level": [vars(c) for c in agg.per_level],
        "y_all": agg.y_all.tolist(),
        "q_hat_cp80": clopper_pearson_upper(len(seeds) - n_c, len(seeds)),
        "exclusion_model": "level" if all(v <= 1 for v in undefined.values()) else "seed",
        "guard_sd_units": REVB_GUARD_SD,
        "secondary_k_gap": kg.k,
        "secondary_k_gap_ratios": [r for _, r in kg.per_matrix],
        "secondary_ratio_at_k_gap_post_share_below_1_1": float(
            np.mean([ratio_at(z, kg.k) < 1.1 for b in ordered for z in b.z_post.values()])
        ),
        "secondary_S_at_k_gap_block_y": [r.y for r in sec_s],
        "secondary_gumbel_cka_block_y": [r.y for r in sec_g],
        "gate_between_class_share_pre_median": med(
            [code_structure(b.z_pre, b.probe_labels)["between_class_share"] for b in ordered]
        ),
        "gate_cka_pre_control_median": med([linear_cka(b.z_pre, b.z_control) for b in ordered]),
        "readout_cka_soft_vs_gumbel_pre_median": med(
            [
                linear_cka(b.z_pre, b.secondary["soft_k8"].z_pre)
                for b in ordered
                if "soft_k8" in b.secondary
            ]
        ),
        "blocks": [
            {"seed": r.seed, "lock": r.lock, "n_valid": r.n_valid, "y": r.y} for r in results
        ],
    }
    if n_c < MIN_COMPLETE_PILOT_BLOCKS:
        out.update({"sigma_hat": None, "fileable_as_drafted": False})
        return out
    sd, df = anova_residual_sd(agg.y)
    out.update(
        {
            "sigma_hat_point": sd,
            "sigma_hat_df": df,
            "sigma_hat_ucl80": sd_ucl_df(sd, df),
            "g_hat": REVB_GUARD_SD,
            "fileable_as_drafted": sd > 0,
        }
    )
    return out
