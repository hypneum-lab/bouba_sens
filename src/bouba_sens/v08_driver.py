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
# Rev. C (PI decision 2026-10-28, about 19:05): straight-through emission
# with the learning gate; encoder representations primary, sender logits
# secondary; class-balanced probe; drift-floor control at the end locks.
PILOT_SEEDS_REVC: tuple[int, ...] = tuple(range(2211, 2221))
REVC_EMISSION = "straight-through"
REVC_PRIMARY = "encoder"
REVC_SECONDARY: tuple[tuple[str, int], ...] = (("logits", 0),)
REVC_FLOOR_LOCKS: tuple[int, ...] = (25, 175)
REVC_FLOOR_OFFSET = 500_000  # control' Phase-2 seed = s + 500 000 (namespaced)
REVC_FLOOR_RATIO = 2.0
REVC_SHARE_MIN = 0.05
REVC_MIN_SEEDS = 8
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
    world: WorldSimulator,
    seed: int,
    lock: int,
    *,
    seed_namespace: int | None = None,
    emission: str = "argmax-bridge",
) -> Checkpoint:
    """Phase-1 checkpoint of block (seed, lock): 200 steps, batch 16, seed s."""
    torch.manual_seed(seed)
    loop = build_loop(
        world, seed, lock_after=lock, seed_namespace=seed_namespace, emission=emission
    )
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
    elif design == "revc":
        return _run_block_revc(world, seed, lock, seed_namespace=seed_namespace, **capture_kw)
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


def revc_probe_rows(world: Any, seed: int, seed_namespace: int) -> torch.Tensor:
    """Class-balanced probe rows of every block of seed s (rev. C, gate 3).

    ``probe.class_balanced_probe_indices`` on the full label sequence: 8
    windows of 16 frames per class, each inside one label run, drawn with
    the generator seed ``ns_seed(code, s, phase2_probe, 0)``.
    """
    from bouba_sens.probe import class_balanced_probe_indices
    from bouba_sens.seeds import STREAMS, ns_seed

    n = int(world.n_train + world.n_probe)
    labels = world.gather(torch.arange(n)).label
    return class_balanced_probe_indices(
        labels, seed=ns_seed(seed_namespace, seed, STREAMS["phase2_probe"], 0)
    )


def check_world_revc(world: object) -> str:
    """Rev. C needs the real bridge with the stratified held-out rows (learning gate)."""
    name = check_world(world)
    if not getattr(world, "stratified", False):
        raise RuntimeError("rev. C needs StudyforrestRealWorld(stratified_probe=True)")
    return name


def _run_block_revc(
    world: Any, seed: int, lock: int, *, seed_namespace: int | None = None, **capture_kw: Any
) -> BlockRun:
    """Rev. C block: straight-through networks, encoder / logit readouts, drift floor."""
    from bouba_sens.learning_gate import heldout_task_metrics

    require_determinism()
    check_world_revc(world)
    ns = revb_namespace(seed) if seed_namespace is None else seed_namespace
    t0 = time.monotonic()
    ckpt = train_checkpoint(world, seed, lock, seed_namespace=ns, emission=REVC_EMISSION)

    def make_loop() -> AdaptationLoop:
        return build_loop(world, seed, lock_after=lock, seed_namespace=ns, emission=REVC_EMISSION)

    probe = world.gather(revc_probe_rows(world, seed, ns))
    heldout, _ = world.probe_sample_with_indices(world.n_probe, 0)
    kw = {
        "probe_demod": REVC_PRIMARY,
        "probe_gumbel_draws": 0,
        "extra_readouts": REVC_SECONDARY,
        "probe_sample": probe,
        "floor_seed": seed + REVC_FLOOR_OFFSET if lock in REVC_FLOOR_LOCKS else None,
        "control_metric": lambda loop: heldout_task_metrics(loop, heldout),
        **capture_kw,
    }
    block = capture_block(make_loop, ckpt, seed=seed, lock=lock, **kw)
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
        "z_control_alt": block.z_control_alt,
        "control_metrics": block.control_metrics,
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
        z_control_alt=d.get("z_control_alt"),
        control_metrics=dict(d.get("control_metrics") or {}),
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
        total_variance,
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
    sec_s = [evaluate_block(b, k=kg.k) for b in ordered] if kg.k > 0 else []
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
        "secondary_k_gap": kg.k or None,
        "secondary_k_gap_n_matrices": len(kg.per_matrix),
        "secondary_k_gap_ratios": [r for _, r in kg.per_matrix],
        "soft_total_variance_pre_median": med([total_variance(b.z_pre) for b in ordered]),
        "soft_total_variance_post_median": med(
            [total_variance(z) for b in ordered for z in b.z_post.values()]
        ),
        "gumbel_total_variance_pre_median": med(
            [
                total_variance(b.secondary["soft_k8"].z_pre)
                for b in ordered
                if "soft_k8" in b.secondary
            ]
        ),
        "gumbel_between_class_share_pre_median": med(
            [
                code_structure(b.secondary["soft_k8"].z_pre, b.probe_labels)["between_class_share"]
                for b in ordered
                if "soft_k8" in b.secondary
            ]
        ),
        "secondary_ratio_at_k_gap_post_share_below_1_1": float(
            np.mean([ratio_at(z, kg.k) < 1.1 for b in ordered for z in b.z_post.values()])
        )
        if kg.k > 0
        else None,
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


def _bootstrap_ci(x: Sequence[float], *, level: float = 0.90, n: int = 10_000) -> list[float]:
    """Percentile CI of the mean of ``x`` over seeds (generator ``20261028 + 808``)."""
    a = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(20261028 + 808)
    means = a[rng.integers(0, len(a), size=(n, len(a)))].mean(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return [float(lo), float(hi)]


def _revc_readout(
    blocks: Sequence[BlockCodes], seeds: Sequence[int], dims: Sequence[int]
) -> dict[str, Any]:
    """Gates 2-5 and the planning inputs of one readout (rev. C)."""
    from bouba_sens.metrics.revc_shift import block_failures, drift_floor, evaluate_block_revc
    from bouba_sens.readout import code_structure

    ordered = sorted(blocks, key=lambda b: (b.seed, b.lock))
    evals = [evaluate_block_revc(b, dims) for b in ordered]
    results = [r for r, _ in evals]
    agg = aggregate(results, n_confirmatory=len(seeds), expected_seeds=seeds)
    n_c = len(agg.seeds_complete)
    undefined = {s: int(np.isnan(agg.y_all[i]).sum()) for i, s in enumerate(agg.seeds_all)}
    fails = {m: sum(f[m] for _, f in evals) for m in evals[0][1]}
    share: dict[int, list[float]] = {}
    pre_bad = 0
    for b in ordered:
        if block_failures(b.z_pre, dims):
            pre_bad += 1
        share.setdefault(b.seed, []).append(
            code_structure(b.z_pre, b.probe_labels)["between_class_share"]
        )
    share_med = {s: float(statistics.median(v)) for s, v in share.items()}
    floor: dict[int, dict[int, dict[str, Any]]] = {}
    for b, r in zip(ordered, results, strict=True):
        if b.z_control_alt is not None:
            floor.setdefault(b.lock, {})[b.seed] = drift_floor(b, r, dims)
    floor_summary: dict[str, Any] = {}
    floor_ok = True
    for lock, per in sorted(floor.items()):
        rs = [v["R"] for v in per.values() if v["R"] is not None]
        med_r = float(statistics.median(rs)) if rs else None
        floor_summary[str(lock)] = {
            "per_seed": {str(s): v for s, v in per.items()},
            "median_R": med_r,
            "n_undefined_R": sum(v["R"] is None for v in per.values()),
        }
        floor_ok = floor_ok and med_r is not None and med_r >= REVC_FLOOR_RATIO
    locks = sorted(floor)
    if len(locks) == 2:
        lo, hi = locks
        common = [s for s in seeds if floor[lo].get(s, {}).get("F") is not None
                  and floor[hi].get(s, {}).get("F") is not None]  # fmt: skip
        contrast = [floor[hi][s]["F"] - floor[lo][s]["F"] for s in common]
        floor_summary["control_only_contrast"] = {
            "mean": float(np.mean(contrast)) if contrast else None,
            "ci90_seed_bootstrap": _bootstrap_ci(contrast) if len(contrast) > 1 else None,
            "n_seeds": len(contrast),
        }
    out: dict[str, Any] = {
        "n_complete": n_c,
        "seeds_complete": agg.seeds_complete,
        "undefined_levels_per_seed": undefined,
        "block_failures_by_modality": fails,
        "z_pre_invalid_blocks": pre_bad,
        "per_level": [vars(c) for c in agg.per_level],
        "y_all": agg.y_all.tolist(),
        "between_class_share_pre_median_per_seed": share_med,
        "drift_floor": floor_summary,
        "gate2_nondegenerate": n_c >= REVC_MIN_SEEDS,
        "gate3_structure": sum(v >= REVC_SHARE_MIN for v in share_med.values()) >= REVC_MIN_SEEDS,
        "gate4_drift_floor": floor_ok,
        "q_hat_cp80": clopper_pearson_upper(len(seeds) - n_c, len(seeds)),
        "exclusion_model": "level" if all(v <= 1 for v in undefined.values()) else "seed",
        "blocks": [
            {"seed": r.seed, "lock": r.lock, "n_valid": r.n_valid, "y": r.y} for r in results
        ],
    }
    if n_c < MIN_COMPLETE_PILOT_BLOCKS:
        out.update({"sigma_hat": None, "gate5_y_nondegenerate": False})
        return out
    sd, df = anova_residual_sd(agg.y)
    out.update(
        {
            "sigma_hat_point": sd,
            "sigma_hat_df": df,
            "sigma_hat_ucl80": sd_ucl_df(sd, df),
            "gate5_y_nondegenerate": sd > 0,
        }
    )
    return out


def pilot_quantities_revc(
    blocks: Sequence[BlockCodes], *, seeds: Sequence[int] = PILOT_SEEDS_REVC
) -> dict[str, Any]:
    """Rev. C pilot summary: learning gate, gates 2-5 per readout, planning inputs.

    The N rule (gate 5, second half) is applied to the primary readout by
    ``scripts/pilots/derive_v08_pilot.py`` (keys ``g_hat``, ``q_hat_cp80``,
    ``exclusion_model``, ``fileable_as_drafted`` at top level).
    """
    from bouba_sens.learning_gate import learning_gate
    from bouba_sens.metrics.revc_shift import ENCODER_DIMS, LOGIT_DIMS

    seeds = list(seeds)
    if sorted({b.seed for b in blocks}) != sorted(seeds):
        raise ValueError("pilot blocks do not match the pilot seeds")
    if len(blocks) != len(seeds) * len(LOCKS):
        raise ValueError(f"expected {len(seeds) * len(LOCKS)} blocks; got {len(blocks)}")
    per_seed: dict[int, list[dict[str, Any]]] = {}
    for b in blocks:
        per_seed.setdefault(b.seed, []).append(b.control_metrics)
    gate1 = learning_gate(per_seed, need_over_single=False)
    primary = _revc_readout(blocks, seeds, ENCODER_DIMS)
    secondary = _revc_readout([b.secondary["logits_k0"] for b in blocks], seeds, LOGIT_DIMS)
    gates = {
        "gate1_learning": gate1["passed"],
        **{k: v for k, v in primary.items() if k.startswith("gate")},
    }
    out: dict[str, Any] = {
        "design": "rev. C",
        "emission": REVC_EMISSION,
        "n_seeds": len(seeds),
        "learning_gate": gate1,
        "primary_encoder": primary,
        "secondary_logits": secondary,
        "gates_primary": gates,
        "guard_sd_units": REVB_GUARD_SD,
        "g_hat": REVB_GUARD_SD,
        "q_hat_cp80": primary["q_hat_cp80"],
        "exclusion_model": primary["exclusion_model"],
        "sigma_hat_ucl80": primary.get("sigma_hat_ucl80"),
        "fileable_as_drafted": bool(primary.get("gate5_y_nondegenerate")),
    }
    return out
