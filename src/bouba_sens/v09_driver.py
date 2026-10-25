"""Driver for the v0.9 decorrelation window: main grid and D variance (sections 3, 4, 7).

Scope (PI decision 2026-10-25): the arms of the main grid and the pilot
variance components of the deficit D. One Phase-1 run is one (seed s,
onset o, arm) cell with window width w = 50 and recovery R:

* world: ``StudyforrestRealWorld(stratified_probe = True)`` (section
  3.4, PI decision 2026-10-25): 512 held-out rows in 32 label-run
  blocks of 16, apportioned to the classes by frame count; training
  windows, donor pools and the check anchors avoid them;
* network: ``builders.build_loop`` with no constellation lock (section
  4), Adam at the constant lr 1e-3 of ``AdaptationLoop`` (checked:
  ``check_learning_rate``), no scheduler, no optimiser reset;
* batches: ``pretrain(o + w + R, batch_size = 16, seed = s)`` through a
  ``DecorrelationSchedule`` (arm, window [o, o + w), training seed s);
  outside the window every arm draws ``world.sample(16, seed = s + t)``;
* seeding: the global torch RNG is re-seeded with s before the loop is
  built, so all arms of a seed share the initial weights and the
  Gumbel noise stream up to the window (outside the window all arms
  are identical, section 3.2);
* outcome: ``decorrelation.fusion_gain`` on the held-out probe (all
  512 held-out rows, the same for every seed and arm; section 3.4).

D(s, o) = FG_joint - FG_decor (section 3.4). ``variance_components``
fits D(s, o) = mu_o + u_s + e_so by ANOVA without replication and
returns the planning SD of section 7: sigma-hat = sqrt(sigma_u^2 +
sigma_e^2), times sqrt(4 / chi2_0.20,4) (= 1.558).

Not in this driver: the fixed-budget grid, the persistence arm, the
secondary arms (D-6, D-7), the train-side fusion gain, D-4 and D-5.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy import stats  # type: ignore[import-untyped]

from bouba_sens.builders import build_loop
from bouba_sens.decorrelation import (
    PROBE_ROWS,
    DecorrelationSchedule,
    DecorrelationWindow,
    FusionGain,
    fusion_gain,
)
from bouba_sens.loop import AdaptationLoop
from bouba_sens.world.base import WorldSimulator

PILOT_SEEDS: tuple[int, ...] = (3101, 3102, 3103, 3104, 3105)
CONFIRMATORY_POOL: tuple[int, ...] = tuple(range(3001, 3045))
ONSETS: tuple[int, ...] = (0, 50, 100, 150, 200)
WIDTH = 50
RECOVERY = 200
BATCH = 16
LEARNING_RATE = 1e-3
MAIN_ARMS: tuple[str, ...] = ("decor", "joint", "plain")
EXPECTED_WORLD = "StudyforrestRealWorld"
UCL_DF = 4  # least favourable df (PI decision 2026-10-24)
UCL_LEVEL = 0.80


def check_world(world: object) -> str:
    """Refuse the stub and a world without the stratified held-out probe."""
    name = type(world).__name__
    if name != EXPECTED_WORLD:
        raise RuntimeError(f"v0.9 needs {EXPECTED_WORLD}; got {name} (technical failure)")
    if not getattr(world, "stratified", False) or getattr(world, "n_probe", 0) != PROBE_ROWS:
        raise RuntimeError("world must hold out the stratified probe (stratified_probe=True)")
    return name


def check_learning_rate(loop: AdaptationLoop) -> None:
    """Section 4: Adam, one parameter group, constant lr 1e-3, no scheduler."""
    opt = loop.opt
    if type(opt) is not torch.optim.Adam:
        raise RuntimeError(f"optimiser must be Adam; got {type(opt).__name__}")
    lrs = {g["lr"] for g in opt.param_groups}
    if lrs != {LEARNING_RATE}:
        raise RuntimeError(f"learning rate must be {LEARNING_RATE}; got {sorted(lrs)}")
    if getattr(loop, "scheduler", None) is not None:
        raise RuntimeError("no learning-rate scheduler is allowed")


@dataclass(frozen=True)
class CellRun:
    seed: int
    onset: int
    arm: str
    recovery: int
    steps: int
    fg: FusionGain
    n_self_donor: int
    wall_s: float


def run_cell(
    world: WorldSimulator,
    seed: int,
    onset: int,
    arm: str,
    *,
    recovery: int = RECOVERY,
    width: int = WIDTH,
) -> CellRun:
    """One Phase-1 run of (seed, onset, arm) and its held-out fusion gain."""
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("call torch.use_deterministic_algorithms(True) first")
    t0 = time.monotonic()
    torch.manual_seed(seed)
    loop = build_loop(world, seed, lock_after=None)
    check_learning_rate(loop)
    sched = DecorrelationSchedule(
        DecorrelationWindow(onset, width),
        arm,
        seed=seed,
        world=world,  # type: ignore[arg-type]
    )
    steps = onset + width + recovery
    loop.pretrain(steps, batch_size=BATCH, seed=seed, decorrelation=sched)
    check_learning_rate(loop)
    fg = fusion_gain(loop, train_seed=seed)
    return CellRun(seed, onset, arm, recovery, steps, fg, sched.n_self_donor, time.monotonic() - t0)


def run_main_grid(
    world: WorldSimulator,
    seed: int,
    *,
    onsets: Sequence[int] = ONSETS,
    arms: Sequence[str] = MAIN_ARMS,
    **cell_kw: Any,
) -> list[CellRun]:
    return [run_cell(world, seed, o, a, **cell_kw) for o in onsets for a in arms]


def deficit_matrix(
    cells: Sequence[CellRun], seeds: Sequence[int], onsets: Sequence[int] = ONSETS
) -> np.ndarray:
    """D(s, o) = FG_joint - FG_decor, rows = seeds, columns = onsets."""
    fg = {(c.seed, c.onset, c.arm): c.fg.fusion_gain for c in cells}
    return np.array(
        [[fg[(s, o, "joint")] - fg[(s, o, "decor")] for o in onsets] for s in seeds],
        dtype=np.float64,
    )


def variance_components(d: np.ndarray) -> dict[str, float]:
    """Two-way ANOVA without replication and the section 7 planning SD."""
    n, m = d.shape
    grand = d.mean()
    ms_seed = m * float(((d.mean(1) - grand) ** 2).sum()) / (n - 1)
    resid = d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + grand
    df_res = (n - 1) * (m - 1)
    ms_res = float((resid**2).sum()) / df_res
    var_e = ms_res
    var_u = max(0.0, (ms_seed - ms_res) / m)
    sd = math.sqrt(var_u + var_e)
    mult = math.sqrt(UCL_DF / stats.chi2.ppf(1 - UCL_LEVEL, UCL_DF))
    return {
        "ms_seed": ms_seed,
        "ms_residual": ms_res,
        "df_residual": float(df_res),
        "sigma_e2": var_e,
        "sigma_u2": var_u,
        "sigma_hat": sd,
        "ucl_multiplier": mult,
        "sigma_plan": sd * mult,
    }
