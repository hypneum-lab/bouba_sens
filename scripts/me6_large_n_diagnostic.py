"""Me6 large-n — exploratory diagnostics D1-D4 (amendment 2026-10-31).

`docs/prereg/me6-large-n/amendment-2026-10-31-diagnostic.md` defines the
statistics; this script implements them. Exploratory: it decides nothing
and changes no preregistered reading of ADR-0026.

Notation (as in `me6_paired_null.py`): for trio k, pair p = (a, b) with
a < b and probe item i, d(k, p, i) = s[a, b, i] - s[b, a, i] in {-1, 0, 1}
and Delta(k, p) = mean_i d. With a block length b > 1 (mock world, the
committed ADR-0026 blocks) the items are grouped into consecutive blocks
and every flip acts on block sums, exactly as `paired_null(block=b)`.

Subcommands:

- `d1 --root GRID --world W --block B` — split-half replicability of
  Delta at n = 4096 (Gaussian: even / odd items; mock: first / second half
  of the blocks) and overdispersion phi_n at every size, each against the
  preregistered paired sign flip.
- `d2 --root GRID --world W --blocks 64:1,...` — joint transpose flip: one
  sign per (trio, item or block) shared by the 10 pairs; median Me6.
- `d3 --root GRID [--size 4096]` — direction consistency of Delta across
  trios (trio-level sign flip) and within seed (exact, 2^6 patterns).
- `reprobe --root GRID --part a|b --out DIR [--shard i/N]` — D4 driver:
  v1 zero-masking probe at n = 4096 on fresh items of the saved lesion
  states (a) or of the pre-lesion states (b).
- `validate --root GRID --cells NAME,...` — reprobe with the original
  probe seed must reproduce `per_query_successes_n4096.json`; with
  `--lesion-cell` a T1 init + lesion phase must reproduce its `model.pt`.
- `d4 --root DIR` — Me6 with the ADR-0026 item-flip null, D1 and D3 on
  reprobe outputs.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from me6_paired_null import (
    PAIR_NAMES,
    Trio,
    analyse,
    discover_trios,
    pair_differences,
    stack_successes,
    summarise,
    tail,
)

SCHEMA = "bouba_sens.me6_large_n_diagnostic.v1"
DIAG_SEED = 31_102_026
DIAG_REPLICATES = 20_000
SIZES: tuple[int, ...] = (64, 256, 1024, 4096)
D1_SPLIT_SIZE = 4096
# D4 probe seed: 5 106 000 + 10 * (seed - 400) + trio (amendment §D4).
REPROBE_BASE = 5_106_000
REPROBE_SIZE = 4096
REPROBE_FILE = "per_query_successes_reprobe_n4096.json"
# Replicates per RNG chunk; fixed so that a seed gives the same draws anywhere.
_CHUNK = 100
# Distinct streams per diagnostic (default_rng([seed, tag, size, start])).
_TAG_D1, _TAG_D2, _TAG_D3, _TAG_PHI = 1, 2, 3, 4


def successes_file(size: int) -> str:
    return f"per_query_successes_n{size}.json"


def _meta(**params: Any) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "params": params,
    }


def block_sums(differences: np.ndarray, block: int) -> np.ndarray:
    """`(..., n)` -> `(..., ceil(n / block))` sums of consecutive items.

    int8 while |sum| <= block <= 127 (keeps the flip arrays small), else int16.
    """
    if block < 1:
        raise ValueError("block must be >= 1")
    dtype = np.int8 if block <= 127 else np.int16
    n = differences.shape[-1]
    if block == 1:
        return differences.astype(dtype)
    n_blocks = -(-n // block)
    pad = n_blocks * block - n
    d = np.pad(differences.astype(np.int32), [(0, 0)] * (differences.ndim - 1) + [(0, pad)])
    return d.reshape(*differences.shape[:-1], n_blocks, block).sum(axis=-1).astype(dtype)


def _signs(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.integers(0, 2, size=shape, dtype=np.int8) * 2 - 1


# ---------------------------------------------------------------- phi / D1


def phi(units: np.ndarray) -> float:
    """Overdispersion: mean over (trio, pair) with variance of z^2.

    `units` is `(K, 10, m)` (block sums; items when block = 1). For each
    entry z^2 = (sum_j u_j)^2 / sum_j u_j^2, whose expectation is exactly 1
    under independent sign flips of the u_j. Entries with sum u^2 = 0 are
    skipped.
    """
    q = (units.astype(np.float64) ** 2).sum(axis=-1)
    s = units.sum(axis=-1, dtype=np.int64).astype(np.float64)
    keep = q > 0
    if not keep.any():
        return float("nan")
    return float((s[keep] ** 2 / q[keep]).mean())


def _phi_null(units: np.ndarray, *, replicates: int, seed: int, size: int) -> np.ndarray:
    q = (units.astype(np.float64) ** 2).sum(axis=-1)
    keep = q > 0
    out = np.empty(replicates)
    start = 0
    while start < replicates:
        g = min(_CHUNK, replicates - start)
        rng = np.random.default_rng([seed, _TAG_PHI, size, start])
        s = (_signs(rng, (g, *units.shape)) * units).sum(axis=-1, dtype=np.int64)
        out[start : start + g] = (s[:, keep].astype(np.float64) ** 2 / q[keep]).mean(axis=-1)
        start += g
    return out


def split_halves(units: np.ndarray, *, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Index masks of the two halves: even / odd items (block 1), else halves."""
    m = units.shape[-1]
    idx = np.arange(m)
    if block == 1:
        return idx % 2 == 0, idx % 2 == 1
    return idx < m // 2, idx >= m // 2


def _pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation along the last axis (nan where a side is constant)."""
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean(axis=-1, keepdims=True)
    den = np.sqrt((xc**2).sum(axis=-1) * (yc**2).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (xc * yc).sum(axis=-1) / den


def split_half_stats(units: np.ndarray, *, block: int) -> dict[str, float]:
    """r over all (trio, pair) and r over each trio's argmax pair."""
    first, second = split_halves(units, block=block)
    a = units[..., first].sum(axis=-1).astype(np.float64)
    b = units[..., second].sum(axis=-1).astype(np.float64)
    full = np.abs(a + b)
    arg = full.argmax(axis=-1)
    rows = np.arange(units.shape[0])
    return {
        "r_all": float(_pearson(a.reshape(-1), b.reshape(-1))),
        "r_argmax": float(_pearson(a[rows, arg], b[rows, arg])),
    }


def _split_null(units: np.ndarray, *, block: int, replicates: int, seed: int) -> np.ndarray:
    first, second = split_halves(units, block=block)
    out = np.empty((replicates, 2))
    start = 0
    while start < replicates:
        g = min(_CHUNK, replicates - start)
        rng = np.random.default_rng([seed, _TAG_D1, units.shape[-1], start])
        flipped = _signs(rng, (g, *units.shape)) * units
        a = flipped[..., first].sum(axis=-1, dtype=np.int64).astype(np.float64)
        b = flipped[..., second].sum(axis=-1, dtype=np.int64).astype(np.float64)
        out[start : start + g, 0] = _pearson(a.reshape(g, -1), b.reshape(g, -1))
        arg = np.abs(a + b).argmax(axis=-1)  # (g, K)
        aa = np.take_along_axis(a, arg[..., None], axis=-1)[..., 0]
        bb = np.take_along_axis(b, arg[..., None], axis=-1)[..., 0]
        out[start : start + g, 1] = _pearson(aa, bb)
        start += g
    return out


def _breakdown(trios: list[Trio], units: np.ndarray) -> dict[str, Any]:
    """Observed phi per pair, timing and SNR (descriptive)."""
    out: dict[str, Any] = {"pair": {}, "timing": {}, "snr": {}}
    for p, name in enumerate(PAIR_NAMES):
        out["pair"][name] = phi(units[:, p : p + 1, :])
    for key, attr in (("timing", "timing"), ("snr", "snr")):
        for value in sorted({getattr(t, attr) for t in trios}):
            sel = [k for k, t in enumerate(trios) if getattr(t, attr) == value]
            out[key][value] = phi(units[sel])
    return out


def d1(
    root: Path,
    *,
    blocks: dict[int, int],
    replicates: int = DIAG_REPLICATES,
    seed: int = DIAG_SEED,
    split_size: int = D1_SPLIT_SIZE,
    files: dict[int, str] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    result = _meta(root=str(root), blocks=blocks, replicates=replicates, seed=seed)
    result["phi"] = {}
    for size, block in blocks.items():
        name = (files or {}).get(size, successes_file(size))
        trios, _ = discover_trios(root, successes_file=name)
        units = block_sums(pair_differences(stack_successes(trios)), block)
        observed = phi(units)
        null = _phi_null(units, replicates=replicates, seed=seed, size=size)
        result["phi"][str(size)] = {
            "trios": len(trios),
            "block": block,
            "observed": observed,
            "null": summarise(null),
            **tail(null, observed),
            "breakdown": _breakdown(trios, units),
        }
        if size == split_size:
            obs = split_half_stats(units, block=block)
            null2 = _split_null(units, block=block, replicates=replicates, seed=seed)
            result["split_half"] = {
                "n": size,
                "block": block,
                "halves": "even/odd items" if block == 1 else "first/second half of blocks",
                **{
                    key: {
                        "observed": obs[key],
                        "null": summarise(null2[:, j][np.isfinite(null2[:, j])]),
                        **tail(null2[:, j][np.isfinite(null2[:, j])], obs[key]),
                    }
                    for j, key in enumerate(("r_all", "r_argmax"))
                },
            }
    result["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return result


# --------------------------------------------------------------------- D2


def transpose_null_medians(
    units: np.ndarray, *, n_items: int, replicates: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """`(per_trio (B, K), median (B,))` Me6 under the joint transpose flip."""
    n_trios, _, m = units.shape
    per_trio = np.empty((replicates, n_trios))
    start = 0
    while start < replicates:
        g = min(_CHUNK, replicates - start)
        rng = np.random.default_rng([seed, _TAG_D2, n_items, start])
        signs = _signs(rng, (g, n_trios, 1, m))
        s = (signs * units).sum(axis=-1, dtype=np.int64)  # (g, K, 10)
        per_trio[start : start + g] = np.abs(s).max(axis=-1) / n_items
        start += g
    return per_trio, np.median(per_trio, axis=1)


def pair_flip_medians(
    units: np.ndarray, *, n_items: int, replicates: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Same as `transpose_null_medians` with independent signs per pair."""
    n_trios, n_pairs, m = units.shape
    per_trio = np.empty((replicates, n_trios))
    start = 0
    while start < replicates:
        g = min(_CHUNK, replicates - start)
        rng = np.random.default_rng([seed, _TAG_D2, n_items, start, 1])
        signs = _signs(rng, (g, n_trios, n_pairs, m))
        s = (signs * units).sum(axis=-1, dtype=np.int64)
        per_trio[start : start + g] = np.abs(s).max(axis=-1) / n_items
        start += g
    return per_trio, np.median(per_trio, axis=1)


def d2(
    root: Path,
    *,
    blocks: dict[int, int],
    replicates: int = DIAG_REPLICATES,
    seed: int = DIAG_SEED,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    result = _meta(root=str(root), blocks=blocks, replicates=replicates, seed=seed)
    result["null"] = (
        "joint transpose flip: one sign per (trio, item or block), shared by the 10 pairs"
    )
    result["rows"] = {}
    for size, block in blocks.items():
        trios, _ = discover_trios(root, successes_file=successes_file(size))
        diffs = pair_differences(stack_successes(trios))
        units = block_sums(diffs, block)
        observed = float(np.median(np.abs(diffs.mean(axis=-1)).max(axis=-1)))
        _, medians = transpose_null_medians(
            units, n_items=diffs.shape[-1], replicates=replicates, seed=seed
        )
        result["rows"][str(size)] = {
            "trios": len(trios),
            "block": block,
            "observed_median": observed,
            "null": summarise(medians),
            **tail(medians, observed),
        }
    result["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return result


# --------------------------------------------------------------------- D3


def exact_sign_p(values: np.ndarray) -> float:
    """Exact p of |mean| under all 2^m sign patterns (identity included)."""
    m = values.shape[0]
    patterns = np.array(list(product((-1, 1), repeat=m)), dtype=np.float64)  # (2^m, m)
    stats = np.abs(patterns @ values) / m
    observed = abs(values.sum()) / m
    return float((stats >= observed - 1e-12).sum() / patterns.shape[0])


def direction_null(
    delta: np.ndarray, *, replicates: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Trio-level sign flip of Delta `(K, 10)`: `(S_p (B, 10), S (B,))`."""
    out = np.empty((replicates, delta.shape[1]))
    start = 0
    while start < replicates:
        g = min(_CHUNK, replicates - start)
        rng = np.random.default_rng([seed, _TAG_D3, start])
        signs = _signs(rng, (g, *delta.shape)).astype(np.float64)
        out[start : start + g] = np.abs((signs * delta).mean(axis=1))
        start += g
    return out, out.max(axis=1)


def d3(
    root: Path,
    *,
    size: int = 4096,
    replicates: int = DIAG_REPLICATES,
    seed: int = DIAG_SEED,
    successes: str | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    name = successes or successes_file(size)
    trios, _ = discover_trios(root, successes_file=name)
    delta = pair_differences(stack_successes(trios)).mean(axis=-1)  # (K, 10)
    s_obs = np.abs(delta.mean(axis=0))
    null_p, null_max = direction_null(delta, replicates=replicates, seed=seed)
    lead = int(s_obs.argmax())
    lead_sign = np.sign(delta[:, lead].mean())
    arg = np.abs(delta).argmax(axis=1)
    by_seed: dict[str, Any] = {}
    for seed_value in sorted({t.seed for t in trios}, key=int):
        sel = [k for k, t in enumerate(trios) if t.seed == seed_value]
        pvals = {PAIR_NAMES[p]: exact_sign_p(delta[sel, p]) for p in range(len(PAIR_NAMES))}
        by_seed[seed_value] = {
            "trios": len(sel),
            "patterns": 2 ** len(sel),
            "p_exact": pvals,
            "min_p": min(pvals.values()),
        }
    result = _meta(root=str(root), successes_file=name, replicates=replicates, seed=seed)
    result.update(
        {
            "trios": len(trios),
            "max_pair": {
                "observed": float(s_obs.max()),
                "null": summarise(null_max),
                **tail(null_max, float(s_obs.max())),
            },
            "per_pair": {
                PAIR_NAMES[p]: {
                    "mean_delta": float(delta[:, p].mean()),
                    "S": float(s_obs[p]),
                    "n_positive": int((delta[:, p] > 0).sum()),
                    "n_negative": int((delta[:, p] < 0).sum()),
                    **tail(null_p[:, p], float(s_obs[p])),
                }
                for p in range(len(PAIR_NAMES))
            },
            "leading_pair": {
                "pair": PAIR_NAMES[lead],
                "same_sign_trios": int((np.sign(delta[:, lead]) == lead_sign).sum()),
            },
            "argmax_sign_table": [
                {
                    "trio": t.label,
                    "pair": PAIR_NAMES[int(arg[k])],
                    "delta": float(delta[k, arg[k]]),
                }
                for k, t in enumerate(trios)
            ],
            "within_seed": by_seed,
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }
    )
    return result


# --------------------------------------------------------------------- D4


def reprobe_seed(seed: int, trio: int) -> int:
    """Amendment §D4: 5 106 000 + 10 * (seed - 400) + trio."""
    if not 0 <= trio <= 5:
        raise ValueError(f"trio must be in 0..5, got {trio}")
    return REPROBE_BASE + 10 * (seed - 400) + trio


def _cell_info(cell: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(cell / "config.yaml")
    meta = json.loads((cell / "metadata.json").read_text())
    seed = int(cell.name.split("_")[0].removeprefix("seed"))
    trio = int(meta["probe_sizes_seed"]) - seed * 10000 - 6000
    mode = str(cfg.get("world_seed_mode", "legacy"))
    return {
        "seed": seed,
        "trio": trio,
        "lesion_seed": int(cfg["seed"]),
        "timing": str(cfg["timing"]),
        "modality": str(cfg["modality"]),
        "world": str(cfg.get("world", "gaussian")),
        "world_seed": int(cfg["world_seed"]) if mode == "cell" else 0,
        "world_seed_mode": mode,
        "original_probe_seed": int(meta["probe_sizes_seed"]),
        "snr_init": float(cfg["snr_init"]),
        "snr_floor": float(cfg["snr_floor"]),
        "k_steps": int(cfg["k_steps"]),
        "steps": int(cfg["steps"]),
    }


def build_loop(info: dict[str, Any]) -> Any:
    """Rebuild a lesion cell's network exactly as `bouba-sens lesion` does.

    Same construction sequence and seeds as `cli.lesion` (grid flags:
    gating hard, tau 1.0, argmax-bridge, no locks), before any restore and
    before the lesion phase: this is the T1 initial state.
    """
    from track_p.multiplexer import GammaThetaMultiplexer

    from bouba_sens.cli import _build_world, _seed_everything
    from bouba_sens.encoders import (
        AudioEncoder,
        ForceEncoder,
        GravityEncoder,
        TactileEncoder,
        VisionEncoder,
    )
    from bouba_sens.head import IntegrationHead
    from bouba_sens.loop import AdaptationLoop
    from bouba_sens.nerve import CrossModalNerve
    from bouba_sens.sensory import SensoryWML

    _seed_everything(info["lesion_seed"])
    world_obj = _build_world(info["world"], seed=info["world_seed"])
    mux = GammaThetaMultiplexer(seed=0, constellation_lock_after=None)
    em = "argmax-bridge"
    sensories = {
        "audio": SensoryWML(0, "audio", AudioEncoder(), mux, seed=1, emission=em),
        "vision": SensoryWML(1, "vision", VisionEncoder(), mux, seed=2, emission=em),
        "tactile": SensoryWML(2, "tactile", TactileEncoder(), mux, seed=3, emission=em),
        "gravity": SensoryWML(3, "gravity", GravityEncoder(), mux, seed=4, emission=em),
        "force": SensoryWML(4, "force", ForceEncoder(), mux, seed=5, emission=em),
    }
    nerve = CrossModalNerve(
        mux,
        seed=0,
        transducer_gating="hard",
        gumbel_tau=1.0,
        codebook_lock_after=None,
        transducer_gating_schedule=None,
        transducer_gating_target=None,
    )
    head = IntegrationHead(n_classes=4)
    return AdaptationLoop(world_obj, mux, sensories, nerve, head)


def load_model_pt(loop: Any, path: Path) -> None:
    import torch

    state = torch.load(path, weights_only=True)
    loop.mux.load_state_dict(state["mux"])
    loop.nerve.load_state_dict(state["nerve"])
    loop.head.load_state_dict(state["head"])
    for m, sd in state["sensory"].items():
        loop.sensories[m].load_state_dict(sd)


def probe(loop: Any, *, items_seed: int, rng_seed: int, size: int) -> dict[str, list[int]]:
    """v1 zero-masking probe, as `cli._write_probe_sizes` (forked, reseeded RNG)."""
    import torch

    from bouba_sens.sensory import MODALITIES

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(rng_seed)
        return {m: loop.query_successes(m, batch_size=size, seed=items_seed) for m in MODALITIES}


def _cells(root: Path) -> list[Path]:
    from me6_paired_null import CELL_RE

    return [c for c in sorted(root.iterdir()) if c.is_dir() and CELL_RE.match(c.name)]


def reprobe(root: Path, *, part: str, out: Path, shard: tuple[int, int] = (0, 1)) -> int:
    """D4 (a) lesion states / (b) pre-lesion states, fresh items, n = 4096."""
    import pickle

    from bouba_sens.sensory import MODALITIES

    index, total = shard
    done = 0
    for j, cell in enumerate(_cells(root)):
        if j % total != index:
            continue
        target = out / cell.name / REPROBE_FILE
        if target.exists():
            continue
        info = _cell_info(cell)
        ps = reprobe_seed(info["seed"], info["trio"])
        loop = build_loop(info)
        rng_seed = ps
        if part == "a":
            load_model_pt(loop, cell / "model.pt")
            state = "lesion state (model.pt)"
        elif info["timing"] == "T2":
            ckpt = root / f"phase1_seed{info['seed']}" / "checkpoint.pkl"
            with ckpt.open("rb") as fh:
                loop.restore(pickle.load(fh))
            rng_seed = ps + MODALITIES.index(info["modality"])
            state = f"phase-1 checkpoint {ckpt.parent.name}"
        else:
            state = "T1 initial state (lesion seed, zero lesion steps)"
        successes = probe(loop, items_seed=ps, rng_seed=rng_seed, size=REPROBE_SIZE)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(successes))
        (target.parent / "reprobe_meta.json").write_text(
            json.dumps(
                {**info, "part": part, "state": state, "items_seed": ps, "rng_seed": rng_seed},
                indent=2,
            )
        )
        done += 1
    return done


def validate(root: Path, *, cells: Sequence[str], lesion_cell: str | None) -> dict[str, Any]:
    """Check the rebuild: original probe seed -> saved n4096 file (bit for bit)."""
    out: dict[str, Any] = {"reprobe_original_seed": {}}
    for name in cells:
        cell = root / name
        info = _cell_info(cell)
        loop = build_loop(info)
        load_model_pt(loop, cell / "model.pt")
        ps = info["original_probe_seed"]
        got = probe(loop, items_seed=ps, rng_seed=ps, size=4096)
        saved = json.loads((cell / successes_file(4096)).read_text())
        out["reprobe_original_seed"][name] = {
            m: int(sum(a != b for a, b in zip(got[m], saved[m], strict=True))) for m in got
        }
    if lesion_cell is not None:
        import torch

        from bouba_sens.lesion import LesionSpec, m2_snr_schedule

        cell = root / lesion_cell
        info = _cell_info(cell)
        if info["timing"] != "T1":
            raise ValueError("--lesion-cell must be a T1 cell")
        loop = build_loop(info)

        def schedule(step: int) -> float:
            return m2_snr_schedule(
                step, snr_init=info["snr_init"], snr_floor=info["snr_floor"], k=info["k_steps"]
            )

        spec = LesionSpec(modality=info["modality"], mode="M2", timing="T1", schedule=schedule)
        loop.lesion_phase(spec, steps=info["steps"], batch_size=16, seed=info["lesion_seed"])
        saved = torch.load(cell / "model.pt", weights_only=True)
        post = loop.snapshot()
        mism: list[str] = []
        for key, mine in (
            ("mux", post.mux_state),
            ("nerve", post.nerve_state),
            ("head", post.head_state),
        ):
            mism += [f"{key}.{k}" for k, v in mine.items() if not torch.equal(v, saved[key][k])]
        for m, sd in post.sensory_states.items():
            mism += [
                f"sensory.{m}.{k}"
                for k, v in sd.items()
                if not torch.equal(v, saved["sensory"][m][k])
            ]
        out["t1_lesion_rebuild"] = {"cell": lesion_cell, "mismatched_tensors": mism}
    return out


def d4(root: Path, *, replicates: int = DIAG_REPLICATES, seed: int = DIAG_SEED) -> dict[str, Any]:
    """Me6 with the ADR-0026 item-flip null (block 1), plus D1 and D3."""
    trios, incomplete = discover_trios(root, successes_file=REPROBE_FILE)
    paired = analyse(
        trios,
        replicates=replicates,
        seed=seed,
        with_binomial=False,
        incomplete=incomplete,
        root=root,
        successes_file=REPROBE_FILE,
    )
    paired.pop("cells", None)
    return {
        "item_flip": paired,
        "d1": d1(
            root,
            blocks={REPROBE_SIZE: 1},
            replicates=replicates,
            seed=seed,
            files={REPROBE_SIZE: REPROBE_FILE},
        ),
        "d3": d3(root, replicates=replicates, seed=seed, successes=REPROBE_FILE),
    }


# -------------------------------------------------------------------- CLI


def parse_blocks(raw: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for part in raw.split(","):
        key, _, value = part.partition(":")
        out[int(key)] = int(value)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("d1", "d2"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--blocks", default="64:1,256:1,1024:1,4096:1")
        p.add_argument("--replicates", type=int, default=DIAG_REPLICATES)
        p.add_argument("--seed", type=int, default=DIAG_SEED)
        p.add_argument("--out", type=Path, required=True)
    p3 = sub.add_parser("d3")
    p3.add_argument("--root", type=Path, required=True)
    p3.add_argument("--size", type=int, default=4096)
    p3.add_argument("--replicates", type=int, default=DIAG_REPLICATES)
    p3.add_argument("--seed", type=int, default=DIAG_SEED)
    p3.add_argument("--out", type=Path, required=True)
    pr = sub.add_parser("reprobe")
    pr.add_argument("--root", type=Path, required=True)
    pr.add_argument("--part", choices=("a", "b"), required=True)
    pr.add_argument("--out", type=Path, required=True)
    pr.add_argument("--shard", default="0/1")
    pv = sub.add_parser("validate")
    pv.add_argument("--root", type=Path, required=True)
    pv.add_argument("--cells", required=True)
    pv.add_argument("--lesion-cell")
    pv.add_argument("--out", type=Path, required=True)
    p4 = sub.add_parser("d4")
    p4.add_argument("--root", type=Path, required=True)
    p4.add_argument("--replicates", type=int, default=DIAG_REPLICATES)
    p4.add_argument("--seed", type=int, default=DIAG_SEED)
    p4.add_argument("--out", type=Path, required=True)
    return parser


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "d1":
        _write(
            args.out,
            d1(
                args.root,
                blocks=parse_blocks(args.blocks),
                replicates=args.replicates,
                seed=args.seed,
            ),
        )
    elif args.cmd == "d2":
        _write(
            args.out,
            d2(
                args.root,
                blocks=parse_blocks(args.blocks),
                replicates=args.replicates,
                seed=args.seed,
            ),
        )
    elif args.cmd == "d3":
        _write(args.out, d3(args.root, size=args.size, replicates=args.replicates, seed=args.seed))
    elif args.cmd == "reprobe":
        i, n = (int(v) for v in args.shard.split("/"))
        print(f"reprobed {reprobe(args.root, part=args.part, out=args.out, shard=(i, n))} cells")
    elif args.cmd == "validate":
        _write(
            args.out,
            validate(
                args.root,
                cells=[c for c in args.cells.split(",") if c],
                lesion_cell=args.lesion_cell,
            ),
        )
    else:
        _write(args.out, d4(args.root, replicates=args.replicates, seed=args.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
