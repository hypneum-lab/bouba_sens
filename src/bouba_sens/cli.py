"""Typer-based CLI per spec §5.3.

Sprint 3 lands the five commands: sim / train / lesion / eval / aggregate.
All commands are self-contained — no Hydra dependency yet; explicit flags
for Sprint 3, Hydra configs deferred to Sprint 4 when the 10-config
(timing x modality) grid lands.
"""

# ruff: noqa: B008  — typer.Option-in-default is the canonical typer pattern

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import typer

from bouba_sens._version import __version__

if TYPE_CHECKING:
    from bouba_sens.world.base import WorldSimulator

app = typer.Typer(
    help="bouba_sens — Cross-modal plasticity benchmark",
    no_args_is_help=True,
)


@app.callback()
def _main() -> None:
    """bouba_sens — Cross-modal plasticity benchmark."""


def _build_world(name: str, seed: int) -> WorldSimulator:
    """Dispatch a world simulator by name. Sprint 6 / Task 6.1 + Sprint 7 / Task 7.6."""
    import os

    from bouba_sens.world.gaussian import GaussianWorld
    from bouba_sens.world.sinusoid import SinusoidWorld
    from bouba_sens.world.studyforrest import StudyforrestWorld
    from bouba_sens.world.xor import XORWorld

    key = name.lower().strip()
    if key == "gaussian":
        return GaussianWorld(seed=seed)
    if key == "xor":
        return XORWorld(seed=seed)
    if key == "sinusoid":
        return SinusoidWorld(seed=seed)
    if key == "studyforrest":
        # Real-data mode is opt-in via BOUBA_SENS_STUDYFORREST_DATA; the
        # default path uses the mock stream (see ADR-0007).
        raw_dir = os.getenv("BOUBA_SENS_STUDYFORREST_DATA")
        data_dir = Path(raw_dir) if raw_dir else None
        return StudyforrestWorld(seed=seed, data_dir=data_dir)
    raise typer.BadParameter(
        f"unknown world '{name}'; pick gaussian | xor | sinusoid | studyforrest"
    )


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(f"bouba_sens {__version__}")


@app.command()
def sim(
    world: str = typer.Option("gaussian", help="gaussian | xor | sinusoid"),
    size: int = typer.Option(1024, help="Batch size to sample"),
    seed: int = typer.Option(0, help="World sample seed"),
    out: Path = typer.Option(..., exists=False, help="Output parquet path"),
) -> None:
    """Dump a WorldSample batch to parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from bouba_sens.world.gaussian import GaussianWorld
    from bouba_sens.world.sinusoid import SinusoidWorld
    from bouba_sens.world.xor import XORWorld

    if world == "gaussian":
        sample = GaussianWorld(seed=seed).sample(batch_size=size, seed=seed)
    elif world == "xor":
        sample = XORWorld(seed=seed).sample(batch_size=size, seed=seed)
    elif world == "sinusoid":
        sample = SinusoidWorld(seed=seed).sample(batch_size=size, seed=seed)
    else:
        typer.echo(f"unknown world '{world}'", err=True)
        raise typer.Exit(code=2)

    # Flatten each tensor to a 1-D float list per row-index prefix.
    table = pa.table(
        {
            "audio": sample.audio.reshape(size, -1).tolist(),
            "vision": sample.vision.reshape(size, -1).tolist(),
            "tactile": sample.tactile.reshape(size, -1).tolist(),
            "gravity": sample.gravity.reshape(size, -1).tolist(),
            "force": sample.force.reshape(size, -1).tolist(),
            "label": sample.label.tolist(),
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    typer.echo(f"wrote {size} rows to {out}")


@app.command()
def train(
    steps: int = typer.Option(100, help="Phase 1 pretrain step count"),
    batch_size: int = typer.Option(16, help="Batch size per step"),
    seed: int = typer.Option(0, help="Training seed"),
    world: str = typer.Option("gaussian", help="gaussian | xor | sinusoid"),
    out: Path = typer.Option(..., help="Run directory"),
) -> None:
    """Phase 1 pretrain on the selected world, save Checkpoint."""
    import pickle

    from track_p.multiplexer import GammaThetaMultiplexer

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
    from bouba_sens.sensory import Modality, SensoryWML

    world_obj = _build_world(world, seed)
    mux = GammaThetaMultiplexer(seed=seed)
    sensories: dict[Modality, SensoryWML] = {
        "audio": SensoryWML(0, "audio", AudioEncoder(), mux, seed=seed + 1),
        "vision": SensoryWML(1, "vision", VisionEncoder(), mux, seed=seed + 2),
        "tactile": SensoryWML(2, "tactile", TactileEncoder(), mux, seed=seed + 3),
        "gravity": SensoryWML(3, "gravity", GravityEncoder(), mux, seed=seed + 4),
        "force": SensoryWML(4, "force", ForceEncoder(), mux, seed=seed + 5),
    }
    nerve = CrossModalNerve(mux, seed=seed)
    head = IntegrationHead(n_classes=4)
    loop = AdaptationLoop(world_obj, mux, sensories, nerve, head)

    ckpt = loop.pretrain(steps=steps, batch_size=batch_size, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "checkpoint.pkl").open("wb") as f:
        pickle.dump(ckpt, f)
    with (out / "metadata.json").open("w") as f:
        json.dump(
            {"phase": "pretrain", "steps": steps, "seed": seed, "version": __version__},
            f,
            indent=2,
        )
    # ADR-0007 Phase A — emit paired model.pt + config.yaml alongside the
    # legacy pickle artefacts. Purely additive; legacy files are untouched.
    import torch as _torch
    from omegaconf import OmegaConf  # type: ignore[import-not-found]

    _torch.save(
        {
            "mux": ckpt.mux_state,
            "nerve": ckpt.nerve_state,
            "head": ckpt.head_state,
            "sensory": ckpt.sensory_states,
        },
        out / "model.pt",
    )
    OmegaConf.save(
        OmegaConf.create(
            {
                "phase": "intact",
                "world": world,
                "seed": seed,
                "steps": steps,
                "batch_size": batch_size,
                "version": __version__,
            }
        ),
        out / "config.yaml",
    )
    typer.echo(f"pretrain done; wrote {out}/checkpoint.pkl")


@app.command()
def lesion(
    ckpt: Path = typer.Option(None, help="Phase 1 run directory; omit for congenital T1"),
    modality: str = typer.Option("audio", help="Lesioned modality"),
    timing: str = typer.Option("T2", help="T1 congenital or T2 late-acquired"),
    steps: int = typer.Option(100, help="Phase 2 lesion step count"),
    seed: int = typer.Option(10000, help="Lesion phase seed"),
    snr_init: float = typer.Option(20.0, help="Initial SNR in dB"),
    snr_floor: float = typer.Option(-20.0, help="Floor SNR in dB"),
    k_steps: int = typer.Option(5000, help="SNR ramp length in steps"),
    world: str = typer.Option("gaussian", help="gaussian | xor | sinusoid"),
    out: Path = typer.Option(..., help="Phase 2 run directory"),
) -> None:
    """Phase 2 lesion_phase. Reuses Phase 1 checkpoint if provided (T2);
    skips pretrain for T1 congenital runs (ckpt=None)."""
    import pickle

    from track_p.multiplexer import GammaThetaMultiplexer

    from bouba_sens.encoders import (
        AudioEncoder,
        ForceEncoder,
        GravityEncoder,
        TactileEncoder,
        VisionEncoder,
    )
    from bouba_sens.head import IntegrationHead
    from bouba_sens.lesion import LesionSpec, m2_snr_schedule
    from bouba_sens.loop import AdaptationLoop
    from bouba_sens.nerve import CrossModalNerve
    from bouba_sens.sensory import Modality, SensoryWML

    world_obj = _build_world(world, seed=0)
    mux = GammaThetaMultiplexer(seed=0)
    sensories: dict[Modality, SensoryWML] = {
        "audio": SensoryWML(0, "audio", AudioEncoder(), mux, seed=1),
        "vision": SensoryWML(1, "vision", VisionEncoder(), mux, seed=2),
        "tactile": SensoryWML(2, "tactile", TactileEncoder(), mux, seed=3),
        "gravity": SensoryWML(3, "gravity", GravityEncoder(), mux, seed=4),
        "force": SensoryWML(4, "force", ForceEncoder(), mux, seed=5),
    }
    nerve = CrossModalNerve(mux, seed=0)
    head = IntegrationHead(n_classes=4)
    loop = AdaptationLoop(world_obj, mux, sensories, nerve, head)

    # T2 late-acquired: restore checkpoint. T1 congenital: skip pretrain.
    if ckpt is not None and timing.upper() == "T2":
        with (ckpt / "checkpoint.pkl").open("rb") as ckpt_in:
            loop.restore(pickle.load(ckpt_in))

    def schedule(step: int) -> float:
        return m2_snr_schedule(step, snr_init=snr_init, snr_floor=snr_floor, k=k_steps)

    spec = LesionSpec(
        modality=modality,  # type: ignore[arg-type]
        mode="M2",
        timing=timing.upper(),  # type: ignore[arg-type]
        schedule=schedule,
    )
    report = loop.lesion_phase(spec, steps=steps, batch_size=16, seed=seed)

    # Sprint 5 / Task 5.2 — per-query Me1 vector for the downstream Me6
    # perf matrix. One accuracy scalar per candidate query modality, all
    # measured after Phase 2 ended, with the lesion-adapted state frozen.
    from bouba_sens.sensory import MODALITIES as _MODS

    per_query_me1 = {m: loop.query_accuracy(m, seed=seed + 777) for m in _MODS}

    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_query_me1.json").open("w") as pq_out:
        json.dump(per_query_me1, pq_out, indent=2)
    with (out / "report.pkl").open("wb") as report_out:
        pickle.dump(report, report_out)
    with (out / "metadata.json").open("w") as meta_out:
        json.dump(
            {
                "phase": "lesion",
                "steps": steps,
                "seed": seed,
                "modality": modality,
                "timing": timing.upper(),
                "snr_init": snr_init,
                "snr_floor": snr_floor,
                "k_steps": k_steps,
                "version": __version__,
            },
            meta_out,
            indent=2,
        )
    # ADR-0007 Phase A — emit paired model.pt + config.yaml alongside the
    # legacy artefacts. Post-lesion state is captured via loop.snapshot().
    import torch as _torch
    from omegaconf import OmegaConf

    post = loop.snapshot()
    _torch.save(
        {
            "mux": post.mux_state,
            "nerve": post.nerve_state,
            "head": post.head_state,
            "sensory": post.sensory_states,
        },
        out / "model.pt",
    )
    t1_ref: str = ""
    if ckpt is not None:
        try:
            t1_ref = str(Path(ckpt).resolve().relative_to(out.resolve().parent))
        except ValueError:
            t1_ref = str(Path(ckpt))
    OmegaConf.save(
        OmegaConf.create(
            {
                "phase": "lesion",
                "modality": modality,
                "timing": timing.upper(),
                "snr_init": snr_init,
                "snr_floor": snr_floor,
                "k_steps": k_steps,
                "seed": seed,
                "steps": steps,
                "world": world,
                "t1_ckpt": t1_ref,
                "version": __version__,
            }
        ),
        out / "config.yaml",
    )
    typer.echo(f"lesion done; wrote {out}/report.pkl")


def _build_model_from_config(cfg):  # type: ignore[no-untyped-def]
    """ADR-0007 Phase B — reconstruct the 6 model components from config.yaml.

    Returns `(mux, sensories, nerve, head, world_obj)`. Mirrors the
    construction sequence used by `train` / `lesion` so a round-trip
    (save state_dicts -> load_state_dict) is bit-exact.
    """
    from track_p.multiplexer import GammaThetaMultiplexer

    from bouba_sens.encoders import (
        AudioEncoder,
        ForceEncoder,
        GravityEncoder,
        TactileEncoder,
        VisionEncoder,
    )
    from bouba_sens.head import IntegrationHead
    from bouba_sens.nerve import CrossModalNerve
    from bouba_sens.sensory import SensoryWML

    seed = int(cfg["seed"])
    world_name = str(cfg.get("world", "gaussian"))
    world_obj = _build_world(world_name, seed)
    mux = GammaThetaMultiplexer(seed=seed)
    sensories = {
        "audio": SensoryWML(0, "audio", AudioEncoder(), mux, seed=seed + 1),
        "vision": SensoryWML(1, "vision", VisionEncoder(), mux, seed=seed + 2),
        "tactile": SensoryWML(2, "tactile", TactileEncoder(), mux, seed=seed + 3),
        "gravity": SensoryWML(3, "gravity", GravityEncoder(), mux, seed=seed + 4),
        "force": SensoryWML(4, "force", ForceEncoder(), mux, seed=seed + 5),
    }
    nerve = CrossModalNerve(mux, seed=seed)
    head = IntegrationHead(n_classes=4)
    return mux, sensories, nerve, head, world_obj


def _compute_me1(mux, sensories, nerve, head, world_obj, *, seed: int) -> float:  # type: ignore[no-untyped-def]
    """ADR-0007 Phase B — canonical Me1 probe pass.

    Wires a fresh `AdaptationLoop` around pre-built components and runs
    `query_accuracy("audio")` on a deterministic probe batch keyed by
    `seed + 777` (matches the per-query offset used by the `lesion`
    command, §Sprint 5 Task 5.2). Returns a float in [0, 1].

    Option 4 invariant: this function must be bit-exact deterministic given
    `(state_dicts, seed)`, independent of caller global RNG state. The
    explicit seeding block below forces torch / numpy / random into a known
    state before AdaptationLoop construction so paired-CLI and eval --run
    paths converge.
    """
    import random as _random

    import numpy as _np
    import torch as _torch

    from bouba_sens.loop import AdaptationLoop

    _torch.manual_seed(seed + 777)
    _np.random.seed(seed + 777)
    _random.seed(seed + 777)

    loop = AdaptationLoop(world_obj, mux, sensories, nerve, head)
    return loop.query_accuracy("audio", seed=seed + 777)


def _load_cell(path: Path) -> tuple[float, int | None, str | None]:
    """Return (me1, seed, cell_name) for a Sprint 4+ cell directory.

    ADR-0007 Phase B: reads `config.yaml` + `model.pt`, rebuilds the full
    model, loads state_dicts, and recomputes Me1 via the canonical probe
    pass. Falls back to the legacy `eval_report.json` path when either
    Phase B artefact is missing (ad-hoc cells fabricated by fixtures that
    never called `train` / `lesion`).
    """
    model_pt = path / "model.pt"
    cfg_path = path / "config.yaml"
    if model_pt.exists() and cfg_path.exists():
        import torch as _torch
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(cfg_path)
        mux, sensories, nerve, head, world_obj = _build_model_from_config(cfg)  # type: ignore[no-untyped-call]
        state = _torch.load(model_pt, weights_only=True)
        mux.load_state_dict(state["mux"])
        nerve.load_state_dict(state["nerve"])
        head.load_state_dict(state["head"])
        for m, sd in state["sensory"].items():
            sensories[m].load_state_dict(sd)
        seed_value = int(cfg["seed"])
        me1 = _compute_me1(mux, sensories, nerve, head, world_obj, seed=seed_value)
        return me1, seed_value, path.name

    # Legacy fallback — pre-Phase-B ad-hoc cells.
    eval_report = json.loads((path / "eval_report.json").read_text())
    me1 = float(eval_report["me1"])
    seed: int | None = None
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        raw_seed = metadata.get("seed")
        if raw_seed is not None:
            seed = int(raw_seed)
    return me1, seed, path.name


def _emit_paired_run(
    *,
    t1_ckpt: Path,
    t2_ckpt: Path,
    modality: str | None,
    snr: float | None,
    out: Path,
    me7_fn: Callable[[float, float], float],
) -> None:
    """ADR-0007 paired-run emitter."""
    from datetime import UTC, datetime

    me1_t1, seed_t1, name_t1 = _load_cell(t1_ckpt)
    me1_t2, seed_t2, _ = _load_cell(t2_ckpt)
    me7 = me7_fn(me1_t1, me1_t2)

    # Derive a pair descriptor from the T1 cell name (e.g. seed0_T1_vision_plus10
    # -> seed0_vision_plus10). Fallback to None per ADR-0007 when the T1 name
    # does not encode a cell id.
    cell_id: str | None = None
    if name_t1:
        cell_id = name_t1.replace("_T1_", "_").replace("_T2_", "_")

    payload = {
        "pair": {
            "seed_t1": seed_t1,
            "seed_t2": seed_t2,
            "cell_id": cell_id,
            "modality": modality,
            "snr": snr,
        },
        "me1_t1": me1_t1,
        "me1_t2": me1_t2,
        "me7": me7,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))


@app.command()
def eval(
    run: Path = typer.Option(None, help="Phase 2 run directory (contains report.pkl)"),
    t1_ckpt: Path = typer.Option(
        None, "--t1-ckpt", help="T1 cell directory (ADR-0007 paired-run mode)"
    ),
    t2_ckpt: Path = typer.Option(
        None, "--t2-ckpt", help="T2 cell directory (ADR-0007 paired-run mode)"
    ),
    modality: str = typer.Option(
        None, "--modality", help="Modality tag for the paired-run JSON (optional)"
    ),
    snr: float = typer.Option(None, "--snr", help="SNR in dB for the paired-run JSON (optional)"),
    metrics: str = typer.Option(
        "Me1,Me2,Me7",
        help="Comma-separated metric ids",
    ),
    out: Path = typer.Option(Path("eval_report.json"), help="Output JSON path"),
) -> None:
    """Compute metrics over a Phase 2 report.pkl, or a paired T1/T2 cell pair (ADR-0007).

    ADR-0007 Phase B (Option 4, v0.4.0) — in ``--run`` mode, the emitted
    ``eval_report.json`` carries BOTH:

      * ``me1`` : ``me1_accuracy(report)`` — mean of the last 10 % of
        ``report.accuracy_curve`` during Phase 2 adaptation, all 5
        modalities active. Primary paper observable; fixed by
        ADR-0004 / ADR-0005. Never changed.
      * ``me1_probe`` : ``query_accuracy("audio", seed+777)`` on the model
        rebuilt from ``model.pt`` + ``config.yaml`` — frozen probe pass
        on the reloaded, post-adaptation state. Reproducibility
        auxiliary introduced in v0.4.0.

    The two are semantically distinct observables and coexist
    additively. ``me1_probe`` is ``null`` on legacy (pre-Phase A) runs
    where ``model.pt`` / ``config.yaml`` are absent.
    """
    import pickle

    from bouba_sens.metrics import (
        EvalReport,
        me1_accuracy,
        me2_recovery_auc,
        me3_delta,
        me7_congenital_gap,
    )

    # ADR-0007 paired-run mode — both --t1-ckpt and --t2-ckpt provided.
    if t1_ckpt is not None and t2_ckpt is not None:
        _emit_paired_run(
            t1_ckpt=t1_ckpt,
            t2_ckpt=t2_ckpt,
            modality=modality,
            snr=snr,
            out=out,
            me7_fn=me7_congenital_gap,
        )
        typer.echo(f"paired eval done; wrote {out}")
        return

    if run is None:
        raise typer.BadParameter("--run is required unless --t1-ckpt and --t2-ckpt are provided")

    with (run / "report.pkl").open("rb") as f:
        report = pickle.load(f)

    eval_report = EvalReport()
    selected = {m.strip() for m in metrics.split(",")}
    if "Me1" in selected:
        eval_report.me1 = me1_accuracy(report)
    if "Me2" in selected:
        eval_report.me2 = me2_recovery_auc(report)
    # Sprint 5 / Task 5.1: wire real me3_delta from the pre / post probe
    # codes captured in lesion_phase. Missing fields (older report.pkl
    # artifacts) fall back to None so the CLI stays backwards compatible
    # with v0.1 pickles.
    if (
        "Me3" in selected
        and report.pre_lesion_codes is not None
        and report.post_lesion_codes is not None
        and report.probe_labels is not None
    ):
        eval_report.me3_delta = me3_delta(
            report.pre_lesion_codes,
            report.post_lesion_codes,
            report.probe_labels,
        )
    # Note: Me6 and Me7 are aggregation-level metrics (see ADR-0003 + Sprint
    # 5 plan). They move to scripts/aggregate_grid.py in Task 5.2, not here.

    # ADR-0007 Phase B Option 4 — additively emit me1_probe by rebuilding
    # the model from the Phase A artefacts (model.pt + config.yaml) and
    # running the canonical probe pass. Legacy ``me1`` above is left
    # untouched (primary observable, fixed by ADR-0004 / ADR-0005).
    # Degrades to None when either Phase A artefact is absent (pre-Phase-A
    # legacy runs).
    me1_probe: float | None = None
    model_pt = run / "model.pt"
    cfg_path = run / "config.yaml"
    if model_pt.exists() and cfg_path.exists():
        import torch as _torch
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(cfg_path)
        mux, sensories, nerve, head, world_obj = _build_model_from_config(cfg)
        state = _torch.load(model_pt, weights_only=True)
        mux.load_state_dict(state["mux"])
        nerve.load_state_dict(state["nerve"])
        head.load_state_dict(state["head"])
        for m, sd in state["sensory"].items():
            sensories[m].load_state_dict(sd)
        me1_probe = _compute_me1(mux, sensories, nerve, head, world_obj, seed=int(cfg["seed"]))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(
            {
                "me1": eval_report.me1,
                "me1_probe": me1_probe,
                "me2": eval_report.me2,
                "me3_delta": eval_report.me3_delta,
                "me6_max_abs": eval_report.me6_max_abs,
                "me7": eval_report.me7,
                "me8": eval_report.me8,
                "me9": eval_report.me9,
            },
            f,
            indent=2,
        )
    typer.echo(f"eval done; wrote {out}")


@app.command()
def aggregate(
    glob: str = typer.Option(..., help="Glob pattern for run dirs"),
    out: Path = typer.Option(..., help="Output HTML path"),
) -> None:
    """Aggregate eval reports into an HTML summary (§5.4)."""
    from bouba_sens.report import render_html

    render_html(run_glob=glob, out_path=str(out))
    typer.echo(f"aggregate done; wrote {out}")


if __name__ == "__main__":
    app()
