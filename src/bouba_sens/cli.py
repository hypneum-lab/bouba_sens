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

import typer

from bouba_sens._version import __version__

app = typer.Typer(
    help="bouba_sens — Cross-modal plasticity benchmark",
    no_args_is_help=True,
)


@app.callback()
def _main() -> None:
    """bouba_sens — Cross-modal plasticity benchmark."""


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
    out: Path = typer.Option(..., help="Run directory"),
) -> None:
    """Phase 1 pretrain on GaussianWorld, save Checkpoint."""
    import pickle

    from track_p.multiplexer import (  # type: ignore[import-not-found]
        GammaThetaMultiplexer,
    )

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
    from bouba_sens.world.gaussian import GaussianWorld

    world = GaussianWorld(seed=seed)
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
    loop = AdaptationLoop(world, mux, sensories, nerve, head)

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
    from bouba_sens.world.gaussian import GaussianWorld

    world = GaussianWorld(seed=0)
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
    loop = AdaptationLoop(world, mux, sensories, nerve, head)

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

    out.mkdir(parents=True, exist_ok=True)
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
    typer.echo(f"lesion done; wrote {out}/report.pkl")


@app.command()
def eval(
    run: Path = typer.Option(..., help="Phase 2 run directory (contains report.pkl)"),
    metrics: str = typer.Option(
        "Me1,Me2,Me7",
        help="Comma-separated metric ids",
    ),
    out: Path = typer.Option(Path("eval_report.json"), help="Output JSON path"),
) -> None:
    """Compute metrics over a Phase 2 report.pkl."""
    import pickle

    from bouba_sens.metrics import (
        EvalReport,
        me1_accuracy,
        me2_recovery_auc,
        me7_congenital_gap,
    )

    with (run / "report.pkl").open("rb") as f:
        report = pickle.load(f)

    eval_report = EvalReport()
    selected = {m.strip() for m in metrics.split(",")}
    if "Me1" in selected:
        eval_report.me1 = me1_accuracy(report)
    if "Me2" in selected:
        eval_report.me2 = me2_recovery_auc(report)
    if "Me7" in selected:
        # Placeholder — needs both T1 + T2 runs. CLI passes through None
        # unless the caller wired them.
        eval_report.me7 = me7_congenital_gap(eval_report.me1 or 0.0, eval_report.me1 or 0.0)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(
            {
                "me1": eval_report.me1,
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
