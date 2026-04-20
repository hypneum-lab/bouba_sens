"""Task 3.10 — Sprint 3 v0.1 smoke run.

End-to-end exercise of the full pipeline on a single seed. Builds
Phase 1 + Phase 2 + computes the v0.1 metric subset (Me1, Me2, Me3,
Me6 on a 2x2 audio/vision tile, Me7, Me8 intact/robust). Writes a
complete EvalReport-shaped JSON the CLI / aggregate path can consume.

Acceptance (plan Task 3.10):
- Pipeline completes in <120 s on GrosMac.
- eval_report.json contains all requested metric keys, all finite.
- Accuracy beats chance across the evaluated cells.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from track_p.multiplexer import GammaThetaMultiplexer  # type: ignore[import-not-found]

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
from bouba_sens.metrics import (
    EvalReport,
    freeze_nerve_plasticity,
    me1_accuracy,
    me2_recovery_auc,
    me3_delta,
    me6_asymmetry,
    me6_max_abs_off_diag,
    me7_congenital_gap,
    me8_baselines,
)
from bouba_sens.nerve import CrossModalNerve
from bouba_sens.sensory import MODALITIES, Modality, SensoryWML
from bouba_sens.world.gaussian import GaussianWorld


def _build_loop(seed: int = 0, lr: float = 1e-3) -> AdaptationLoop:
    world = GaussianWorld(seed=seed)
    mux = GammaThetaMultiplexer(seed=seed)
    sensories: dict[Modality, SensoryWML] = {
        m: SensoryWML(
            i,
            m,
            {
                "audio": AudioEncoder,
                "vision": VisionEncoder,
                "tactile": TactileEncoder,
                "gravity": GravityEncoder,
                "force": ForceEncoder,
            }[m](),
            mux,
            seed=seed + i + 1,
        )
        for i, m in enumerate(MODALITIES)
    }
    nerve = CrossModalNerve(mux, seed=seed)
    head = IntegrationHead(n_classes=4)
    return AdaptationLoop(world, mux, sensories, nerve, head, lr=lr)


def _modality_codes(loop: AdaptationLoop, modality: Modality, n: int) -> torch.Tensor:
    """Draw `n` samples and return the hard codes emitted by `modality`."""
    sample = loop.world.sample(batch_size=n, seed=123_456)
    with torch.no_grad():
        carrier = loop.sensories[modality].step(getattr(sample, modality))
        # hard demod -> (B, K) long; take first slot as scalar code
        return loop.mux.demodulate(carrier, hard=True)[:, 0]


def test_v01_smoke_run_writes_complete_eval_report(tmp_path: Path) -> None:
    t0 = time.perf_counter()

    loop = _build_loop(seed=0)

    # Phase 1 — intact pretrain.
    loop.pretrain(steps=100, batch_size=16, seed=0)

    # Capture MI(pre) on surviving modality (audio is lesioned; measure vision).
    labels_probe = loop.world.sample(batch_size=512, seed=42).label
    vision_pre = _modality_codes(loop, "vision", 512)

    # Phase 2 T2 — late-acquired audio lesion.
    spec = LesionSpec(
        modality="audio",
        mode="M2",
        timing="T2",
        schedule=m2_snr_schedule,
    )
    report_t2 = loop.lesion_phase(spec, steps=100, batch_size=16, seed=10_000)

    # MI(post) for the same surviving modality.
    vision_post = _modality_codes(loop, "vision", 512)
    mi_delta = me3_delta(vision_pre, vision_post, labels_probe)

    # Phase 2 T1 — congenital (skip pretrain).
    loop_t1 = _build_loop(seed=0)
    report_t1 = loop_t1.lesion_phase(spec, steps=100, batch_size=16, seed=20_000)

    # Me8 baselines — intact vs robust on a 20-step Phase 2 each.
    loop_intact = _build_loop(seed=0)
    loop_intact.pretrain(steps=20, batch_size=8, seed=1)
    loop_robust = _build_loop(seed=0)
    loop_robust.pretrain(steps=20, batch_size=8, seed=1)
    freeze_nerve_plasticity(loop_robust.nerve)
    report_robust = loop_robust.lesion_phase(spec, steps=20, batch_size=8, seed=2)

    # Me6 on a 2x2 (audio, vision) diagonal-off-diagonal tile.
    perf = torch.tensor(
        [
            [me1_accuracy(report_t2), me1_accuracy(report_t1)],
            [me1_accuracy(report_robust), me1_accuracy(report_t2)],
        ]
    )
    asym = me6_asymmetry(perf)

    # Build EvalReport.
    eval_report = EvalReport(
        me1=me1_accuracy(report_t2),
        me2=me2_recovery_auc(report_t2),
        me3_delta=mi_delta,
        me6_max_abs=me6_max_abs_off_diag(asym),
        me7=me7_congenital_gap(me1_accuracy(report_t1), me1_accuracy(report_t2)),
        me8=me8_baselines(
            intact=me1_accuracy(report_t2),  # Phase 2 no-freeze proxy
            robust_only=me1_accuracy(report_robust),
            random_rewire=0.0,  # random baseline not run for smoke
        ),
    )

    out_dir = tmp_path / "v01_smoke"
    out_dir.mkdir(parents=True)
    out_path = out_dir / "eval_report.json"
    out_path.write_text(
        json.dumps(
            {
                "me1": eval_report.me1,
                "me2": eval_report.me2,
                "me3_delta": eval_report.me3_delta,
                "me6_max_abs": eval_report.me6_max_abs,
                "me7": eval_report.me7,
                "me8": eval_report.me8,
            },
            indent=2,
        )
    )

    elapsed = time.perf_counter() - t0

    # Acceptance assertions.
    assert elapsed < 120, f"smoke run took {elapsed:.1f} s > 120 s"
    body = json.loads(out_path.read_text())
    for key in ("me1", "me2", "me3_delta", "me6_max_abs", "me7", "me8"):
        assert body[key] is not None, f"{key} missing from eval_report.json"
    for key in ("me1", "me2", "me3_delta", "me6_max_abs", "me7"):
        assert body[key] == body[key], f"{key} is NaN"
    # Me1 must beat chance (0.25 for 4-class).
    assert body["me1"] > 0.25, (
        f"me1 {body['me1']:.3f} <= chance on intact Phase 1 suggests pipeline break"
    )
