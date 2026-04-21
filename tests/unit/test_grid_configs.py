"""Task 4.1 acceptance tests for the Hydra config grid."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from hydra import compose, initialize_config_dir
except ImportError:  # pragma: no cover
    pytest.skip("hydra not available", allow_module_level=True)

_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def test_grid_override_matrix_has_150_cells() -> None:
    """5 seeds x 5 modalities x 2 timings x 3 SNR = 150 unique cells."""
    seeds = range(5)
    modalities = ("audio", "vision", "tactile", "gravity", "force")
    timings = ("t1", "t2")
    snrs = ("floor", "minus10", "plus10")
    cells = {(s, m, t, snr) for s in seeds for m in modalities for t in timings for snr in snrs}
    assert len(cells) == 150


def test_grid_config_loads_under_hydra_compose() -> None:
    """Compose grid.yaml with explicit timing + modality + snr overrides."""
    with initialize_config_dir(config_dir=str(_CONFIG_ROOT), version_base=None):
        cfg = compose(
            config_name="grid",
            overrides=[
                "timing=t1",
                "modality=audio",
                "snr=floor",
                "run.seed=3",
            ],
        )
    assert cfg.timing.kind == "T1"
    assert cfg.modality.name == "audio"
    assert cfg.snr.label == "floor"
    assert cfg.run.seed == 3


def test_grid_timing_t2_pretrains() -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_ROOT), version_base=None):
        cfg = compose(
            config_name="grid",
            overrides=["timing=t2", "modality=vision", "snr=plus10"],
        )
    assert cfg.timing.kind == "T2"
    assert cfg.timing.skip_pretrain is False
    assert cfg.modality.name == "vision"
    assert cfg.snr.floor_db == 10.0


def test_grid_every_modality_config_exists() -> None:
    for m in ("audio", "vision", "tactile", "gravity", "force"):
        assert (_CONFIG_ROOT / "modality" / f"{m}.yaml").exists()


def test_run_grid_default_metrics_includes_me3() -> None:
    """Guard: scripts/run_grid.sh default METRICS must match the Sprint 5+ pipeline.

    The aggregator computes B-2 from `me3_delta` at the per-cell level and
    Me7 / Me6 at the aggregation level. Missing Me3 in the shell default
    silently breaks B-2 (cells_counted == 0) unless the caller overrides
    METRICS. See ADR-0005 / Sprint 7 micro-fix G1.
    """
    import re

    script = Path(__file__).resolve().parents[2] / "scripts" / "run_grid.sh"
    content = script.read_text()
    match = re.search(r'^METRICS="\$\{METRICS:-([^"]+)\}"', content, re.MULTILINE)
    assert match is not None, "METRICS default line not found in scripts/run_grid.sh"
    metrics = [m.strip() for m in match.group(1).split(",")]
    assert "Me1" in metrics, f"Me1 missing from default METRICS: {metrics!r}"
    assert "Me2" in metrics, f"Me2 missing from default METRICS: {metrics!r}"
    assert "Me3" in metrics, f"Me3 missing from default METRICS: {metrics!r}"
    assert "Me7" not in metrics, (
        f"Me7 must NOT be in per-cell METRICS default (it is computed at "
        f"aggregation level); got {metrics!r}"
    )
