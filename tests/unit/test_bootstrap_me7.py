from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from collections.abc import Callable

from typer import Typer

from scripts.bootstrap_me7 import bootstrap_me7_median_ci


def typer_app_from_main(main_fn: Callable[..., None]) -> Typer:
    app = Typer()
    app.command()(main_fn)
    return app


def test_tight_ci_on_well_separated_sample() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=0.1, scale=0.01, size=75)
    ci = bootstrap_me7_median_ci(sample, n_boot=2000, seed=0)
    assert 0.08 <= ci["ci_low"] <= ci["median"] <= ci["ci_high"] <= 0.12


def test_ci_straddles_zero_on_near_zero_sample() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=0.001, scale=0.02, size=75)
    ci = bootstrap_me7_median_ci(sample, n_boot=2000, seed=0)
    assert ci["ci_low"] <= 0.0 <= ci["ci_high"]


def test_determinism_on_seed() -> None:
    sample = np.linspace(-0.01, 0.02, 75)
    a = bootstrap_me7_median_ci(sample, n_boot=500, seed=7)
    b = bootstrap_me7_median_ci(sample, n_boot=500, seed=7)
    assert a == b


def test_load_me7_sample_raises_on_missing_key(tmp_path: Path) -> None:
    from scripts.bootstrap_me7 import _load_me7_sample

    p = tmp_path / "no_me7.json"
    p.write_text(json.dumps({"unrelated": 1}))
    with pytest.raises(KeyError, match="raw Me7 pairs"):
        _load_me7_sample(p)


def test_pairwise_disjoint_symmetric_with_none_diagonal(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from scripts import bootstrap_me7 as bm

    runner = CliRunner()
    # Build 3 world aggregates with well-separated samples
    specs = {"gaussian": -0.5, "xor": 0.0, "sinusoid": 0.5}
    paths: dict[str, Path] = {}
    for w, loc in specs.items():
        p = tmp_path / f"v0.2_aggregate_{w}.json"
        sample = np.random.default_rng(0).normal(loc=loc, scale=0.01, size=75).tolist()
        p.write_text(json.dumps({"raw_me7_pairs": sample}))
        paths[w] = p
    out_path = tmp_path / "me7_bootstrap.json"
    result = runner.invoke(
        bm.app if hasattr(bm, "app") else typer_app_from_main(bm.main),
        [
            "--gaussian",
            str(paths["gaussian"]),
            "--xor",
            str(paths["xor"]),
            "--sinusoid",
            str(paths["sinusoid"]),
            "--out",
            str(out_path),
            "--n-boot",
            "500",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text())
    dj = payload["pairwise_disjoint"]
    # symmetry
    assert dj["gaussian"]["xor"] == dj["xor"]["gaussian"]
    assert dj["xor"]["sinusoid"] == dj["sinusoid"]["xor"]
    # self-diagonal is None
    assert dj["gaussian"]["gaussian"] is None
    assert dj["xor"]["xor"] is None
    assert dj["sinusoid"]["sinusoid"] is None
    # well-separated means Gaussian vs Sinusoid must be disjoint
    assert dj["gaussian"]["sinusoid"] is True


def test_cli_collects_load_errors(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from scripts import bootstrap_me7 as bm

    runner = CliRunner()
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"raw_me7_pairs": [0.0] * 75}))
    missing = tmp_path / "missing.json"  # intentionally not created
    out_path = tmp_path / "out.json"
    result = runner.invoke(
        bm.app if hasattr(bm, "app") else typer_app_from_main(bm.main),
        [
            "--gaussian",
            str(good),
            "--xor",
            str(missing),
            "--sinusoid",
            str(good),
            "--out",
            str(out_path),
            "--n-boot",
            "200",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text())
    assert "xor" in payload["load_errors"]
    assert "gaussian" in payload["per_world"]
    assert "sinusoid" in payload["per_world"]
    assert "xor" not in payload["per_world"]
