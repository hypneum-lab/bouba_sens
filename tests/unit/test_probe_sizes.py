"""ADR-0026 — `bouba-sens lesion --probe-sizes` (v1 probe at several sizes)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch
import typer

from bouba_sens.cli import (
    PROBE_SIZES_META_FILE,
    parse_probe_sizes,
    probe_sizes_accuracy_file,
    probe_sizes_successes_file,
)
from bouba_sens.loop import successes_to_accuracy
from bouba_sens.sensory import MODALITIES


def test_parse_probe_sizes() -> None:
    assert parse_probe_sizes("") == []
    assert parse_probe_sizes("  ") == []
    assert parse_probe_sizes("256, 64,64") == [64, 256]
    for bad in ("0", "-4", "abc", "64,,x"):
        with pytest.raises(typer.BadParameter):
            parse_probe_sizes(bad)


def test_file_names() -> None:
    assert probe_sizes_successes_file(1024) == "per_query_successes_n1024.json"
    assert probe_sizes_accuracy_file(64) == "per_query_me1_n64.json"


def _lesion(out: Path, *extra: str) -> None:
    subprocess.run(
        [
            "uv", "run", "bouba-sens", "lesion", "--modality", "vision", "--timing", "T1",
            "--steps", "3", "--seed", "11", "--probe-v2-seed", "5000", *extra,
            "--out", str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )  # fmt: skip


def test_cli_probe_sizes_leave_every_other_artefact_bit_identical(tmp_path: Path) -> None:
    base, both, one = tmp_path / "base", tmp_path / "both", tmp_path / "one"
    _lesion(base)
    _lesion(both, "--probe-sizes", "16,8", "--probe-sizes-seed", "4242")
    _lesion(one, "--probe-sizes", "16", "--probe-sizes-seed", "4242")

    for name in (
        "per_query_successes.json",
        "per_query_me1.json",
        "per_query_successes_v2.json",
        "per_query_me1_v2.json",
    ):
        assert (base / name).read_bytes() == (both / name).read_bytes(), name
    a = torch.load(base / "model.pt", weights_only=True)
    b = torch.load(both / "model.pt", weights_only=True)
    for part in ("mux", "nerve", "head"):
        for k in a[part]:
            assert torch.equal(a[part][k], b[part][k]), (part, k)

    # Default: no extra file, metadata unchanged in keys.
    assert not list(base.glob("per_query_successes_n*.json"))
    assert not (base / PROBE_SIZES_META_FILE).exists()
    assert "probe_sizes" not in json.loads((base / "metadata.json").read_text())

    meta = json.loads((both / "metadata.json").read_text())
    assert meta["probe_sizes"] == [8, 16] and meta["probe_sizes_seed"] == 4242
    for n in (8, 16):
        succ = json.loads((both / probe_sizes_successes_file(n)).read_text())
        acc = json.loads((both / probe_sizes_accuracy_file(n)).read_text())
        assert set(succ) == set(MODALITIES)
        assert all(len(v) == n and set(v) <= {0, 1} for v in succ.values())
        assert acc == {m: successes_to_accuracy(v) for m, v in succ.items()}
    probe_meta = json.loads((both / PROBE_SIZES_META_FILE).read_text())
    assert probe_meta["sizes"] == [8, 16] and probe_meta["probe_seed"] == 4242

    # Outcomes at one size do not depend on the other sizes asked.
    assert (both / probe_sizes_successes_file(16)).read_bytes() == (
        one / probe_sizes_successes_file(16)
    ).read_bytes()
