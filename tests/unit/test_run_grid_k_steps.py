"""v0.7 / v0.8 pre-filing task: ``K_STEPS`` pass-through in run_grid.sh.

Amendments v0.7 (section 4.3 item 7) and v0.8 (section 4) freeze the SNR
ramp at ``k_steps`` = 50 (ADR-0020). ``bouba-sens lesion`` already takes
``--k-steps``; ``run_grid.sh`` must forward ``K_STEPS`` to it, and must
leave the command line unchanged when ``K_STEPS`` is unset (historical
grids). ``uv`` is replaced by a stub that logs its arguments.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_grid.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _run_grid(tmp_path: Path, **env_extra: str) -> list[list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "uv.log"
    stub = bin_dir / "uv"
    stub.write_text(f'#!/usr/bin/env bash\nprintf "%s\\t" "$@" >> "{log}"\necho >> "{log}"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "OUT_ROOT": str(tmp_path / "grid"),
        "SEEDS": "7",
        **env_extra,
    }
    subprocess.run(["bash", str(_SCRIPT)], env=env, check=True, capture_output=True, text=True)
    calls = [line.rstrip("\t").split("\t") for line in log.read_text().splitlines() if line]
    return [c for c in calls if c[:3] == ["run", "bouba-sens", "lesion"]]


def _k_steps(call: list[str]) -> str | None:
    return call[call.index("--k-steps") + 1] if "--k-steps" in call else None


def test_k_steps_is_forwarded_to_every_lesion_cell(tmp_path: Path) -> None:
    lesions = _run_grid(tmp_path, K_STEPS="50")
    assert len(lesions) == 30  # 1 seed x 5 modalities x 2 timings x 3 SNR
    assert all(_k_steps(c) == "50" for c in lesions)


def test_k_steps_unset_leaves_command_line_unchanged(tmp_path: Path) -> None:
    lesions = _run_grid(tmp_path)
    assert len(lesions) == 30
    assert all(_k_steps(c) is None for c in lesions)
