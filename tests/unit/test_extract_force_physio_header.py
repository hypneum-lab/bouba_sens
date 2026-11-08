"""Issue #72 — phase-2 physio columns are read by name from the sidecar.

BIDS ``*_physio.tsv.gz`` files have no header line; the column names
and the sampling rate live in ``*_physio.json`` sidecars (ds000113
phase 2 keeps ``recording-cardresp_physio.json`` at the dataset root:
500 Hz, ``["trigger", "cardiac", "respiratory"]``). The earlier
``extract_force`` consumed the first data line as a header and fell back
to column 0 (trigger) and column 1 (cardiac).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from bouba_sens.world.studyforrest_features import (
    TARGET_HZ,
    extract_force,
    physio_layout,
    read_physio_sidecar,
)

_HZ = 500
_SECONDS = 6
_N_FRAMES = _SECONDS * TARGET_HZ
_STEP = _HZ // TARGET_HZ
_NAME = "sub-01_ses-localizer_task-movielocalizer_run-1_recording-cardresp_physio.tsv.gz"


def _signals(n: int) -> dict[str, np.ndarray]:
    """Distinguishable synthetic channels: values never coincide."""
    t = np.arange(n)
    trigger = np.zeros(n)
    trigger[:: 2 * _HZ] = 1.0
    return {
        "trigger": trigger,
        "cardiac": 1000.0 + t,  # strictly increasing, offset 1000
        "respiratory": -5000.0 - 2.0 * t,  # strictly decreasing, negative
        "spo2": np.full(n, 97.0),
    }


def _write_physio(directory: Path, columns: list[str], name: str = _NAME) -> tuple[Path, int]:
    directory.mkdir(parents=True, exist_ok=True)
    n = _SECONDS * _HZ
    sig = _signals(n)
    data = np.stack([sig[c] for c in columns], axis=1)
    path = directory / name
    with gzip.open(path, "wt") as fh:
        np.savetxt(fh, data, delimiter="\t", fmt="%.1f")
    return path, n


def _write_json(path: Path, content: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


def _dataset(tmp_path: Path, columns: list[str], hz: float = _HZ) -> Path:
    """BIDS-like tree with the column list in a root-level sidecar."""
    root = tmp_path / "ds"
    _write_json(root / "dataset_description.json", {"Name": "synthetic"})
    _write_json(
        root / "recording-cardresp_physio.json",
        {"SamplingFrequency": hz, "StartTime": 0.0, "Columns": columns},
    )
    # Not applicable (different recording entity): must be ignored.
    _write_json(
        root / "recording-eyegaze_physio.json",
        {"SamplingFrequency": 1000.0, "Columns": ["x", "y", "pupil_dil", "movieframe_idx"]},
    )
    # Applicable but without Columns: must not erase the inherited ones.
    _write_json(root / "task-movielocalizer_physio.json", {"TaskName": "movielocalizer"})
    path, _ = _write_physio(root / "sub-01" / "ses-localizer" / "func", columns)
    return path


def _expected(column: str) -> np.ndarray:
    return _signals(_SECONDS * _HZ)[column][::_STEP][:_N_FRAMES].astype(np.float32)


@pytest.mark.parametrize(
    "columns",
    [
        ["trigger", "cardiac", "respiratory"],  # nominal ds000113 phase 2
        ["respiratory", "cardiac", "trigger"],  # reversed order
        ["trigger", "respiratory", "cardiac", "spo2"],  # 4 columns
    ],
)
def test_columns_read_by_name(tmp_path: Path, columns: list[str]) -> None:
    force = extract_force(_dataset(tmp_path, columns), _N_FRAMES).numpy()
    assert force.shape == (_N_FRAMES, 6)
    np.testing.assert_array_equal(force[:, 0], _expected("cardiac"))
    np.testing.assert_array_equal(force[:, 3], _expected("respiratory"))
    np.testing.assert_allclose(force[:, 1], np.gradient(_expected("cardiac")))


def test_first_data_line_is_kept(tmp_path: Path) -> None:
    force = extract_force(_dataset(tmp_path, ["trigger", "cardiac", "respiratory"]), _N_FRAMES)
    assert float(force[0, 0]) == 1000.0  # sample 0, not sample 1
    assert float(force[0, 3]) == -5000.0


def test_trigger_never_used(tmp_path: Path) -> None:
    force = extract_force(_dataset(tmp_path, ["trigger", "cardiac", "respiratory"]), _N_FRAMES)
    assert bool((force[:, 0] >= 1000.0).all())
    assert bool((force[:, 3] <= -5000.0).all())


def test_missing_column_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="respiratory"):
        extract_force(_dataset(tmp_path, ["trigger", "cardiac", "spo2"]), _N_FRAMES)


def test_column_count_mismatch_raises(tmp_path: Path) -> None:
    path, _ = _write_physio(tmp_path, ["trigger", "cardiac", "respiratory"])
    sidecar: dict[str, object] = {
        "SamplingFrequency": 500.0,
        "Columns": ["trigger", "cardiac", "respiratory", "spo2"],
    }
    with pytest.raises(ValueError, match="4 columns"):
        extract_force(path, _N_FRAMES, sidecar=sidecar)


def test_no_sidecar_raises(tmp_path: Path) -> None:
    path, _ = _write_physio(tmp_path, ["trigger", "cardiac", "respiratory"])
    with pytest.raises(ValueError, match="Columns"):
        extract_force(path, _N_FRAMES)


def test_explicit_sidecar_and_rate(tmp_path: Path) -> None:
    path, _ = _write_physio(tmp_path, ["cardiac", "trigger", "respiratory"])
    sidecar: dict[str, object] = {
        "SamplingFrequency": 250.0,
        "Columns": ["cardiac", "trigger", "respiratory"],
    }
    force = extract_force(path, _N_FRAMES, sidecar=sidecar).numpy()
    sig = _signals(_SECONDS * _HZ)["cardiac"][::25][:_N_FRAMES].astype(np.float32)
    np.testing.assert_array_equal(force[:, 0], sig)


def test_non_integer_decimation_raises(tmp_path: Path) -> None:
    path, _ = _write_physio(tmp_path, ["trigger", "cardiac", "respiratory"])
    sidecar: dict[str, object] = {
        "SamplingFrequency": 333.0,
        "Columns": ["trigger", "cardiac", "respiratory"],
    }
    with pytest.raises(ValueError, match="integer step"):
        extract_force(path, _N_FRAMES, sidecar=sidecar)


def test_sidecar_inheritance_deeper_wins(tmp_path: Path) -> None:
    path = _dataset(tmp_path, ["trigger", "cardiac", "respiratory"])
    _write_json(
        path.parent / path.name.replace(".tsv.gz", ".json"),
        {"Columns": ["trigger", "respiratory", "cardiac"]},
    )
    meta = read_physio_sidecar(path)
    assert meta["Columns"] == ["trigger", "respiratory", "cardiac"]
    assert meta["SamplingFrequency"] == 500
    assert meta["TaskName"] == "movielocalizer"


def test_layout_rejects_missing_rate() -> None:
    with pytest.raises(ValueError, match="SamplingFrequency"):
        physio_layout({"Columns": ["trigger", "cardiac", "respiratory"]}, 3)


def test_short_file_is_zero_padded(tmp_path: Path) -> None:
    n_frames = 2 * _N_FRAMES  # the file covers only _N_FRAMES frames
    force = extract_force(
        _dataset(tmp_path, ["trigger", "cardiac", "respiratory"]), n_frames
    ).numpy()
    assert force.shape == (n_frames, 6)
    np.testing.assert_array_equal(force[:_N_FRAMES, 0], _expected("cardiac"))
    np.testing.assert_array_equal(force[:_N_FRAMES, 3], _expected("respiratory"))
    assert bool((force[_N_FRAMES:, [0, 3]] == 0.0).all())
