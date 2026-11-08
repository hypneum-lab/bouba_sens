"""Issue #74 — phase-1 physio is read with its real layout and rate.

ds000113 ``ses-forrestgump`` physio files have four columns
``[trigger, respiratory, cardiac, SpO2]`` at 100 Hz (Hanke et al. 2014,
Sci. Data 1:140003, p. 11). The ``recording-cardresp`` layout has three
columns ``[trigger, cardiac, respiratory]`` at 500 Hz.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from fetch_studyforrest_sample import (  # type: ignore[import-not-found]  # noqa: E402
    TARGET_HZ,
    _build_force,
    _load_physio_tsv,
    _run_seconds,
)

_SECONDS = 20


def _write_tsv_gz(path: Path, data: np.ndarray) -> bytes:
    """Write a header-less, tab-separated, gzip-compressed physio file."""
    with gzip.open(path, "wt") as fh:
        np.savetxt(fh, data, delimiter="\t", fmt="%.6f")
    return path.read_bytes()


def _trigger(n: int, hz: float) -> np.ndarray:
    trig = np.zeros(n)
    trig[:: int(2 * hz)] = 1.0  # one pulse per 2 s TR
    return trig


def _four_column_100hz(tmp_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hz = 100.0
    n = int(_SECONDS * hz)
    t = np.arange(n) / hz
    resp = np.sin(2 * np.pi * 0.3 * t)
    cardiac = 10.0 + np.sin(2 * np.pi * 1.4 * t)  # offset keeps it distinct
    spo2 = np.full(n, 97.0)
    data = np.stack([_trigger(n, hz), resp, cardiac, spo2], axis=1)
    run = _load_physio_tsv(_write_tsv_gz(tmp_path / "run-01_physio.tsv.gz", data))
    return run, cardiac, resp


def test_four_columns_at_100hz_map_cardiac_and_resp(tmp_path: Path) -> None:
    run, cardiac, resp = _four_column_100hz(tmp_path)
    assert run.shape == (_SECONDS * 100, 4)
    n_frames = _SECONDS * TARGET_HZ
    force = _build_force([run], n_frames).numpy()
    assert force.shape == (n_frames, 6)
    np.testing.assert_allclose(force[:, 0], cardiac[::10], atol=1e-5)
    np.testing.assert_allclose(force[:, 3], resp[::10], atol=1e-5)


def test_four_columns_at_100hz_keep_the_time_base(tmp_path: Path) -> None:
    """20 s of 100 Hz physio fills exactly 20 s of 10 Hz frames, no padding."""
    run, _, _ = _four_column_100hz(tmp_path)
    assert _run_seconds(run) == pytest.approx(_SECONDS)
    force = _build_force([run], _SECONDS * TARGET_HZ + 5).numpy()
    assert np.all(force[-5:, 0] == 0.0)
    assert np.all(force[: _SECONDS * TARGET_HZ, 0] != 0.0)


def test_three_columns_at_500hz_map_cardiac_and_resp(tmp_path: Path) -> None:
    hz = 500.0
    n = int(_SECONDS * hz)
    t = np.arange(n) / hz
    cardiac = 10.0 + np.sin(2 * np.pi * 1.4 * t)
    resp = np.sin(2 * np.pi * 0.3 * t)
    data = np.stack([_trigger(n, hz), cardiac, resp], axis=1)
    run = _load_physio_tsv(_write_tsv_gz(tmp_path / "cardresp_physio.tsv.gz", data))
    assert _run_seconds(run) == pytest.approx(_SECONDS)
    n_frames = _SECONDS * TARGET_HZ
    force = _build_force([run], n_frames).numpy()
    np.testing.assert_allclose(force[:, 0], cardiac[::50], atol=1e-5)
    np.testing.assert_allclose(force[:, 3], resp[::50], atol=1e-5)


def test_sidecar_overrides_inferred_rate_and_columns() -> None:
    hz = 200.0
    n = int(_SECONDS * hz)
    cardiac = np.linspace(5.0, 6.0, n)
    resp = np.linspace(-1.0, 1.0, n)
    run = np.stack([_trigger(n, hz), cardiac, resp, np.full(n, 97.0)], axis=1)
    sidecar = {
        "SamplingFrequency": hz,
        "Columns": ["trigger", "cardiac", "respiratory", "spo2"],
    }
    force = _build_force([run], _SECONDS * TARGET_HZ, [sidecar]).numpy()
    np.testing.assert_allclose(force[:, 0], cardiac[::20], atol=1e-5)
    np.testing.assert_allclose(force[:, 3], resp[::20], atol=1e-5)


@pytest.mark.parametrize("n_cols", [2, 5])
def test_unexpected_column_count_fails_explicitly(n_cols: int) -> None:
    run = np.zeros((1000, n_cols))
    with pytest.raises(ValueError, match=f"{n_cols} columns"):
        _build_force([run], 10)
