"""Issue #72 — phase-2 physio header and column mapping in extract_force."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from bouba_sens.world.studyforrest_features import extract_force

_NAME = "sub-01_ses-localizer_task-movielocalizer_run-1_recording-cardresp_physio.tsv.gz"


def _write_tsv(path: Path, rows: np.ndarray, header: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [] if header is None else ["\t".join(header)]
    lines += ["\t".join(f"{v:g}" for v in row) for row in rows]
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _phase2_rows(n: int) -> np.ndarray:
    """[trigger, cardiac, respiratory] with distinct, recognisable values."""
    idx = np.arange(n, dtype=np.float64)
    trigger = (idx % 1000 == 0).astype(np.float64)
    cardiac = 1000.0 + idx
    resp = -5000.0 - idx
    return np.stack([trigger, cardiac, resp], axis=1)


def _bids_root(tmp_path: Path, sidecar: dict[str, object] | None) -> Path:
    (tmp_path / "dataset_description.json").write_text("{}")
    if sidecar is not None:
        (tmp_path / "recording-cardresp_physio.json").write_text(json.dumps(sidecar))
    return tmp_path / "sub-01" / "ses-localizer" / "func" / _NAME


def test_headerless_file_keeps_first_row_and_maps_by_sidecar(tmp_path: Path) -> None:
    rows = _phase2_rows(5000)  # 10 s at 500 Hz
    path = _write_tsv(
        _bids_root(
            tmp_path,
            {"SamplingFrequency": 500.0, "Columns": ["trigger", "cardiac", "respiratory"]},
        ),
        rows,
    )
    force = extract_force(path, n_frames=100).numpy()
    assert force.shape == (100, 6)
    # Frame 0 is the first data row: it is not consumed as a header.
    assert force[0, 0] == pytest.approx(1000.0)
    assert force[0, 3] == pytest.approx(-5000.0)
    # Every 50th sample (500 Hz -> 10 Hz) of cardiac and respiratory.
    np.testing.assert_allclose(force[:, 0], rows[::50, 1])
    np.testing.assert_allclose(force[:, 3], rows[::50, 2])
    # The trigger column is not used.
    assert not np.allclose(force[:, 0], 0.0)


def test_sidecar_column_order_is_honoured(tmp_path: Path) -> None:
    rows = _phase2_rows(5000)[:, [0, 2, 1]]  # [trigger, respiratory, cardiac]
    path = _write_tsv(
        _bids_root(
            tmp_path,
            {"SamplingFrequency": 500.0, "Columns": ["trigger", "respiratory", "cardiac"]},
        ),
        rows,
    )
    force = extract_force(path, n_frames=100).numpy()
    np.testing.assert_allclose(force[:, 0], rows[::50, 2])  # cardiac
    np.testing.assert_allclose(force[:, 3], rows[::50, 1])  # respiratory


def test_header_line_is_consumed_when_present(tmp_path: Path) -> None:
    rows = _phase2_rows(5000)
    path = _write_tsv(tmp_path / _NAME, rows, header=["trigger", "cardiac", "respiratory"])
    (tmp_path / "dataset_description.json").write_text("{}")
    force = extract_force(path, n_frames=100).numpy()
    np.testing.assert_allclose(force[:, 0], rows[::50, 1])
    np.testing.assert_allclose(force[:, 3], rows[::50, 2])


def test_short_file_is_zero_padded(tmp_path: Path) -> None:
    rows = _phase2_rows(1000)  # 2 s -> 20 frames
    path = _write_tsv(
        _bids_root(
            tmp_path,
            {"SamplingFrequency": 500.0, "Columns": ["trigger", "cardiac", "respiratory"]},
        ),
        rows,
    )
    force = extract_force(path, n_frames=50).numpy()
    assert force.shape == (50, 6)
    assert np.all(force[20:, 0] == 0.0)
    assert np.all(force[:20, 0] != 0.0)


def test_column_count_mismatch_raises(tmp_path: Path) -> None:
    rows = _phase2_rows(500)[:, :2]
    path = _write_tsv(
        _bids_root(
            tmp_path,
            {"SamplingFrequency": 500.0, "Columns": ["trigger", "cardiac", "respiratory"]},
        ),
        rows,
    )
    with pytest.raises(ValueError, match="2 data columns"):
        extract_force(path, n_frames=10)


def test_no_sidecar_no_header_uses_phase2_layout(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rows = _phase2_rows(5000)
    path = _write_tsv(_bids_root(tmp_path, None), rows)
    with caplog.at_level("WARNING"):
        force = extract_force(path, n_frames=100).numpy()
    assert "no physio sidecar" in caplog.text
    assert force[0, 0] == pytest.approx(1000.0)  # first row kept
    np.testing.assert_allclose(force[:, 0], rows[::50, 1])
    np.testing.assert_allclose(force[:, 3], rows[::50, 2])
