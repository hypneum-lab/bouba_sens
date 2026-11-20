"""Unit tests for scripts/sha256_manifest.py --check mode.

Uses tmp_path so no real reports/ files are touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add scripts/ to sys.path so the module can be imported directly.
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from sha256_manifest import build_manifest, check_manifest  # noqa: E402


def _write_file(path: Path, content: bytes = b"hello") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------


def test_build_manifest_basic(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    _write_file(root / "a.json", b'{"x": 1}')
    _write_file(root / "b.csv", b"col\n1\n")
    manifest = build_manifest(root)
    assert set(manifest) == {"a.json", "b.csv"}
    assert all("sha256" in v and "size" in v for v in manifest.values())


def test_build_manifest_skips_manifest_itself(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    _write_file(root / "MANIFEST.json", b"{}")
    _write_file(root / "data.json", b"{}")
    manifest = build_manifest(root)
    assert "MANIFEST.json" not in manifest
    assert "data.json" in manifest


# ---------------------------------------------------------------------------
# check_manifest — identical tree → exit 0
# ---------------------------------------------------------------------------


def test_check_identical_returns_zero(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    _write_file(root / "a.json", b'{"x": 1}')
    manifest = build_manifest(root)
    out = tmp_path / "MANIFEST.json"
    out.write_text(json.dumps(manifest) + "\n")
    assert check_manifest(root, out) == 0


# ---------------------------------------------------------------------------
# check_manifest — modified byte → exit 1
# ---------------------------------------------------------------------------


def test_check_modified_file_returns_one(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    f = _write_file(root / "a.json", b'{"x": 1}')
    manifest = build_manifest(root)
    out = tmp_path / "MANIFEST.json"
    out.write_text(json.dumps(manifest) + "\n")

    # Mutate the file after writing the manifest.
    f.write_bytes(b'{"x": 2}')
    assert check_manifest(root, out) == 1


# ---------------------------------------------------------------------------
# check_manifest — missing file (removed from tree) → exit 1
# ---------------------------------------------------------------------------


def test_check_missing_file_returns_one(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    f = _write_file(root / "a.json", b'{"x": 1}')
    manifest = build_manifest(root)
    out = tmp_path / "MANIFEST.json"
    out.write_text(json.dumps(manifest) + "\n")

    f.unlink()
    assert check_manifest(root, out) == 1


# ---------------------------------------------------------------------------
# check_manifest — extra file in tree (not in manifest) → exit 1
# ---------------------------------------------------------------------------


def test_check_extra_file_returns_one(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    _write_file(root / "a.json", b'{"x": 1}')
    manifest = build_manifest(root)
    out = tmp_path / "MANIFEST.json"
    out.write_text(json.dumps(manifest) + "\n")

    _write_file(root / "b.json", b'{"y": 2}')
    assert check_manifest(root, out) == 1


# ---------------------------------------------------------------------------
# check_manifest — missing manifest file itself → exit 1
# ---------------------------------------------------------------------------


def test_check_missing_manifest_file_returns_one(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    _write_file(root / "a.json", b"{}")
    out = tmp_path / "MANIFEST.json"  # does not exist
    assert check_manifest(root, out) == 1
