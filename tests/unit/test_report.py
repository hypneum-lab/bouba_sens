"""Task 3.9 acceptance tests for `bouba_sens.report.render_html`."""

from __future__ import annotations

import json
from pathlib import Path

from bouba_sens.report import SECTIONS, render_html


def _write_eval(dir_: Path, **metrics: object) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "eval_report.json").write_text(json.dumps(metrics))


def test_report_writes_html_file(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    _write_eval(run_a, me1=0.8, me2=0.6)
    out = tmp_path / "summary.html"
    render_html(run_glob=str(tmp_path / "run_*"), out_path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_report_contains_all_sections(tmp_path: Path) -> None:
    _write_eval(tmp_path / "run_1", me1=0.7, me2=0.5, me3_delta=0.12, me7=0.08)
    out = tmp_path / "summary.html"
    render_html(run_glob=str(tmp_path / "run_*"), out_path=str(out))
    body = out.read_text()
    for section in SECTIONS:
        assert section in body, f"section {section!r} missing from report"


def test_report_renders_row_per_run(tmp_path: Path) -> None:
    _write_eval(tmp_path / "run_1", me1=0.7, me2=0.5)
    _write_eval(tmp_path / "run_2", me1=0.8, me2=0.6)
    out = tmp_path / "summary.html"
    render_html(run_glob=str(tmp_path / "run_*"), out_path=str(out))
    body = out.read_text()
    # Both runs appear in the table.
    assert "run_1" in body
    assert "run_2" in body
    # The metric values surface as strings.
    assert "0.7" in body
    assert "0.8" in body


def test_report_empty_glob_produces_empty_table(tmp_path: Path) -> None:
    out = tmp_path / "summary.html"
    render_html(run_glob=str(tmp_path / "none_*"), out_path=str(out))
    body = out.read_text()
    assert "0 runs matched" in body
