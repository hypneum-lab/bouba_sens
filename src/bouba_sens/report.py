"""HTML report generator — Jinja2 template on curves.parquet + metrics.parquet.

Sprint 3 Task 3.9 ships a static-table HTML summary per spec §5.4.
Interactive plotly heatmaps / curves are deferred to Sprint 4 once the
full 5-seed grid is available (plan R-sprint3-3).
"""

from __future__ import annotations

import html
import json
from glob import glob as _glob
from pathlib import Path

SECTIONS = (
    "Me1/Me2 recap table",
    "Me6 asymmetry matrix",
    "Me2 recovery curves T1 vs T2",
    "Me7 congenital gap",
    "Me9 IC 95%",
)


def _collect_reports(run_glob: str) -> list[tuple[str, dict[str, object]]]:
    """Resolve `run_glob` to a list of (name, eval_report dict).

    Accepts both a glob of run directories (each containing eval_report.json)
    and a glob of JSON files directly.
    """
    rows: list[tuple[str, dict[str, object]]] = []
    for path_str in sorted(_glob(run_glob)):
        path = Path(path_str)
        candidate = path / "eval_report.json" if path.is_dir() else path
        if candidate.exists():
            rows.append((str(path), json.loads(candidate.read_text())))
    return rows


def render_html(run_glob: str, out_path: str) -> None:
    """Aggregate eval reports into a static-table HTML summary."""
    rows = _collect_reports(run_glob)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    html_buf: list[str] = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>bouba_sens v0.1 summary</title></head><body>",
        "<h1>bouba_sens v0.1 summary</h1>",
        f"<p>{len(rows)} runs matched glob <code>{html.escape(run_glob)}</code>.</p>",
        f"<h2>{SECTIONS[0]}</h2>",
        "<table border='1' cellpadding='4'>",
        "<tr><th>run</th><th>Me1</th><th>Me2</th><th>Me3 delta</th><th>Me7</th></tr>",
    ]
    for name, rep in rows:
        html_buf.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{rep.get('me1', '-')}</td>"
            f"<td>{rep.get('me2', '-')}</td>"
            f"<td>{rep.get('me3_delta', '-')}</td>"
            f"<td>{rep.get('me7', '-')}</td>"
            "</tr>"
        )
    html_buf.append("</table>")

    # Remaining sections are placeholders until Sprint 4 lands the
    # multi-seed grid that these visualisations summarise.
    for section in SECTIONS[1:]:
        html_buf.append(f"<h2>{section}</h2>")
        html_buf.append("<p><em>(plot deferred to Sprint 4 — needs multi-seed grid runs)</em></p>")

    html_buf.append("</body></html>")
    out.write_text("\n".join(html_buf))
