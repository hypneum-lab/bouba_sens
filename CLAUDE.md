# CLAUDE.md — bouba_sens

Project-level guidance for Claude Code. See `README.md` for the full
scientific framing.

## Project

**bouba_sens** — TMLR-track benchmark for the bouba/kiki cross-modal
plasticity phenomenon, developed under the **GENIAL framework** at
**Hypneum Lab**. Studies how a 5-modality agent (audio, vision,
tactile, gravity, force) reorganises itself when one channel is lost
or degraded.

Three pre-registered invariants (OSF `10.17605/OSF.IO/Q6JYN`,
amendment v0.6):

- **B-1** — Congenital gap (lesion pre-training > lesion post-conv)
- **B-2** — MI migration (post-lesion MI rise)
- **B-3** — Perceptive/proprioceptive asymmetry (architectural)

## Status

- **HEAD ≈ v0.5.9**, ~156 commits, 19 ADRs (0004–0019), 185+ tests.
- **B-3 PASS** at 5–6× threshold (architectural invariant; anchors
  `dream-of-kiki` Stratum 4).
- **B-1 §5.5 FINAL Retract** (Q3+ 10-seed, ADR-0019) — TMLR
  submission **BLOCKED** until §5.5 is reformulated.
- **B-2 Null** (Kraskov + MINE agree within 0.1 bit).
- **N12 subgroup replication** running (kx6tm-23, tactile-floor /
  force-plus10) — picks reformulation draft A/B/C.
- Earlier landmarks: Sprints 0–10 closed 2026-04-21 (v0.5.0-final),
  Amedi dose-response curve (B-1 peak @ LOCK_AFTER=100), nerve-wml
  v1.5.3 methodology bootstrap CI, paper §5.5 populated (now under
  reformulation).

## Tech stack

- Python **3.14**, `uv` package manager (PEP 668)
- pytest (+ pytest-asyncio, pytest-cov)
- numpy / scipy / matplotlib for the benchmark + figures
- Tied to `nerve-wml` (methodology bootstrap), `kiki_oniric.axioms`
  (DR-0..DR-4)

## Key layout

| Path | Purpose |
|------|---------|
| `src/bouba_sens/` | Benchmark engine, lesions, metrics |
| `tests/{unit,property,smoke,integration,empirical}/` | Test pyramid |
| `docs/adr/` | 19 ADRs (0004–0019) — decisions of record |
| `docs/paper/` | TMLR draft + §5.5 reformulation drafts |
| `docs/osf/` | OSF pre-registration + amendments |
| `docs/milestones/` | Sprint closing notes |
| `docs/superpowers/` | Plans / brainstorms |
| `reports/` | Per-version aggregates, curves, MANIFEST |
| `configs/` | Grid + world configs |
| `scripts/` | Run/orchestration helpers |

## Commands

```bash
# Sync env
uv sync

# Full test suite
uv run python -m pytest tests/ -x

# Single bucket (fast iteration)
uv run python -m pytest tests/unit/ -x
uv run python -m pytest tests/smoke/ -x

# CLI (mirror README for exact invocations)
uv run python -m bouba_sens.cli --help
```

## Gotchas

- **Multi-seed first-class** (post-Q3 lesson). Single-seed or
  under-seeded qualitative claims are barred from publication; the
  rule was adopted portfolio-wide (nerve-wml, dream-of-kiki, here).
- **Real vs synthetic** datasets must be banner-tagged in every
  reported figure — the 4.5-modal Studyforrest Phase-2 bridge is the
  real anchor; 5 worlds are synthetic.
- **Threshold-lock**: B-1 ≥ 0.05, B-2 ≥ 0.10, B-3 ≥ 0.02 — locked
  by OSF before the campaign, unchanged across 19 ADRs. Do not
  touch.
- **Critic before ship** (paper-submission gate): mandatory
  independent review prior to TMLR/NeurIPS submission. Validated by
  the Sprints 10/11 review catching 1 MAJOR contradiction.

## Hypneum Lab naming

- Lab = **Hypneum Lab** (post-pivot 2026-04-20)
- Framework = **GENIAL** (descriptive acronym, not a brand)
- Never write "GENIAL LAB" or "Hypneum" alone.

## Closest CLAUDE.md wins

Nested `src/bouba_sens/CLAUDE.md` and `tests/CLAUDE.md` exist and
take precedence inside their subtrees.
