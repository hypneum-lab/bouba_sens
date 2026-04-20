# bouba_sens — Sprint 4 Implementation Plan (Weeks 7-8: 150-run grid + empirical B-1/B-2/B-3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the feature-complete v0.1 engine from `v0.1.0` and produce the statistical evidence that either validates or falsifies the three invariants B-1 / B-2 / B-3. Ship (1) a Hydra-driven config grid covering the full spec §4.5 replication budget (5 seeds × 5 modalities × 2 timings × 3 SNR = 150 runs per version), (2) an orchestration script + aggregator that calls the Sprint 3 CLI 150 times on Studio, (3) interactive plotly visualisations in the HTML report, (4) empirical tests that load aggregate results and assert structural completeness, and (5) ADR-0003 with the numeric B-1/B-2/B-3 validation verdicts. Paper v0.1 draft remains a Sprint 5 concern.

**Architecture:** Pure reuse of the Sprint 3 CLI + metrics — no new runtime code in `src/bouba_sens/` beyond small aggregation helpers. Studio is the compute target; GrosMac only runs the unit/empirical tests against downloaded result artifacts.

**Tech Stack:** Python 3.14, uv, PyTorch ≥ 2.5, scipy (Me9 already in Sprint 3 deps), plotly (new), hydra-core (already in Sprint 0 deps, finally activated for configs), jinja2 (already in Sprint 3 deps), pyarrow. Studio needs the repo cloned + `uv sync --all-extras`.

**Parent spec:** `docs/superpowers/specs/2026-04-20-bouba-sens-design.md` §4 (protocol) + §4.5 (replication) + §5.4 (report).

**Sprint 4 scope:** Tasks 4.1 → 4.9. Sprint 5 (paper v0.1 draft: manuscript, figures, submission to TMLR or NeurIPS D&B) is deliberately out.

**Compute target:** **Studio** for Tasks 4.2 + 4.7 (the actual 150-run grid). **GrosMac** for Tasks 4.1, 4.3, 4.4, 4.5, 4.6, 4.8, 4.9 (code + tests + ADR + docs).

---

## File structure touched in Sprint 4

```
bouba_sens/
├── src/bouba_sens/
│   ├── report.py                    [Task 4.4]  add plotly heatmap + curves
│   ├── aggregate.py                 [Task 4.3]  new module: grid-level Me9
│   └── __init__.py                  [Task 4.9]  bump to v0.1.1
├── configs/
│   ├── grid.yaml                    [Task 4.1]  Hydra base
│   ├── timing/
│   │   ├── t1.yaml                  [Task 4.1]  congenital
│   │   └── t2.yaml                  [Task 4.1]  late-acquired
│   ├── modality/
│   │   ├── audio.yaml               [Task 4.1]
│   │   ├── vision.yaml              [Task 4.1]
│   │   ├── tactile.yaml             [Task 4.1]
│   │   ├── gravity.yaml             [Task 4.1]
│   │   └── force.yaml               [Task 4.1]
│   └── snr/
│       ├── floor.yaml               [Task 4.1]  -20 dB
│       ├── minus10.yaml             [Task 4.1]  -10 dB
│       └── plus10.yaml              [Task 4.1]  +10 dB
├── scripts/
│   ├── run_grid.sh                  [Task 4.2]  orchestrator
│   └── aggregate_grid.py            [Task 4.3]  post-processor
├── tests/empirical/
│   └── test_grid_structural.py      [Task 4.5]  assert aggregate shape
├── .github/workflows/
│   └── full-benchmark.yml           [Task 4.6]  nightly / workflow_dispatch
├── docs/adr/
│   └── 0003-v01-invariant-validation.md  [Task 4.8]
└── CHANGELOG.md                     [Task 4.9]  v0.1.1 entry
```

---

## Tasks

### Task 4.1 — Hydra config grid

- [ ] In `configs/grid.yaml`, declare the base Hydra config with three
  override groups: `timing`, `modality`, `snr`. Each run pins
  `{seed, timing, modality, snr}`.
- [ ] Create 2 timing files (T1, T2), 5 modality files (one per sensory
  channel), 3 SNR files (SNR_init=20, SNR_floor=-20 per spec §4.1 + two
  intermediate `-10` and `+10`).
- [ ] Each config exposes `train.steps`, `lesion.steps`, `world`,
  `lesion.spec.modality`, `lesion.spec.timing`, `lesion.spec.snr_init`,
  `lesion.spec.snr_floor`. Sprint 3 defaults stay the no-override path.

**TDD acceptance:**
- `test_grid_override_matrix_has_150_cells` — iterating over 5 seeds × 5
  modalities × 2 timings × 3 SNR yields 150 unique override tuples.
- `test_grid_config_loads_under_hydra_compose` — a smoke test that
  composes `timing=t1 modality=audio snr=floor` and extracts the
  expected modality / timing / SNR pair.

### Task 4.2 — Orchestration script `run_grid.sh`

- [ ] `scripts/run_grid.sh` loops over the 150 cells and invokes:
  ```
  uv run bouba-sens train    --config .../intact.yaml --seed $SEED --out runs/...
  uv run bouba-sens lesion   --ckpt  runs/... --config .../cell.yaml --out runs/...
  uv run bouba-sens eval     --run   runs/... --metrics Me1,Me2,Me3,Me6,Me7,Me8 \
                              --out   runs/.../eval_report.json
  ```
- [ ] Parallelism: 1 run at a time on Studio (PyTorch grabs multiple
  threads; spawning concurrent Python processes oversubscribes the CPU).
  Output layout per spec §5.1: `runs/{date}_{config}_{seed}/`.
- [ ] Idempotent: skip cells whose `eval_report.json` already exists
  (fault tolerance — 150 runs × ~30 s = ~75 min; a crash mid-grid
  should not require a full restart).

**TDD acceptance:**
- `test_run_grid_skips_completed_cells` — pre-seeding a fake
  `eval_report.json` in a cell dir makes the script skip that cell.
  (Test via `bash -c` with a mocked CLI.)

### Task 4.3 — `aggregate_grid.py` post-processor

- [ ] `scripts/aggregate_grid.py` walks `runs/*/eval_report.json`,
  groups by `(timing, modality, snr)`, applies `me9_bootstrap` per
  metric per cell (bootstrap over the 5 seeds), writes
  `reports/v0.1_aggregate.json` with schema:
  ```
  {
    "cells": {
      "t1_audio_floor": {
        "me1": {"mean": 0.67, "ci_low": 0.61, "ci_high": 0.72},
        "me2": {...}, "me7": {...}, ...
      },
      ...
    },
    "invariants": {
      "b1": {"passes": true,  "median_me7": 0.08, "cells_passing": 4/5},
      "b2": {"passes": true,  "median_me3_delta": 0.13, "cells_passing": 5/5},
      "b3": {"passes": false, "median_max_abs": 0.015, ...}
    }
  }
  ```
- [ ] Each cell feeds `me9_bootstrap(values, n_resamples=1000,
  confidence=0.95)` per metric.

**TDD acceptance:**
- `test_aggregate_packs_cells_and_invariants` — synthetic
  `eval_report.json` fixtures produce the expected JSON schema with
  per-cell `mean/ci_low/ci_high` and an `invariants` top-level key.
- `test_aggregate_b1_b2_b3_thresholds_read_from_spec` — the threshold
  values `0.05`, `0.10`, `0.02` are centralised constants, not scattered
  magic numbers.

### Task 4.4 — Interactive plotly HTML report

- [ ] Extend `src/bouba_sens/report.py::render_html` to render:
  - Me1/Me2 mean table per cell with CI bars.
  - Me6 asymmetry matrix as plotly heatmap (real `go.Heatmap` this time).
  - Me2 recovery curves overlaid for T1 vs T2 per modality.
  - Me7 congenital gap bar chart with error bars.
  - Me9 IC 95 % table.
- [ ] Keep the static-table fallback for the single-seed smoke case —
  detect plotly availability + enough data to render interactive.

**TDD acceptance:**
- `test_report_emits_plotly_html` — output contains `<script>` tags
  referencing plotly.
- `test_report_renders_all_five_v01_figures` — each of the 5 section
  bodies contains either the plotly container or the static table.

### Task 4.5 — Empirical tests (structural)

- [ ] `tests/empirical/test_grid_structural.py`: load
  `reports/v0.1_aggregate.json` (committed as an artifact stub for CI;
  actual results land post-Studio run).
- [ ] Assert:
  - Exactly 30 cells (5 modalities × 2 timings × 3 SNR).
  - Each cell has all 6 metric summaries.
  - Every value is finite.
  - `invariants` has the 3 B-1/B-2/B-3 keys with `passes: bool`.
- [ ] Test is skipped (`pytest.skip`) if the aggregate file is
  missing — so it runs only post-grid.

**TDD acceptance:**
- `test_grid_shape_30_cells`.
- `test_grid_all_metrics_finite`.
- `test_grid_invariants_packed_correctly`.

### Task 4.6 — GitHub Actions `full-benchmark.yml`

- [ ] `.github/workflows/full-benchmark.yml` with
  `on: [workflow_dispatch, schedule]` (nightly 02:00 UTC).
  Runs only on `self-hosted` runner (Studio is the only machine
  that can practically run the grid — GrosMac is too small).
- [ ] Job steps: checkout, `uv sync`, `scripts/run_grid.sh`,
  `python scripts/aggregate_grid.py`, upload artifacts
  (`reports/v0.1_aggregate.json`, `reports/v0.1_summary.html`).

**TDD acceptance:**
- `yamllint` on the workflow file.
- `actionlint` (if available) on the workflow file.

### Task 4.7 — Studio run + artifact collection

- [ ] `ssh studio 'cd ~/Projets/bouba_sens && git pull && uv sync'`.
- [ ] `ssh studio 'cd ~/Projets/bouba_sens && bash scripts/run_grid.sh 2>&1 | tee runs/grid.log'`.
  Expected runtime: ~1-2 hours on Studio M3 Ultra (150 runs × 30-45 s).
- [ ] `scp studio:~/Projets/bouba_sens/reports/v0.1_aggregate.json ./reports/`.
- [ ] `uv run pytest tests/empirical/` on GrosMac against the downloaded
  aggregate — structural checks pass.

**TDD acceptance:**
- `reports/v0.1_aggregate.json` exists on GrosMac post-Studio-run.
- Empirical tests green against the real aggregate.

### Task 4.8 — ADR-0003 empirical invariant validation

- [ ] In `docs/adr/0003-v01-invariant-validation.md`, record:
  - Grid parameters (150 runs, 5 seeds, 5 modalities, 2 timings, 3 SNR).
  - Per-cell tables from `aggregate_grid.py`.
  - **B-1 verdict**: Me7 > 0.05 at SNR_floor, 95 % CI — pass / fail.
  - **B-2 verdict**: Me3_delta > 0.10 bit, 95 % CI — pass / fail.
  - **B-3 verdict**: Me6 max abs off-diag > 0.02 with reproducible sign
    across seeds — pass / fail.
  - Failure analysis (if any): which cells failed, hypothesised cause.
  - Paper v0.1 implications: invariants that pass go into the main
    result table; failures go into the "Limitations" section.

### Task 4.9 — Sprint 4 close

- [ ] Bump `_version.py` to `0.1.1` (maintenance release — metadata
  + empirical results, no new engine features).
- [ ] Update `CHANGELOG.md` with Sprint 4 entry.
- [ ] Commit, push, tag `v0.1.1` on `main`.
- [ ] Update memory with Sprint 4 closure + ADR-0003 verdict summary.

**Acceptance criteria:**
- `uv run pytest` green (~155/155 expected, +15 Sprint 4 new tests).
- `uv run ruff check src tests` clean.
- `uv run mypy src tests` no new errors.
- `reports/v0.1_aggregate.json` exists with 30 cells + 3 invariants.
- `git tag -l | grep v0.1.1` returns the tag.

---

## Out of scope (Sprint 5+)

- Paper v0.1 draft (manuscript, figures, LaTeX) — Sprint 5.
- TMLR / NeurIPS D&B submission — Sprint 5.
- OQ5 prioritised replay — v0.2.
- OQ2 auto-encoding / contrastive alternative heads — v0.2.
- Nightly CI actually wired (workflow exists but scheduling requires
  Studio to be registered as a self-hosted runner — that's a follow-up
  infra task).

## Risks / checkpoints

- **R-sprint4-1** — **Studio runtime**. 150 runs × ~30 s each = ~75 min
  optimistic, ~2 hours realistic. Orchestrator must be idempotent (R
  if grid crashes halfway, rerun picks up where it stopped).
- **R-sprint4-2** — **Invariant threshold fragility on 5 seeds**. Me9
  bootstrap IC 95 % on 5 samples is wide. If a B-1/B-2/B-3 verdict is
  borderline, document "inconclusive" — do not fabricate a pass.
- **R-sprint4-3** — **Plotly dep**. plotly 5.x is heavy (~15 MB). Keep
  the static-table fallback as the default CI artifact; plotly renders
  only when available + cells ≥ 2.
- **R-sprint4-4** — **Studio connectivity**. If `ssh studio` drops mid-
  run, the idempotent orchestrator saves the day but the Task 4.7
  acceptance needs manual re-run. Build in checkpointing every 10 cells.
- **R-sprint4-5** — **GitHub Actions self-hosted runner**. The nightly
  workflow needs Studio registered as self-hosted — infra task. Ship
  the YAML under `workflow_dispatch` only at first; schedule activates
  when the runner is online.

## Parent spec sections cited

- §4.1 — M2 SNR schedule.
- §4.4 — Me8 baseline triplet.
- §4.5 — replication budget (150 runs/version).
- §5.1 — run directory layout.
- §5.4 — HTML report sections.

## Related context

- bouba_sens `main@8368763`, tag `v0.1.0` — Sprint 3 closed with full
  engine + metrics + CLI + static HTML report. Sprint 4 adds the
  compute-heavy validation layer on top.
- ADR-0002 — Sprint 3 GO decision (single-seed smoke). ADR-0003 will
  supersede with multi-seed empirical results.
- `scripts/v01_smoke.sh` — Sprint 3 single-seed wrapper; Sprint 4's
  `run_grid.sh` is the multi-cell generalisation.
