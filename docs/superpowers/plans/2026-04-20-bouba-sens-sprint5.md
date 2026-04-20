# bouba_sens — Sprint 5 Implementation Plan (v0.2: CLI coverage + real B-1/B-2/B-3 verdicts)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal.** Close the three CLI coverage gaps flagged in ADR-0003 so that the next 150-run Studio grid yields real B-1 / B-2 / B-3 verdicts against the pre-registered thresholds (0.05 / 0.10 / 0.02). No threshold change, no metric-math change — only plumbing and orchestration.

**Architectural insight (revised from Sprint 4).** Me1, Me2, Me3-delta are **per-cell** metrics (computed from one lesion report). **Me6 and Me7 are aggregation metrics** — they require multiple cells. Sprint 4 tried to emit them per-cell, which is why the v0.1 CLI produced `me7 = me1 - me1 = 0` and never emitted me6. Sprint 5 moves Me6/Me7 from `eval` CLI to `aggregate_grid.py` where they belong.

**Tech Stack:** unchanged (Python 3.14, uv, PyTorch >= 2.5, scipy, sklearn, hydra, plotly, jinja2, pyarrow).

**Parent spec:** `docs/superpowers/specs/2026-04-20-bouba-sens-design.md` sections 1.2 / 4.5 / 5. ADR-0003 is the blocking context.

**Sprint 5 scope:** Tasks 5.1 -> 5.6. Paper draft remains Sprint 6.

**Compute target:** GrosMac for code + tests (5.1 -> 5.3, 5.5 -> 5.6). Studio for the re-run (5.4).

---

## File structure touched in Sprint 5

```
bouba_sens/
|-- src/bouba_sens/
|   |-- loop.py                         [Task 5.1]  capture pre-lesion MI into AdaptationReport
|   |-- metrics/__init__.py             [Task 5.1]  EvalReport gains me3_delta populated
|   |-- cli.py                          [Task 5.1]  eval emits me1+me2+me3_delta per cell
|   `-- _version.py                     [Task 5.6]  bump to 0.2.0
|-- scripts/
|   |-- run_grid.sh                     [Task 5.2]  pair T1/T2 by (seed,modality,snr) + rich eval
|   `-- aggregate_grid.py               [Task 5.2]  build 5x5 perf matrix -> me6 + T1/T2 pair -> me7
|-- tests/
|   |-- unit/test_aggregate_grid.py     [Task 5.3]  me6 + me7 fixtures
|   |-- integration/test_v02_smoke.py   [Task 5.3]  new: 1-seed full grid smoke < 60 s
|   `-- empirical/test_grid_structural.py [Task 5.3]  assert cells_counted > 0 on b1/b2/b3
|-- docs/adr/
|   `-- 0004-v02-invariant-verdicts.md  [Task 5.5]  real verdicts
`-- CHANGELOG.md                        [Task 5.6]  v0.2.0 entry
```

---

## Tasks

### Task 5.1 - Per-cell metrics: capture pre-lesion MI and wire me3_delta

Spec invariant: **B-2** needs `me3_delta = MI(codes_post; labels) - MI(codes_pre; labels) > 0.10` per cell.

Implementation:

- [ ] In `bouba_sens.loop.AdaptationLoop.lesion_phase`, just before the first `on_lesion` fires, collect one batch of **pre-lesion** `(codes, labels)` from the intact network and stash it on the returned `AdaptationReport` under a new field `pre_lesion_codes` + `pre_lesion_labels` (np arrays or torch tensors, shape `(B, D)` / `(B,)`).
- [ ] Same report gains `post_lesion_codes` + `post_lesion_labels` (last batch sampled from the steady-state post-lesion network at the end of Phase 2).
- [ ] `src/bouba_sens/cli.py::eval`: when `Me3` is in the requested metric set, call `me3_delta(pre, post, labels)` using the report fields. No more `me3_delta=None` fallback.
- [ ] `EvalReport.me3_delta` JSON key populates for every cell.
- [ ] Unit test covers both pre and post capture (deterministic seed, assert codes differ between the two).

### Task 5.2 - Grid-level metrics: me6 + me7 from aggregator, not CLI

- [ ] `scripts/run_grid.sh`: pass `METRICS="Me1,Me2,Me3"` (drop Me7 from per-cell; it moves to aggregator).
- [ ] `scripts/aggregate_grid.py::_compute_invariants` gains two helpers:
  - `_build_perf_matrix(cells_for_timing_snr)` -> `Tensor[5,5]`: rows = **query modality**, columns = **lesioned modality**. Use Me1 (post-lesion accuracy on that query modality) as the entry. Requires extending the eval CLI OR the runner to compute Me1 per-query-modality. See sub-task below.
  - `_compute_me7_paired(t1_cells, t2_cells)` -> `dict[(modality, snr), gap]`: pair by `(modality, snr)`, compute `me1_T1 - me1_T2` for each, then median across pairs for B-1.
- [ ] **Sub-task: query-modality perf capture.** Add `AdaptationLoop.query_accuracy(query_modality, batch_size=16, seed=...)` that runs the frozen post-lesion network on a batch and reports Me1 on just that modality's accuracy. `run_grid.sh` loop emits a nested `per_query_me1.json` per cell: `{audio: 0.xx, vision: 0.xx, tactile: 0.xx, gravity: 0.xx, force: 0.xx}`.
- [ ] Aggregator stacks per-query Me1 across the 5 lesioned-modality cells (same seed, same timing, same SNR) to build the 5x5 matrix; feeds it to `me6_asymmetry` + `me6_max_abs_off_diag` already in `src/bouba_sens/metrics/asymmetry.py`.
- [ ] New invariant output shape preserves `B1_ME7_THRESHOLD`, `B2_ME3_DELTA_THRESHOLD`, `B3_ME6_THRESHOLD` constants exactly.

### Task 5.3 - Tests: unit + integration + empirical

- [ ] `tests/unit/test_aggregate_grid.py` gains fixtures that seed per-query Me1 dicts + pre/post MI pairs; asserts me6 off-diag matches hand-computed value and me7 median matches hand-computed pair average.
- [ ] `tests/integration/test_v02_smoke.py`: runs the full 1-seed grid (5 modalities x 2 timings x 3 SNR = 30 cells) end-to-end in under 60 s on GrosMac, then aggregates. Asserts `cells_counted > 0` for b1, b2, b3.
- [ ] `tests/empirical/test_grid_structural.py` already skips if `v0.1_aggregate.json` missing; extend to assert `cells_counted >= 6` for each invariant when present (was `>= 0` implicitly).
- [ ] Existing 152 tests stay green.

### Task 5.4 - Re-run Studio grid (v0.2)

- [ ] `ssh studio "cd ~/Projets/bouba_sens && git pull && export PATH=/opt/homebrew/bin:\$PATH && uv sync --all-extras"`.
- [ ] Launch `nohup bash scripts/run_grid.sh > logs/grid-v02-...log 2>&1 &` with `STEPS_TRAIN=200 STEPS_LESION=100 OUT_ROOT=runs/v02_grid METRICS="Me1,Me2,Me3"`. Expected runtime ~20 min (slightly longer because per-query Me1 adds 5x sampling per cell).
- [ ] Aggregate on Studio, scp artifacts back to GrosMac.
- [ ] Run `uv run pytest tests/empirical/` -> all green, `cells_counted > 0` for all invariants.

### Task 5.5 - ADR-0004: real B-1 / B-2 / B-3 verdicts

- [ ] `docs/adr/0004-v02-invariant-verdicts.md`: for each invariant, record (threshold, median, cells_counted, passes). Contrast honestly with ADR-0003. Do NOT move thresholds.
- [ ] If any invariant fails with real data, the ADR states it as a scientific result (not a bug). If any passes, the ADR records the effect size.
- [ ] Cross-link to pre-registration OSF DOI (when the repo is mirrored there).

### Task 5.6 - Sprint 5 close + v0.2.0 release

- [ ] Bump `src/bouba_sens/_version.py` to `"0.2.0"`, `pyproject.toml` to `0.2.0`.
- [ ] `CHANGELOG.md` v0.2.0 entry: list task deliverables + ADR-0004 verdict summary.
- [ ] Update both version-pinned tests (`tests/smoke/test_imports.py`, `tests/unit/test_smoke.py`).
- [ ] `git tag -a v0.2.0 -m "v0.2.0 Sprint 5 close - first real B-1/B-2/B-3 verdicts"`.
- [ ] Push main + tag.
- [ ] Update memory `project_bouba_sens_sprint0_2026_04_20.md` + `MEMORY.md` index line.

---

## Risks + mitigations

| Risk | Mitigation |
|------|------------|
| R-sprint5-1: per-query Me1 blows up grid time | Budget: +5x query cost per cell ~= 30% total. Accept up to 30 min on Studio; otherwise reduce `batch_size` for query-only pass. |
| R-sprint5-2: pre-lesion code capture adds memory | Capture one batch only (B=16). Negligible. |
| R-sprint5-3: Me6 still 0 because symmetric perf matrix | If so, that IS the scientific verdict on B-3. Report honestly in ADR-0004. |
| R-sprint5-4: me3_delta degenerate (MI flat) | Kraskov estimator is noisy but not degenerate on 16-dim real codes. If it fails, me9 degenerate-CI fix from Sprint 4 keeps aggregate finite. |
| R-sprint5-5: Studio repo drift | `git pull --ff-only` before every grid launch; fail loud on diverge. |

---

## Exit criteria

Sprint 5 closes when:

1. All 152 existing tests + new unit + new integration + empirical tests green.
2. Studio v0.2 grid complete, artifact downloaded to `reports/v0.2_aggregate.json`.
3. `cells_counted > 0` for b1, b2, b3 in the aggregate.
4. ADR-0004 committed with real verdicts (pass or fail — either is acceptable).
5. Tag `v0.2.0` pushed.
