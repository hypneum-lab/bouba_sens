# bouba_sens — Sprint 3 Implementation Plan (Week 6: Metrics + CLI + v0.1 smoke run)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the Sprint 0 skeletons `metrics/*.py`, `cli.py`, and `report.py`. Land the seven v0.1 metrics (Me1/Me2/Me3/Me6/Me7/Me8/Me9) tied to the three invariants B-1/B-2/B-3, a typer-based CLI covering the five commands of spec §5.3, a Jinja2 HTML report (§5.4), and an end-to-end v0.1 smoke run that exercises the whole pipeline on a single seed. The full 150-run statistical grid (5 seeds × 5 modalities × 2 timings × 3 SNR) is deferred to Sprint 4 on Studio — Sprint 3 proves the pipeline works and produces an ADR with the v0.1 go/no-go decision.

**Architecture:** Pure reuse of Sprint 2 engine (`AdaptationLoop`, `CrossModalNerve`, `LesionScheduler`, `IntegrationHead`). Metrics attach to `AdaptationReport` + `EvalReport` objects; CLI orchestrates `loop.pretrain` / `loop.lesion_phase` through Hydra configs; report renders to HTML.

**Tech Stack:** Python 3.14, uv, PyTorch ≥ 2.5, pytest + hypothesis + pytest-xdist, ruff + mypy + pyright, hydra, typer + rich, jinja2, scipy (for bootstrap), pyarrow (for parquet I/O). Most deps are already installed from Sprint 0 bootstrap.

**Parent spec:** `docs/superpowers/specs/2026-04-20-bouba-sens-design.md` §5 + §1.2 (invariants).

**Sprint 3 scope:** Tasks 3.1 → 3.12. Sprint 4 (full 150-run grid on Studio + statistical aggregation) will be planned after Sprint 3 closes with a shipped v0.1 go/no-go ADR.

**Compute target:** GrosMac for all scaffolding + unit tests + the Task 3.10 single-seed smoke run. Studio reserved for the full grid (Sprint 4).

---

## File structure touched in Sprint 3

```
bouba_sens/
├── src/bouba_sens/
│   ├── metrics/
│   │   ├── __init__.py              [Task 3.7]  re-exports + Metric Protocol
│   │   ├── performance.py           [Task 3.1]  Me1 + Me2
│   │   ├── mi_migration.py          [Task 3.2]  Me3 (Kraskov kNN)
│   │   ├── asymmetry.py             [Task 3.3]  Me6 (5x5 matrix)
│   │   ├── congenital.py            [Task 3.4]  Me7 (T1 - T2 gap)
│   │   ├── baselines.py             [Task 3.5]  Me8 (intact / robust / random)
│   │   └── replication.py           [Task 3.6]  Me9 (bootstrap IC 95%)
│   ├── cli.py                        [Task 3.8]  typer: sim/train/lesion/eval/aggregate
│   ├── report.py                     [Task 3.9]  Jinja2 HTML renderer
│   └── __init__.py                   [Task 3.12] bump to v0.1.0
├── tests/unit/
│   ├── test_metrics_performance.py   [Task 3.1]
│   ├── test_metrics_mi.py            [Task 3.2]
│   ├── test_metrics_asymmetry.py     [Task 3.3]
│   ├── test_metrics_congenital.py    [Task 3.4]
│   ├── test_metrics_baselines.py     [Task 3.5]
│   ├── test_metrics_replication.py   [Task 3.6]
│   ├── test_metrics_registry.py      [Task 3.7]
│   ├── test_cli.py                   [Task 3.8]
│   └── test_report.py                [Task 3.9]
├── tests/integration/
│   └── test_v01_smoke_run.py         [Task 3.10]
├── scripts/
│   └── v01_smoke.sh                  [Task 3.10]  one-liner wrapper
├── configs/
│   ├── v0.1_intact.yaml              [Task 3.8]  (if not already)
│   ├── t1_audio_m2.yaml              [Task 3.8]  T1 congenital, audio, M2
│   └── t2_audio_m2.yaml              [Task 3.8]  T2 late-acquired, audio, M2
├── docs/adr/
│   └── 0002-v01-go-no-go.md          [Task 3.11]  ADR with smoke results
└── CHANGELOG.md                       [Task 3.12] v0.1.0 entry
```

---

## Tasks

### Task 3.1 — Me1 (accuracy) + Me2 (recovery AUC)

- [ ] In `src/bouba_sens/metrics/performance.py`, implement:
  - `me1_accuracy(report: AdaptationReport) -> float` — mean of the last 10 % of `accuracy_curve`. Represents "accuracy post-adaptation" per spec §5.2.
  - `me2_recovery_auc(report: AdaptationReport) -> float` — trapezoidal integral of `accuracy_curve` over Phase 2 steps, normalised by steps × max_accuracy.

**TDD acceptance:**
- `test_me1_accuracy_on_constant_curve` — a curve of all 0.75 returns 0.75.
- `test_me2_auc_on_flat_curve` — flat 0.75 curve gives AUC ≈ 0.75.
- `test_me2_auc_increasing_curve` — monotonically increasing 0 → 1 gives AUC ≈ 0.5.

### Task 3.2 — Me3 (MI migration)

- [ ] In `src/bouba_sens/metrics/mi_migration.py`, implement `me3_mi_migration(codes: Tensor[B], labels: Tensor[B]) -> float` using sklearn's kNN-based `mutual_info_regression` (already used for Task 1.5).
- [ ] Accept pre/post code tensors and return the MI delta: `mi_post - mi_pre`. Spec §1.2 invariant B-2 threshold: `Me3_delta > 0.10 bit`.

**TDD acceptance:**
- `test_me3_zero_for_independent_codes_and_labels`.
- `test_me3_positive_when_codes_predict_labels` — synthetic `codes = labels * 2 + noise` gives MI > 0.5 bit.
- `test_me3_delta_increases_with_post_alignment`.

### Task 3.3 — Me6 (asymmetry matrix)

- [ ] In `src/bouba_sens/metrics/asymmetry.py`, implement `me6_asymmetry(perf_matrix: Tensor[5, 5]) -> Tensor[5, 5]` that returns the signed off-diagonal antisymmetry: `A[i, j] = perf[i, j] - perf[j, i]`. Spec §1.2 invariant B-3: `max abs off-diag > 0.02` with seed-reproducible sign structure.
- [ ] `perf_matrix[i, j]` = accuracy when lesion is applied to modality `i` and query probes modality `j`.

**TDD acceptance:**
- `test_me6_zero_for_symmetric_matrix` — `perf = perf.T` returns an all-zero asymmetry.
- `test_me6_max_abs_exceeds_threshold_for_asym_matrix`.
- `test_me6_diagonal_is_zero`.

### Task 3.4 — Me7 (congenital gap)

- [ ] In `src/bouba_sens/metrics/congenital.py`, implement `me7_congenital_gap(perf_t1: float, perf_t2: float) -> float` returning `perf_t1 - perf_t2`. Spec invariant B-1: `Me7 > 0.05` at SNR_floor.

**TDD acceptance:**
- `test_me7_positive_when_t1_beats_t2`.
- `test_me7_zero_for_equal_perf`.
- `test_me7_negative_when_t2_beats_t1` — documented negative gap signals an invariant failure.

### Task 3.5 — Me8 (baselines)

- [ ] In `src/bouba_sens/metrics/baselines.py`, implement `me8_baselines(intact: float, robust_only: float, random_rewire: float) -> dict[str, float]` returning the three regime values. Also implement a helper `run_baseline(loop, regime: Literal["intact", "robust", "random"]) -> float` that configures the nerve for the regime before running `lesion_phase`.
- [ ] `"intact"` = no lesion (reference ceiling).
- [ ] `"robust"` = lesion active, P1/P2/P3 frozen (plasticity-free floor).
- [ ] `"random"` = P1/P2/P3 re-initialised with frozen random weights (chance control).

**TDD acceptance:**
- `test_me8_returns_three_regime_values`.
- `test_run_baseline_robust_freezes_plasticity` — after `run_baseline(..., "robust")`, nerve parameters have not changed.

### Task 3.6 — Me9 (bootstrap IC 95%)

- [ ] In `src/bouba_sens/metrics/replication.py`, implement `me9_bootstrap(values: list[float], *, n_resamples: int = 1000, confidence: float = 0.95) -> tuple[float, float, float]` using `scipy.stats.bootstrap`. Returns `(mean, lo, hi)`.
- [ ] Accepts any scalar metric; AdaptationLoop will call it per cell (arm × modality × SNR).

**TDD acceptance:**
- `test_me9_mean_matches_numpy_mean`.
- `test_me9_ci_contains_true_mean` — bootstrap of a known-mean distribution brackets the true mean in its CI.
- `test_me9_reproducible_with_seed`.

### Task 3.7 — Metric Protocol + registry

- [ ] In `src/bouba_sens/metrics/__init__.py`, define `Metric` `Protocol` per spec §3.7:
  ```python
  @runtime_checkable
  class Metric(Protocol):
      name: str
      def update(self, report: EvalReport) -> None: ...
      def compute(self) -> dict[str, float]: ...
  ```
- [ ] Re-export all 7 metric functions as well.

**TDD acceptance:**
- `test_metric_protocol_is_runtime_checkable` — a dummy class with the three members satisfies `isinstance`.

### Task 3.8 — `cli.py` with 5 commands

- [ ] In `src/bouba_sens/cli.py`, extend the existing `version` command with:
  - `bouba-sens sim --world gaussian --size 100k --out data/world_v1.parquet` — dump a WorldSample batch to parquet.
  - `bouba-sens train --config configs/v0.1_intact.yaml --seed 0 --out runs/<auto>/` — Phase 1 pretrain, save Checkpoint.
  - `bouba-sens lesion --ckpt runs/intact_seed0 --spec configs/t2_audio_m2.yaml --out runs/<auto>/` — Phase 2 lesion_phase.
  - `bouba-sens eval --run runs/... --metrics Me1,Me2,Me3,Me6,Me7 --out eval_report.json`.
  - `bouba-sens aggregate --glob 'runs/v0.1_*' --out reports/v0.1_summary.html`.
- [ ] Each command uses Hydra for config loading, writes a `metadata.json` with git SHA + timestamps, and parquet output per spec §5.1.

**TDD acceptance:**
- `test_cli_sim_writes_parquet` — `bouba-sens sim --world gaussian --size 32 --out /tmp/test.parquet` creates a readable parquet file.
- `test_cli_train_writes_checkpoint` — small-config train saves a valid Checkpoint.
- `test_cli_eval_writes_report` — eval against a tiny run produces the expected metric keys.

### Task 3.9 — Jinja2 HTML report

- [ ] In `src/bouba_sens/report.py`, implement `render_html(run_glob: str, out_path: str) -> None` per spec §5.4:
  - Me1/Me2 recap table per (config × modality).
  - Me6 asymmetry matrix as interactive heatmap (plotly).
  - Me2 recovery curves overlaid for T1 vs T2 per modality.
  - Me7 congenital gap bar chart.
  - Me9 IC 95% table.
- [ ] Template lives at `src/bouba_sens/templates/v0.1_summary.html.j2` (new dir).

**TDD acceptance:**
- `test_report_writes_html_file` — output file exists and is non-empty.
- `test_report_contains_all_sections` — rendered HTML contains the five section headers.

### Task 3.10 — End-to-end v0.1 smoke run

- [ ] In `tests/integration/test_v01_smoke_run.py`:
  1. Build full pipeline (GaussianWorld + 5 SensoryWMLs + CrossModalNerve + IntegrationHead + AdaptationLoop).
  2. Phase 1: 200 steps intact pretrain.
  3. Phase 2 T1: 100 steps congenital audio lesion (skip pretrain, start with lesion).
  4. Phase 2 T2: 100 steps late-acquired audio lesion (use T1-equivalent seed for comparability).
  5. Compute Me1, Me2, Me3, Me6 (subset 2×2 audio/vision), Me7, Me8 (intact + robust), Me9 (ignored — needs multiple seeds).
  6. Write `EvalReport` to `out/v01_smoke_<timestamp>/eval_report.json`.
- [ ] `scripts/v01_smoke.sh` wraps the CLI: `bouba-sens train --config configs/v0.1_intact.yaml --seed 0 && bouba-sens lesion --ckpt ... --spec configs/t2_audio_m2.yaml && bouba-sens eval --run ...`.

**TDD acceptance:**
- Pipeline runs under 2 minutes on GrosMac.
- `eval_report.json` contains all requested metric keys with finite values.
- Accuracy beats chance across all evaluated cells.

### Task 3.11 — v0.1 go/no-go ADR

- [ ] In `docs/adr/0002-v01-go-no-go.md`, write:
  - Context: Sprint 3 closed with working end-to-end pipeline.
  - Smoke results from Task 3.10 (all metric values).
  - Sanity check: are B-1/B-2/B-3 *plausible* in the single-seed data? (Single-seed cannot validate the invariants per Me9 — requires 5-seed replication on Studio.)
  - Decision: **GO** if pipeline runs end-to-end + metrics are all finite + accuracy beats chance across intact run. **NO-GO** if any metric is NaN or inf, or final accuracy is ≤ 0.25 on a 4-class task after 200 pretrain steps.
  - Next step: Sprint 4 scope — full 150-run grid on Studio (5 seeds × 5 modalities × 2 timings × 3 SNR) + Me9 bootstrap IC aggregation.

### Task 3.12 — Sprint 3 close

- [ ] Bump `src/bouba_sens/_version.py` to `0.1.0`.
- [ ] Update `CHANGELOG.md` with Sprint 3 entry (12 tasks + first v0.1 metrics + CLI + HTML report).
- [ ] Commit, push, tag `v0.1.0` on `main` (first public milestone — not a sprint tag; marks feature-completeness of the v0.1 design).
- [ ] Update memory `project_bouba_sens_sprint0_2026_04_20.md` with Sprint 3 closure.

**Acceptance criteria:**
- `uv run pytest` green (target ≥ 115/115).
- `uv run ruff check src tests` clean.
- `uv run mypy src tests` no new errors.
- `bouba-sens --help` lists the 5 commands + version.
- `git tag -l | grep v0.1.0` returns the tag.
- Studio smoke: `ssh studio 'cd ~/Projets/bouba_sens && git pull && uv run bouba-sens version'` prints `bouba_sens 0.1.0`.

---

## Out of scope (Sprint 4+)

- Full 150-run grid on Studio with Me9 bootstrap IC aggregation — Sprint 4.
- Nightly `full-benchmark.yml` activation — Sprint 4.
- Prioritised replay (OQ5) — Sprint 4 or v0.2.
- OQ2 alternative IntegrationHead task (auto-encoding, contrastive) — v0.2.
- V-JEPA / ImageBind-eval comparative baselines — v0.2 satellite.
- Hebbian / STDP PlasticityRule variants (OQ4) — v2 satellite.

## Risks / checkpoints

- **R-sprint3-1** — **Metric robustness on small batches.** Me3 (Kraskov kNN) is noisy below N=1000 samples. Task 3.10 smoke run must draw at least 1024 eval samples per modality; unit tests can use much smaller synthetic data because they assert structural properties (zero-MI for independent data, positive for aligned) rather than specific numeric thresholds.
- **R-sprint3-2** — **CLI Hydra config schema drift.** The 10-config grid (timing × modality) from spec §6.1 is too much for Sprint 3; ship only `v0.1_intact.yaml`, `t1_audio_m2.yaml`, `t2_audio_m2.yaml` and document the remaining 8 as "Sprint 4 config expansion".
- **R-sprint3-3** — **HTML report template complexity.** Interactive plotly heatmap + curves is tempting but Sprint 3 should ship a working static-table HTML first; add interactivity in Sprint 4 if time allows.
- **R-sprint3-4** — **`v0.1.0` tag semantics.** This is the first *non-sprint* tag (unlike `v0.0.2-sprint1` / `v0.0.3-sprint2`). Marks feature-completeness of the v0.1 design, not of the paper experiments. Paper v0.1 draft is a separate Sprint 5 concern.

## Parent spec sections cited

- §1.2 — three invariants B-1, B-2, B-3 tied to Me7, Me3, Me6.
- §3.7 — `Metric` Protocol.
- §4.4 — baselines (intact, robust-only, random-rewire) for Me8.
- §4.5 — statistical replication (5 seeds) for Me9 — deferred in scope to Sprint 4.
- §5 — full metrics catalogue, CLI, report.

## Related context

- bouba_sens `main@cdfecae`, tag `v0.0.3-sprint2` — Sprint 2 closed with 98 tests, full spec §2.1 block diagram wired.
- nerve-wml `master@77efb4d` — multiplexer merged, `py.typed` covered.
- ADR-0001 — OQ1 shared-codebook decision honoured throughout Sprints 1-2; Me3 operates on the *shared* alphabet.
- Planned ADR-0002 — v0.1 go/no-go (Task 3.11).
