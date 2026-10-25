# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.10] — 2026-10-25 (seeded CLI + reproducibility erratum)

### Fixed

- **CLI runs are now byte-reproducible** (#35). `bouba-sens train` and
  `bouba-sens lesion` call `_seed_everything(seed)` before building any
  module: Python `random`, NumPy and the global PyTorch generator are
  seeded and `torch.use_deterministic_algorithms(True, warn_only=True)`
  is enabled. Before this fix the modality encoders and the
  `IntegrationHead` drew their initial weights from the unseeded global
  PyTorch generator, so two identical runs of a cell started from
  different weights (75 of 173 tensors differed after 20 steps,
  max |Δ| 1.13) and gave different `eval_report.json`. The autouse
  `_deterministic_seed` fixture hid this from the test suite; the new
  `tests/unit/test_cli_determinism.py` runs train → lesion → eval twice
  in subprocesses and compares the sha256 of `model.pt`,
  `per_query_me1.json` and `eval_report.json`.

### Erratum

- Paper §Artefacts claimed byte-identical `eval_report.json` on a fresh
  clone. That was false for every release up to v0.5.9, and every grid
  cited in paper v0.1 came from the unseeded pipeline: the cited numbers
  are reproducible in distribution across seeds, not byte for byte. The
  paragraph now states the v0.5.10 guarantee and carries a dated
  erratum; a re-run of `scripts/reproduce_paper_v01.sh` with the seeded
  CLI is under way.

### Changed (documentation only — phase 0 "truth", 2026-10-25)

- **"Pre-registered on OSF" withdrawn** from `CITATION.cff`, `README.md`,
  `.zenodo.json`, `CLAUDE.md`, `AGENTS.md`, `docs/paper/` and the
  `docs/osf/` amendment headers. B-1/B-2/B-3 are pre-registered locally
  (git-dated design spec, commit `87f723f`, 2026-04-20); OSF
  `10.17605/OSF.IO/Q6JYN` covers dream-of-kiki H1–H4 only. Issue #20.
- **B-3 requalified as an uncalibrated magnitude** (README F1, paper
  abstract and §4.4, `.zenodo.json`, notes in ADR-0008 / ADR-0009):
  an independent binomial null (ADR-0021) invalidates the reading of
  the bare exceedance of 0.02; no compatibility or rejection verdict
  before a paired calibration of the median. Values unchanged. Refs #22.
- **Paper §4.4** no longer says "Key claim, load-bearing"; aligned with
  the abstract. `.zenodo.json` no longer carries the withdrawn "minimal
  configuration" claim. Issue #21.
- **ADR index rebuilt** (`docs/adr/README.md`): the seven shared numbers
  (0005, 0007, 0014, 0015, 0016, 0017, 0019) are disambiguated a/b in
  the titles, file names unchanged; ADR-0006 and tags `v0.4.0` /
  `v0.5.0` are documented as absent from `main` (PI action). Issue #21.
- **`refs.bib` `amedi2007`** now carries the metadata of Amedi et al.
  2007, Nat Neurosci 10(6):687–689, `10.1038/nn1912` (Crossref) instead
  of Amedi et al. 2003 (`10.1038/nn1072`). Key unchanged. Issue #25.

### Added

- **Me6 paired null** (`scripts/me6_paired_null.py`,
  `reports/me6_paired_null.json`, ADR-0022 *Proposed*). The B-3 median is
  compared with a paired sign-flip null built from the saved per-item
  probe outcomes: for every trio, pair (m, q) and item i the outcomes
  `s_mq(i)` and `s_qm(i)` are swapped with probability 1/2 (H0: symmetry
  of the expected performances), then Me6 per trio and the median over
  trios are recomputed (B = 20 000, seed 25102026, numpy only). Reports
  the observed median, its null mean / P99 / tail probability, the same
  for every trio maximum, and the ADR-0021 iid reference at the same K,
  n and p. Follows the 2026-10-25 cross-review gate: no compatibility or
  rejection verdict before a paired calibration.
- **Per-item probe outcomes** (`AdaptationLoop.query_successes`,
  `successes_to_accuracy`; `bouba-sens lesion` writes
  `per_query_successes.json` next to `per_query_me1.json`).
  `query_accuracy` is now the float32 mean of `query_successes`, bit-
  identical to the previous value for every batch size (non-regression
  test). The five query modalities of a cell share one probe batch
  (`seed + 777`); the docstring now says so.
- **Me6 null calibration** (`scripts/me6_null_calibration.py`,
  `reports/me6_null_calibration.json`, ADR-0021 *Accepted*). Simulates
  the B-3 statistic when the 25 accuracies of the 5x5 matrix are equal
  and differ only by iid binomial noise at the production probe size
  n = 64 (an idealized reference: shared probe batches and the
  contiguous Studyforrest windows are not modelled). For base
  accuracies 0.2-0.5 the median-over-30-trios Me6 is 0.13-0.16 under
  this reference (P99 0.15-0.19) and the 0.02 threshold is cleared by
  every simulated grid at n = 64 and 256. Synthetic-world medians
  (0.141-0.148) show no upper-tail excess over it; mock (0.3125) and
  ECG (0.4453) exceed its P99, but the reference is not a valid null
  for their contiguous probes. No metric, verdict or threshold changed.
- **Subspace-shift metric** (`src/bouba_sens/metrics/subspace.py`).
  Top-k principal-direction shift S between pre/post-lesion codes, with
  rank checks, split-half noise floor and eigengap report, as drafted in
  amendment v0.8 (PR #9). Not yet used by any run.
- **Ross (2014) mixed discrete-continuous k-NN MI estimator**
  (`src/bouba_sens/metrics/mi_ross.py`): `mi_ross_discrete_continuous`
  (bits), `_nats` helper, duplicate counter. Primary B-2 estimator of the
  v0.7 amendment draft; not yet wired into the E1 runner.
- **Configurable Me3 probe — full representation `(B, 896)` default**
  (`src/bouba_sens/loop.py`, `src/bouba_sens/metrics/mi_migration.py`).
  The Me3 sonde pooling is now configurable; the default is the full
  fused representation instead of the 1-D mean-pool. The old 1-D
  mean-pool **pinned MINE to exactly 0.000 on every seed** — a
  probe-shape artefact (the Donsker-Varadhan critic floors on a scalar
  probe), **not** a genuine two-estimator agreement. `mi_migration` now
  handles multi-D codes for MINE/Kraskov.
- **Two real bridges with honestly-labelled per-channel provenance.**
  - **Studyforrest real bridge** (`scripts/fetch_studyforrest_sample.py
    --source openneuro`, `StudyforrestRealWorld` mode `real5`): real
    **OpenNeuro ds000113** source over the public S3 REST API (the dead
    phase-1 `studyforrest.org` URL is retired). **force / gravity /
    tactile / labels = REAL** ds000113 signal; **audio / vision =
    annotation PROXIES** (the copyrighted soundtrack/film are absent from
    OpenNeuro — labelled as proxies, never "real movie vision").
  - **CC-film bridge** (`scripts/extract_ccfilm_features.py`,
    `CCFilmWorld` mode `real_av`, `src/bouba_sens/world/ccfilm.py`):
    Big Buck Bunny (Blender Foundation 2008, **CC-BY 3.0**). **audio
    (128-D log-mel) / vision (16×16 frame patches) / labels (shot-cuts)
    = REAL** perceptual signal; **tactile / gravity / force = SYNTHETIC**
    (a film has no proprioception). `cli._build_world` wiring +
    `tests/unit/test_ccfilm_world.py`.
  - The two bridges carry the real signal on **opposite halves** of the
    modality set.

### Methodology

- **B-2 re-estimated and confirmed Null (ADR-0019).** Closes
  EXPERIMENT-BACKLOG E1 + E2. The Me3 probe has ambient dimension 896 but
  **effective dimension D ≈ 12** (participation ratio 11.7 / two-NN 12.4,
  agreeing across 5 seeds). The under-derived "probe batch ≥128"
  heuristic is replaced by the estimator-aware bound **N ≳ f(D)** (MI
  sample complexity scales with intrinsic, not ambient, dimension).
  Re-estimating B-2 = MI_post − MI_pre with **MINE on the full 896-D
  probe** (the v0.2 "richer probe rescues B-2" hypothesis) surfaces **no
  positive signal**; Kraskov and adaptive binning bracket zero, and
  binning **collapses in 896-D** exactly as the `N ≫ (#bins)^D` argument
  predicts. **No estimator crosses the +0.10 threshold on any seed**, on
  the gaussian reference **and on both real bridges** (Studyforrest +
  CC-film) — the null is robust to which half of the modalities is real.
  Verdict **B-2 Null confirmed/renforced**, not overturned. Artefacts:
  `reports/e1_b2_dimensional_mine_studyforrest.json`,
  `reports/e1_b2_dimensional_mine_ccfilm.json`,
  `scripts/spikes/e1_b2_dimensional_mine.py`,
  `scripts/spikes/e1_b2_full_sonde.py`. Paper draft §2.3 / §6.3.1(a–e)
  and the abstract are aligned to the per-channel REAL/PROXY/SYNTHETIC
  provenance.

- N8/N9-Q3+ 10-seed verdict `Retract` **FINAL** (2026-05-11) : the
  §5.5 single-seed non-monotone B-1 peak at LOCK_AFTER=100
  (originally reported as +0.0125 on seed=0) does NOT robustly
  reproduce across 10 seeds (5 base + 5 plus =
  {0,17,42,73,101,7,23,31,53,89}). Per critic MAJOR #4, the
  verdict is now computed via **per-seed quadratic regression**
  (`y = a + b*x + c*x^2`, test `c < 0`), not Jonckheere-Terpstra
  (wrong statistical instrument for an inverted-U hypothesis).
  Only 5 of 10 seeds yield concave-down fits (c<0) ; sign-test
  binomial p=0.62 ; one-sample t-test on mean(c)<0 yields p=0.28.
  Both fail to reject H0 at α=0.05. Median peak position
  (argmax) is LOCK_AFTER=100 but drift is 100% (range [50, 150]).
  Pre-registered decision criteria `Retract` **CONFIRMED**.
  **TMLR submission blocked** until §5.5 reformulation lands and
  ADR-0019 is drafted. Reusable analysis script :
  `scripts/analyse_amedi_quadratic.py`. Final verdict JSON :
  `reports/v0.5_amedi_curve_multiseed_10seed_final.json`. See
  `docs/milestones/q3-amedi-seeds-2026-05-10.md` Q3+ closeout
  section. (Supersedes 5-seed interim notice.)
  - **Pooled v2 subgroup discovery (2026-05-11)** : follow-up
    pooled subgroup-discovery analysis (150 me7 per LOCK_AFTER,
    15 sub-pops modality × SNR) confirms the pooled non-monotone
    signature is NOT present (quadratic c=-2.1e-6, p=0.19, peak
    at LOCK≈148.9). Two sub-populations exhibit naïve-significant
    concave-down fits in the original Amedi window : tactile-floor
    (c=-2.107e-5, uncorrected p=0.020, peak@94.6) and force-plus10
    (c=-1.786e-5, uncorrected p=0.048, peak@106.9). Neither
    survives Bonferroni correction across 15 subgroups
    (α=0.0033). Hypothesis-generating only ; pre-registered
    targeted replication queued as N12 sprint (N=20 seeds, only
    2 PRE-REGISTERED subgroups → Bonferroni α=0.025). Artefacts :
    `reports/v0.5_amedi_curve_pooled_v2.json`,
    `scripts/analyse_amedi_pooled_subgroup.py`. Pre-registration :
    `docs/milestones/n12-amedi-subgroup-replication-2026-05-11.md`
    (mirrors `HYPNEUM-PLANS/preregistrations/n12_amedi_subgroup_replication.md`).
    §5.5 reformulation drafts (3 versions A/B/C conditional on
    N12 verdict) at `docs/paper/§5.5-reformulation-draft.md`.
  - N12 subgroup replication (2026-05-11) : 0/2 pre-registered subgroups
    (tactile+floor, force+plus10) replicate at Bonferroni α=0.025 on 10 NEW
    seeds. tactile-floor sign inverted (c=+2.14e-6, p=0.595) — winner's curse
    manifest. §5.5 retract confirmed at subgroup level. ADR-0019 closed.

## [0.5.9] — 2026-05-10 (TMLR submission package + Zenodo refresh)

### Added

- `docs/paper/cover-letter-tmlr.md` — TMLR submission cover letter
  with Reproducibility track framing, narrow-and-falsifiable claim
  positioning, open-science statement, and concurrent-work
  paragraph addressing RecursiveMAS (Yang et al. arXiv:2604.25917).
- Concrete 4-person reviewer panel (Cees Snoek UvA, Tessa Dekker
  UCL, Jessica Hullman Northwestern, Alessandro Achille AWS/Caltech)
  with selection criteria (publication recency, no co-authorship,
  institutional+topical diversity).
- Nested `CLAUDE.md` files at root and under `configs/`, `scripts/`,
  `src/`, `tests/` for Claude Code progressive disclosure.

### Notes

- `version: "0.5.9"` in `CITATION.cff` and tag pushed to fire the
  GitHub→Zenodo webhook with the up-to-date `.zenodo.json`
  metadata. Companion to nerve-wml v1.8.1 release of the same day.
- The previously local-only `v0.5.8` tag (commit `4eab231`) is
  superseded by this release; the remote `v0.5.8` tag (commit
  `30bc7ba`, distinct content) is preserved untouched.

## [0.3.0] — 2026-04-20 (Sprint 6 T6.1+T6.2 — cross-world B-3 replicated)

### Added

- **Task 6.1** — `--world {gaussian,xor,sinusoid}` flag on both
  `bouba-sens lesion` CLI and `scripts/run_grid.sh` (merged in
  `fee06cd`).
- **Task 6.2** — cross-world replication of the full 150-cell
  grid on XOR and Sinusoid worlds. Aggregates written to
  `reports/v0.3_{xor,sinusoid}_aggregate.json` (30 cells each).
- ADR-0005 `docs/adr/0005-v03-cross-world-verdicts.md` — records
  the cross-world verdicts and interpretation.

### Studio replication (Task 6.2)

Both grids launched concurrently on Studio (M3 Ultra) with the
same configuration as v0.2 (`STEPS_TRAIN=200`, `STEPS_LESION=100`,
`METRICS="Me1,Me2,Me3"`), 17 min wall-clock each in parallel.

### Cross-world verdict (from ADR-0005)

| Invariant | gaussian | xor | sinusoid | 3/3 worlds ? |
|-----------|---------:|----:|---------:|:---:|
| B-1 (Me7 > 0.05) | -0.0063 | -0.0062 | **+0.0125** | 3x FAIL, sign flips on sinusoid |
| B-2 (Me3 delta > 0.10) | 0.0275 | 0.0034 | 0.0019 | 3x FAIL, decays Gaussian > XOR > Sinusoid |
| B-3 (Me6 max-abs > 0.02) | 0.1484 | 0.1406 | 0.1406 | **3x PASS (first cross-world replicated PASS)** |

### Headline finding

**B-3 is now world-agnostic.** The perceptive / proprioceptive
asymmetry is robust across three structurally different synthetic
worlds at ~7-8x the pre-registered threshold. B-1 shows a world-
topology-dependent sign pattern (negative on orthogonal-factored
worlds, positive on circular-latent) — seeds hypothesis H-B1 for
a future OSF amendment.

### Changed

- Package `version`, `_version.__version__`, and version-pinned
  test assertions (`tests/smoke/test_imports.py`,
  `tests/unit/test_smoke.py`) bumped to `0.3.0`.

### Still deferred to Sprint 6+

- Paper v0.1 draft (manuscript, figures, TMLR / NeurIPS D&B).
- OSF amendment filing for hypothesis H-B1 (world-topology-
  dependent critical-period effect).
- Possible B-2 threshold recalibration keyed on baseline
  MI(modality; label) — contingent on reviewer feedback.
- Denser world battery (Torus-world, ManifoldWorld, linguistic
  micro-world) for Sprint 7.

## [0.2.0] — 2026-04-20 (Sprint 5 close — real B-1/B-2/B-3 verdicts, B-3 PASS)

### Added

- Sprint 5 plan (`docs/superpowers/plans/2026-04-20-bouba-sens-sprint5.md`).
- Per-cell `me3_delta`: `AdaptationLoop.lesion_phase` now captures
  pre-lesion and post-lesion probe codes + labels on the returned
  `AdaptationReport`; `eval` CLI wires them into `me3_delta` so
  B-2 is evaluated against real data (Task 5.1).
- Aggregation-side Me6 and Me7 — `scripts/aggregate_grid.py` builds
  a 5x5 per-query accuracy matrix per `(seed, timing, SNR)` and
  feeds it to `me6_asymmetry` + `me6_max_abs_off_diag`; Me7 is
  now paired via `(seed, modality, SNR)` across T1 and T2 runs
  (Task 5.2).
- `AdaptationLoop.query_accuracy(query_modality, ...)` — single-
  query Me1 probe used by `run_grid.sh` to emit a per-cell
  `per_query_me1.json` that feeds the aggregator's 5x5 matrix.
- `tests/unit/test_aggregate_grid.py` + `tests/integration/
  test_v02_smoke.py` + extended `tests/empirical/
  test_grid_structural.py` asserting `cells_counted >= 6` on
  b1 / b2 / b3 (Task 5.3).
- ADR-0004 `docs/adr/0004-v02-invariant-verdicts.md` — real
  v0.2 empirical verdicts (see "Studio grid v0.2" below).

### Changed

- `scripts/run_grid.sh` emits `Me1,Me2,Me3` per cell (Me7 and Me6
  moved to the aggregator — they are aggregation-level metrics).
- Package `version`, `_version.__version__`, and version-pinned
  test assertions (`tests/smoke/test_imports.py`,
  `tests/unit/test_smoke.py`) bumped to `0.2.0`.

### Studio grid v0.2 (Task 5.4)

- 150 / 150 cells processed on Studio (M3 Ultra, commit `242b292`),
  `STEPS_TRAIN=200 STEPS_LESION=100`, ~17 min wall time.
- Aggregate : 30 cells, written to `reports/v0.2_aggregate.json`.

### v0.2 empirical verdict (from ADR-0004)

| Invariant | Threshold | Median | Cells | Passes |
|-----------|-----------|--------|-------|--------|
| B-1 (Me7 > 0.05) | 0.05 | -0.0063 | 75 | No |
| B-2 (Me3 delta > 0.10) | 0.10 | 0.0275 | 30 | No |
| B-3 (Me6 max-abs > 0.02) | 0.02 | 0.1484 | 30 | **Yes** |

**1 / 3 invariants pass with real data.** B-3 (perceptive /
proprioceptive asymmetry) is confirmed at 7.4x the pre-registered
threshold. B-1 is directionally falsified (Me7 slightly negative
on GaussianWorld). B-2 shows a positive but under-threshold effect
(~3.6x below 0.10). No threshold changes, no metric-math changes —
only CLI coverage was closed between v0.1 and v0.2, so the v0.2
verdict is the scientific one.

### Still deferred to Sprint 6+

- XOR-world + Sinusoid-world replication pass (triangulate whether
  the B-1 reversal is GaussianWorld-specific).
- Me3 Kraskov estimator calibration review if reviewers flag the
  3.6x gap as methodologically load-bearing.
- Paper v0.1 draft (manuscript, figures, TMLR / NeurIPS D&B).
- Interactive plotly heatmap for Me6 and recovery curves for Me2.

## [0.1.1] — 2026-04-20 (Sprint 4 close — 150-run Studio grid + ADR-0003)

### Added

- Sprint 4 plan (`docs/superpowers/plans/2026-04-20-bouba-sens-sprint4.md`).
- Hydra grid configs (`configs/{grid,timing,modality,snr}/*.yaml`) —
  flat overrides via `@package _global_` for 5x5x2x3 composition.
- `scripts/run_grid.sh` — idempotent orchestrator for the 5-seed x
  5-modality x 2-timing x 3-SNR = 150-cell grid. Reuses a single
  Phase 1 pretrain per seed across T2 cells; skips pretrain for T1.
- `scripts/aggregate_grid.py` — bootstraps `me9` across seeds per
  cell and packs B-1/B-2/B-3 invariant verdicts into
  `reports/v0.1_aggregate.json`.
- CLI `lesion` extended with `--timing`, `--snr-init`, `--snr-floor`,
  `--k-steps`; `ckpt` optional (None -> T1 congenital).
- `tests/empirical/test_grid_structural.py` — 4 structural tests
  over the post-Studio aggregate artifact (30-cell shape, finite
  bootstrap stats, invariant packing, threshold fidelity).
- `.github/workflows/full-benchmark.yml` — self-hosted Studio
  runner workflow (`workflow_dispatch` only; cron deferred).
- ADR-0003 recording v0.1 empirical verdicts (B-1/B-2/B-3 all
  NO-GO, driven by v0.1 CLI coverage gaps, not engine failure).

### Fixed

- `me9_bootstrap` collapses CI onto the mean when input is
  degenerate (zero range) — scipy BCa returns NaN otherwise, which
  broke the empirical-finite structural test.

### Studio grid

- 150/150 cells completed in ~14 min on MacStudio (M3 Ultra).
- 30 unique cells aggregated with finite (mean, ci_low, ci_high)
  triples for every metric (post-fix).
- All three invariants report `passes=false` with honest
  attribution: me3/me6 absent from `eval_report.json` because the
  v0.1 CLI emits only `Me1,Me2,Me7`; me7 degenerate because the
  CLI computes `me1_T1 - me1_T1 = 0`. v0.2 scope set in ADR-0003.

## [0.1.0] — 2026-04-20 (Sprint 3 close — feature-complete v0.1 design)

### Added

- Sprint 3 plan (`docs/superpowers/plans/2026-04-20-bouba-sens-sprint3.md`).
- Seven v0.1 metrics covering the three B-1/B-2/B-3 invariants:
  - `me1_accuracy` + `me2_recovery_auc` over `AdaptationReport.accuracy_curve`.
  - `me3_mi` + `me3_delta` via sklearn Kraskov kNN estimator.
  - `me6_asymmetry` + `me6_max_abs_off_diag` on the n×n lesion-query
    perf matrix.
  - `me7_congenital_gap` scalar on T1 / T2 perfs.
  - `me8_baselines` + `freeze_nerve_plasticity` +
    `random_rewire_nerve` for the three §4.4 regimes.
  - `me9_bootstrap` CI 95 % via `scipy.stats.bootstrap`.
- `metrics/__init__.py` `Metric` Protocol + `EvalReport` dataclass
  re-exporting all 11 callables.
- CLI (spec §5.3): `sim`, `train`, `lesion`, `eval`, `aggregate`
  via typer with parquet + pickle + JSON round-trips.
- `render_html` static-table summary per spec §5.4 (5 section
  headers; interactive plotly deferred to Sprint 4).
- Integration gate `tests/integration/test_v01_smoke_run.py`:
  4.78 s single-seed end-to-end on GrosMac, all metrics finite,
  Me1 > 0.25.
- `scripts/v01_smoke.sh` CLI wrapper chaining train -> lesion
  -> eval -> aggregate.
- ADR-0002 `docs/adr/0002-v01-go-no-go.md` — GO decision.

### Verified

- 115/115 tests pass under `uv run pytest` on GrosMac.
- `uv run ruff check src tests` clean.
- `uv run mypy src tests` clean.
- CLI round-trip: `bouba-sens sim | train | lesion | eval | aggregate`
  all exit 0.

### Still deferred to Sprint 4

- Full 150-run grid on Studio + Me9 bootstrap IC aggregation.
- Interactive plotly heatmap for Me6 and recovery curves for Me2.
- Hydra config-driven experiments (10-config timing × modality grid).
- `tests/empirical/` asserts on real run results (nightly CI).
- `full-benchmark.yml` activation.

### Note on versioning

`v0.1.0` is the first non-sprint tag (Sprint-0/1/2 were
`v0.0.2-sprint1` / `v0.0.3-sprint2`). Marks feature-completeness of
the v0.1 design — code + metrics + CLI + report. Paper v0.1 draft
remains a separate Sprint 5 concern.

## [0.0.3] — 2026-04-20 (Sprint 2 close)

### Added

- Sprint 2 plan (`docs/superpowers/plans/2026-04-20-bouba-sens-sprint2.md`).
- `PlasticityGate` — channel-wise softmax over 5 modalities (P1, Task 2.1).
- `AdaptiveCodebook` — Gumbel-softmax-capable projection,
  `nn.Parameter[A, d_hidden]`, optionally seeded from
  `mux.constellation` (P2, Task 2.2).
- `CrossModalTransducer` — directed-edge MLP with 0/1 gating
  (P3, Task 2.3); 20 instances in `nn.ModuleDict` keyed
  `"{src}->{dst}"`.
- `CrossModalNerve` — assembles P1+P2+P3, implements `fuse()`
  via soft demod (`mux.demodulate(hard=False)`), codebook
  projection, per-pair transducer activation per spec §4.3,
  and gate-weighted sum (Task 2.4). Plus `on_lesion()` and
  `migration_stats() -> MigrationReport` (Task 2.5).
- `LesionSpec` frozen dataclass + `m2_snr_schedule` per spec
  §3.4/§4.1 (Task 2.6).
- `LesionScheduler.apply()` — SNR-scaled additive Gaussian
  noise on the targeted modality, returns a new frozen
  `WorldSample` (Task 2.7).
- `IntegrationHead` — minimal `n_classes` linear classifier
  over mean-pooled `(B, K, d_hidden)` (Task 2.8).
- `AdaptationLoop.pretrain()` Phase 1 + `.lesion_phase()`
  Phase 2 with `AdaptationReport` trajectories and FIFO
  theta-replay buffer (Tasks 2.9-2.10).
- Integration acceptance gate `tests/integration/test_phase1_
  phase2_smoke.py` — 100 + 100 steps, no NaN, accuracy beats
  chance, audio gate drops ≥10 % from pre-lesion baseline
  (Task 2.11).

### Verified

- 76/76 tests pass under `uv run pytest` on GrosMac.
- `uv run ruff check src tests` clean.
- `uv run mypy src tests` clean.
- Sprint 2 acceptance gate runs in ~2 s.

### Still deferred to Sprint 3+

- Metric harness Me1/Me2/Me3/Me6/Me7/Me8/Me9.
- `cli.py` typer/rich interface.
- Jinja2 HTML report (`report.py`).
- Empirical tests (`tests/empirical/`).
- Full configs grid for (timing × modality) variants.
- Training runs on Studio.

### Note on versioning

Sprint 1 was tagged `v0.0.2-sprint1` but the `_version.py`
was not bumped from `0.0.1`. Sprint 2 jumps the package
version to `0.0.3` to honour the git tag semantics going
forward.

## [0.0.2] — 2026-04-20 (Sprint 1 close)

### Added

- Sprint 1 plan (`docs/superpowers/plans/2026-04-20-bouba-sens-sprint1.md`).
- `WorldSample` frozen dataclass + `@runtime_checkable` `WorldSimulator`
  Protocol (Task 1.1).
- `GaussianWorld` — N(0, I_32) latent, 5 orthogonally-projected modalities,
  4-class sign-pattern label (Task 1.2).
- `XORWorld` — Rademacher latent, 2-class parity label (Task 1.3).
- `SinusoidWorld` — circular-latent (sin, cos)-on-unit-circle with Gaussian
  noise on remaining 30 dims, 4-class quantisation label (Task 1.4).
- `test_pairwise_mi_below_0_2_bit` parametrised across all 3 worlds —
  verifies the orthogonal-factored projection scheme keeps pairwise
  modality MI under 0.2 bit (Task 1.5, spec §3.1 invariant).
- `SensoryWML(MlpWML)` — modality-typed subclass with shared-mux identity
  via `object.__setattr__` bypass to prevent `nn.Module` double-registration
  (Task 1.6; ADR-0001 shared-codebook).
- 5 modality-specific encoders (`AudioEncoder`, `VisionEncoder`,
  `TactileEncoder`, `GravityEncoder`, `ForceEncoder`) all emitting
  `(B, d_hidden=128)` and preserving the global `torch.get_rng_state()`
  (Task 1.7).
- `SensoryWML.step(x)` — encodes modality tensors into a PAC carrier via
  the shared multiplexer; differentiable softmax bridge restores gradient
  to `input_proj` through the otherwise-argmax-blocked path (Task 1.8).
- Integration smoke gate `tests/integration/test_five_modality_emission.py`
  — 5 modalities × shared mux → 5 carriers × demod → valid code indices
  (Task 1.9).

### Changed

- `track_p.multiplexer` dep → nerve-wml `master@77efb4d` (PR #2 merged,
  all Q1-Q5 arbitrated) with `GammaThetaMultiplexer`, `GammaThetaConfig`,
  `AWGN`, `HardwareJitterNoise`, `NoiseModel` exports + `py.typed`.
- §6.3 of the design spec updated to reflect the shipped multiplexer API
  (commit `cbab3e3`).

### Verified

- 53/53 tests pass under `uv run pytest` on GrosMac (Python 3.14.3, torch
  post-sync).
- All ruff + mypy pre-commit hooks green.
- Studio smoke import verified 2026-04-20: `T=175 samples` bin-aligned.

### Still deferred to Sprint 2+

- `CrossModalNerve.fuse` + `PlasticityGate` + `AdaptiveCodebook` +
  `CrossModalTransducer` (Sprint 2).
- `LesionScheduler` + M2/T3 SNR protocol (Sprint 2).
- `IntegrationHead` + 10-class classifier (Sprint 3).
- `AdaptationLoop` + θ-replay buffer (Sprint 2–3 split per spec).
- Metric harness (Me1/Me2/Me3/Me6/Me7/Me8/Me9) (Sprint 3).

## [0.0.1] — 2026-04-20

### Added

- Initial design spec (`docs/superpowers/specs/2026-04-20-bouba-sens-design.md`).
- Sprint 0 plan (`docs/superpowers/plans/2026-04-20-bouba-sens-sprint0.md`).
- Package skeleton `src/bouba_sens/` with module stubs for sensory, nerve,
  lesion, head, loop, cli, report, world/*, metrics/*.
- CLI entrypoint `bouba-sens version`.
- Local dependency on `nerve-wml` v0.1.0 (sibling workspace clone; not yet on
  PyPI) — pinned via `[tool.uv.sources]`.
- Smoke test `tests/smoke/test_nerve_wml_api.py` verifying the 4 required
  public symbols (spec §6.3): `nerve_core.protocols.Nerve`,
  `nerve_core.neuroletter.Neuroletter`, `track_w.mlp_wml.MlpWML`,
  `track_p.transducer.Transducer`. The 5th expected symbol
  (`GammaThetaMultiplexer`) is tracked as an upstream gap (issue #1).
- Pre-commit hooks (ruff v0.15.11, mypy v1.20.1, hygiene hooks).
- GitHub Actions: `ci.yml` (lint + mypy + unit/smoke tests on Python 3.14)
  and `full-benchmark.yml` (nightly stub, activated in Sprint 3).
- OQ1 spike script (`scripts/spikes/oq1_codebook.py`) — execution deferred
  to Studio (M3 Ultra) per compute-routing directive.
- ADR-0001 stub (`docs/adr/0001-codebook-sharing.md`, status = Proposed /
  execution pending).
- Hydra-style config skeleton (`configs/v0.1_intact.yaml`) with Me9 in
  `metrics.enabled` and a conservative `codebook_shared=true` default.
- LICENSE (MIT), README, CITATION.cff, SECURITY.md.

### Deferred to Sprint 1+

- OQ1 spike execution (requires Studio).
- All implementation bodies — every module stub raises `NotImplementedError`.
- Empirical tests (`tests/empirical/`) and property tests (`tests/property/`).
