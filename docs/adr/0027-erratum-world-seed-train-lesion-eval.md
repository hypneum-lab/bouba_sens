# ADR-0027 — Erratum: `train`, `lesion` and `eval` build different synthetic worlds

**Status:** Proposed — awaiting PI sign-off. Erratum; no value, metric,
threshold or past verdict is deleted or changed.
**Date:** 2026-10-30
**Decision source:** PI decision of 2026-10-30 ≈ 16:45 (add a corrected
mode, keep the historical one as default, run both arms in ADR-0026);
review fixes and replay decision of 2026-10-30 ≈ 18:00.
**Builds on:** ADR-0024 (erratum: the default regime does not learn).
**Artefacts:** `src/bouba_sens/cli.py` (`lesion --world-seed-mode`,
`--world-seed`; `eval --world-seed-mode`; `resolve_lesion_world_seed`,
`config_world_seed`), `src/bouba_sens/world/base.py` (`world_sha256`),
`scripts/run_grid.sh`, `scripts/run_grid_probe_v2.sh` (`WORLD_SEED_MODE`),
`tests/unit/test_world_seed_mode.py`,
`tests/unit/test_grid_world_mode_guard.py`.

## Fact

The CLI builds the world with a different seed in each command:

| command | world seed | line (main `a1fd6ff`) |
|---|---|---|
| `train` (phase 1) | the training seed s | `cli.py:296`, `_build_world(world, seed)` |
| `lesion` (phase 2, v1 and v2 probes) | **0**, always | `cli.py:486`, `_build_world(world, seed=0)` |
| `eval` (`me1_probe` rebuild, paired-run Me1) | the **lesion** seed (`config.yaml` `seed`, e.g. s · 10000 + count) | `cli.py:631`, `_build_world(world_name, seed)` |

For the synthetic worlds the world seed fixes the content: Gaussian, XOR
and Sinusoid draw their projections from it, and the Studyforrest mock
draws its whole stream from it (`_build_mock_cache(n, seed)`).
`tests/unit/test_world_seed_mode.py` checks that the fingerprint
(`world_sha256`, a fixed 256-item batch) differs between seeds 0 and 7
for all four. So, for a training seed s ≠ 0, a T2 network is trained in
phase 1 on one world and then adapted, scored (Me1, Me2, Me3) and probed
(v1, v2) on another. `me1_probe` in `eval` is computed on a third world
for every cell, seed 0 included.

## Dates (`git blame`, `git log -S`)

- `d900964` (2026-04-20 17:59 +0200, "Task 3.8 five-command CLI"):
  `train` uses `GaussianWorld(seed=seed)` and `lesion` uses
  `GaussianWorld(seed=0)`. This is the origin.
- `fee06cd` (2026-04-20 20:29 +0200, "Task 6.1 --world flag"): both
  calls move into `_build_world` and keep their seeds; current lines 296
  and 486 are blamed on it. XOR, Sinusoid and later the Studyforrest mock
  inherit the defect.
- `ea31ffa` (2026-04-21 09:02 +0200, ADR-0007 Phase B): the `eval`
  rebuild uses the lesion seed.

Every tag from `v0.1.0` (2026-04-20) to `v0.5.9`, and `main` at v0.5.10, include it.

## Scope

**Affected:** every grid of the repository on a synthetic world (Gaussian,
XOR, Sinusoid, Studyforrest mock) run through `train` → `lesion`:

- T2 cells of every seed s ≠ 0. The default seeds 0–4 have four of five
  seeds affected; the ADR-0022 / ADR-0023 grids (100–101, 200–204) and
  the straight-through lesion × query grid (300–309, the only grid
  whose network learns, ADR-0024) are affected for all their seeds.
  B-1 (Me7 = T1 − T2 gap), B-2 (Me3), B-3 (Me6 v1 / v2), the Amedi
  curve, the lock / gating / codebook scans, N12 and Q3 are all built on
  such cells.
- The v2 positive control (`train`, world s) and the v2 B-3 probes
  (`lesion`, world 0) of ADR-0023 read different worlds.
- `me1_probe` (`eval`) of every synthetic cell.

**Not affected:**

- T1 cells. They have no phase 1 and are trained, scored and probed in
  `lesion`'s world 0. Their `me1_probe` is still on the lesion-seed world.
- T2 cells of seed 0 for training, adaptation and the v1 / v2 probes
  (train and lesion both use world 0). Their `me1_probe` is still
  affected: `eval` rebuilds the world with the lesion seed, not 0.
- Phase-1-only measurements: the emission pilot (seeds 110–111) and the
  phase-1 losses of ADR-0024.
- Real-data worlds: Studyforrest real (2- and 5-modal), CCFilm and
  ActionSense. Their content is loaded from the cache files and the seed
  is only stored, never used (`test_real_worlds_do_not_depend_on_the_world_seed`:
  the same fingerprint for seeds 0, 7 and 400). The v0.7 ActionSense
  study is therefore not affected by this defect.
- Analyses that only read saved outcomes (ADR-0021 and ADR-0022 nulls,
  aggregators) and the ADR-0001 spike.

**Size of the effect.** Under the default argmax-bridge emission the
network does not learn (ADR-0024), so a world change between phases has
little to transfer, and the requalification of ADR-0024 already
applies. Under straight-through (Gaussian, seeds 300–309) a T2 network is
adapted on a world with different projections from the one it learned:
"late-acquired" then also means "moved to a new world". Its size is not
measured here.

## Correction (opt-in, default unchanged)

- `lesion --world-seed-mode {legacy,cell}`, default `legacy` (seed 0,
  bit-identical to today). `cell` uses the phase-1 seed: it is read from
  `--ckpt/metadata.json` for T2, given by `--world-seed` for T1 (required
  there), and the two must agree when both are present. `cell` records
  `world_seed_mode`, `world_seed` and `world_sha256` in `metadata.json`
  and `config.yaml`.
- `eval --world-seed-mode {legacy,cell}`. The default follows the mode
  recorded in `config.yaml` (legacy if absent, which covers every past
  run); paired-run Me1 follows it too. In `cell` mode the rebuild uses
  the recorded `world_seed` and `eval_report.json` gets the same three
  keys. Nothing is added in legacy mode. Paired mode
  (`--t1-ckpt` / `--t2-ckpt`) rebuilds both cells with the same mode,
  refuses a T1 / T2 pair whose modes differ, and in `cell` mode records
  `world_seed_mode`, `world_sha256_t1` and `world_sha256_t2`.
- `WORLD_SEED_MODE` (default `legacy`) in `run_grid.sh` and
  `run_grid_probe_v2.sh`. In `cell` mode it passes
  `--world-seed-mode cell --world-seed <grid seed>` to every lesion cell.
  On resume, a finished cell whose `config.yaml` records another mode
  (legacy if absent) stops the script with an error: one `OUT_ROOT` per
  arm. Phase 1 needs no guard, since `train` does not read the mode.
- Tests (`tests/unit/test_world_seed_mode.py`, Gaussian and mock):
  explicit `legacy` equals the default byte for byte (probe files,
  `metadata.json`, `config.yaml`, `eval_report.json`, `model.pt`). In
  `cell` mode, `train` (world s), `lesion` T2 and T1 and `eval` share one
  world fingerprint, which differs from the world-0 fingerprint, with
  the v2 probe on or off. A frozen reference
  (`test_default_reproduces_the_pre_adr0027_reference`) checks the
  default against outputs computed with the code before this erratum
  (`a1fd6ff`, electron-server): lesion world fingerprint, sha256 of the
  per-item and per-query files, metrics (dyadic values exactly,
  `me3_delta` to 10⁻⁶), paired-run Me1 / Me7. `model.pt` is not frozen,
  as float weights may differ in the last bits across CPU instruction
  sets; it was byte-identical between `a1fd6ff` and this branch on
  electron-server. `tests/unit/test_grid_world_mode_guard.py` runs both
  grid scripts with a fake `uv` (`bash -n`, pass-through, resume guard).

## Replays (PI decision 2026-10-30 ≈ 18:00)

- The straight-through lesion × query grid (Gaussian, seeds 300–309,
  `docs/prereg/2026-10-26-b3-lesion-interaction.md`) will be **replayed
  in `cell` mode after the ADR-0026 Me6 nulls**, as an **exploratory**
  analysis. It does not replace or reopen the confirmatory reading of
  that grid; it reports how far the legacy result depends on the world
  change.
- Every other historical grid receives this erratum only, with **no
  replay**.

## What this erratum does not change

No number in any ADR, report, paper section or README is changed or
deleted, and no verdict is reopened. Past results are additionally
qualified as *measured with phase 1 and phase 2 on different worlds for
T2 cells of seed ≠ 0 (synthetic worlds)*, on top of the ADR-0024
qualification. The default behaviour of every command and grid script
is unchanged. Any new grid states its `WORLD_SEED_MODE`; ADR-0026 runs
both arms.

## References

ADR-0007, ADR-0022, ADR-0023, ADR-0024, ADR-0026;
`docs/prereg/2026-10-26-b3-lesion-interaction.md`; PI decision
2026-10-30 ≈ 16:45.
