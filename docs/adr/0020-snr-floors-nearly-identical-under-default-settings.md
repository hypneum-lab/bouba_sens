# ADR-0020 — SNR floors were nearly identical under default settings

**Status:** Accepted — factual audit. This ADR modifies no existing
result, verdict or locked pre-registration.
**Date:** 2026-10-23
**Sprint:** post-v0.5.9 audit (PI decision 2026-10-23: "verify then
correct")

## Context

The three SNR cells of the 150-cell grid (`floor`, `minus10`,
`plus10`) are meant to be three lesion severities (spec §4.1 and the
grid definition: floors -20 / -10 / +10 dB). A review of `origin/main`,
confirmed independently by Codex, found that the default schedule never
gets near those floors.

`m2_snr_schedule` (`src/bouba_sens/lesion.py:41-57`) is

    SNR(t) = max(floor, init + (floor - init) * min(1, t / k))

with defaults `init = 20 dB` and `k = 5000` (spec §4.1: "K = 5 000
steps"). `bouba-sens lesion` (`src/bouba_sens/cli.py:266-268, 359-360`)
exposes `--snr-init/--snr-floor/--k-steps` with those defaults, and
`AdaptationLoop.lesion_phase` calls the schedule with the loop index
`step = 0 .. steps-1` (`src/bouba_sens/loop.py:213-215`, applied by
`LesionScheduler.apply`, `lesion.py:78-86`). The grid driver
(`scripts/run_grid.sh:9, 98-106`) uses `STEPS_LESION=100` and passes
only `--snr-floor`; it never passes `--k-steps` or `--snr-init`. Over
100 lesion steps the ramp therefore covers 99/5000 = 1.98 % of its
length:

| nominal floor | SNR at step 0 | SNR at step 99 (last) | mean over steps 0-99 | noise std / signal std at step 99 |
|---|---|---|---|---|
| -20 dB (`floor`)   | 20.00 | **19.21** | 19.60 | 0.1095 |
| -10 dB (`minus10`) | 20.00 | **19.41** | 19.70 | 0.1071 |
| +10 dB (`plus10`)  | 20.00 | **19.80** | 19.90 | 0.1023 |

(`noise_factor = 10^(-SNR/20)`, `lesion.py:80`.) The nominal floor is
never reached. The three conditions differ by at most 0.6 dB, about 7 %
in relative noise amplitude, and all sit in a mild-noise regime (noise
about 10 % of the signal std). `metadata.json` / `config.yaml`
(`cli.py:386-392, 421-430`) record the nominal `snr_floor` and `k_steps`
but not the SNR actually reached, so a reader of `snr_floor` alone would
believe the floors were reached.

## Audit method

For each published or archived result the paper and ADRs rely on, the
schedule in effect was reconstructed from (a) the commit recorded in the
ADR or milestone, (b) `scripts/run_grid.sh`, `src/bouba_sens/cli.py` and
`lesion.py` **at that commit** (`git show <sha>:<path>`), and (c) the
`STEPS_LESION` value in the ADR config table. Checked at every recorded
grid commit (1dc5822, 242b292, fee06cd, 579e981, 0673a2b, 483f47c,
d7a3645, eb8dee5, 06e6efb, 667b34a, 9f53eaa, 4869dcd, fe11b4d,
db0fc80): `cli.py` has the `k_steps` default 5000, `m2_snr_schedule`
has `k: int = 5000`, and `run_grid.sh` passes `--snr-floor` but never
`--k-steps` / `--snr-init`. The `--snr-floor` flag exists since 3269e0b
(the first grid orchestrator), so no grid ran with the floor unwired.
The schedule formula has not changed since ef715ac.

**Limits of the evidence.** The committed `reports/*.json` aggregates
and `reports/MANIFEST.json` do not record the schedule, and the per-cell
`metadata.json` / `config.yaml` (which do record `k_steps`) live in the
gitignored `runs/` trees, which are not archived in this repository.
Every `k_steps` value below is therefore **inferred from the code at the
recorded commit plus the absence of any override in the driver**, not
read from the cell artefacts. A grid executed from a locally modified,
uncommitted script would not be detected. Follow-up 1 turns this
inference into measurement.

## Audit table

Effective SNR = SNR at the last lesion step (step 99) for nominal floors
`floor / minus10 / plus10` = -20 / -10 / +10 dB; `init` = 20 dB unless
stated. ADR names are abbreviated by file (e.g. ADR-0005-v03 =
`0005-v03-cross-world-verdicts.md`).

| Artefact | Commit | k_steps | Lesion steps | Effective SNR (floor / minus10 / plus10) | Evidence |
|---|---|---|---|---|---|
| v0.1 150-cell grid, Gaussian (ADR-0003) | 1dc5822 | 5000 (inferred) | 100 | 19.21 / 19.41 / 19.80 | ADR-0003:19-20; `run_grid.sh@1dc5822`; `cli.py@1dc5822` |
| v0.2 grid, Gaussian (ADR-0004) | 242b292 | 5000 (inferred) | 100 | 19.21 / 19.41 / 19.80 | ADR-0004:28-29 |
| v0.3 XOR + Sinusoid (ADR-0005-v03, ADR-0005-cross-world) | fee06cd | 5000 (inferred) | 100 | 19.21 / 19.41 / 19.80 | ADR-0005-v03:21-22; ADR-0005-cross-world:30-31 |
| v0.4 Studyforrest mock (ADR-0008) | 579e981 | 5000 (inferred) | 100 | 19.21 / 19.41 / 19.80 | ADR-0008:32-35 |
| v0.4 real ECG (ADR-0009, headline retracted) | 0673a2b | 5000 (inferred) | 100 | 19.21 / 19.41 / 19.80 | ADR-0009:63-67 |
| B-1 plasticity lock, Gaussian lock=200 (ADR-0010) | 483f47c | 5000 (inferred) | 100 | 19.21 / 19.41 / 19.80 | ADR-0010:32-34 |
| Cross-world lock matrix, 3 grids (ADR-0011) | 483f47c | 5000 (inferred) | 100 | 19.21 / 19.41 / 19.80 | ADR-0011:22-24 |
| Real 4.5-modal sf45 nolock / lock (ADR-0012) | d7a3645 | 5000 (inferred) | 100 | 19.21 / 19.41 / 19.80 | ADR-0012:62-66 |
| Dose-response lock 50/100/200/400/800, 5 grids (ADR-0013) | eb8dee5 | 5000 (inferred) | 100 | 19.21 / 19.41 / 19.80 | ADR-0013:25-29 |
| Compound lock + Gumbel (ADR-0014-compound) | 06e6efb | 5000 (inferred) | 100 | 19.21 / 19.41 / 19.80 | ADR-0014-compound:32-36 |
| Gumbel tau scan, 4 grids (ADR-0015-gumbel) | 667b34a | 5000 (inferred) | 100 | 19.21 / 19.41 / 19.80 | ADR-0015-gumbel:22-26 |
| Finer tau + codebook freeze, 5 grids (ADR-0016-finer) | 9f53eaa | 5000 (inferred) | 100 | 19.21 / 19.41 / 19.80 | ADR-0016-finer:27-31 |
| Sprint 14 joint HARD to GUMBEL grid (ADR-0017-sprint14) | not stated (script identical at fe11b4d and 63efb7b) | 5000 (inferred) | 100 | 19.21 / 19.41 / 19.80 | ADR-0017-sprint14:120-124 |
| B-3 partition control and follow-ups (ADR-0014-b3, ADR-0015-lemma, ADR-0016-next-gen, ADR-0017-mean, ADR-0018) | 4869dcd (pipeline) | inherits source grids | 100 | inherits source grids (re-analysis, no new lesion run) | ADR-0014-b3:27-28; ADR-0017-mean "Empirical re-aggregation" |
| Q3 (5 seeds) and Q3+ (10 seeds) via `replay_amedi_seeds.py` and `run_grid.sh` | db0fc80 | 5000 (inferred) | 100 | 19.21 / 19.41 / 19.80 | `scripts/replay_amedi_seeds.py:65-80` (env sets only SEEDS, OUT_ROOT, WORLD, LOCK_AFTER); q3-amedi-seeds:42; q3plus-10seeds:5 |
| N12 replication (10 new seeds, 1500 cells) | db0fc80 | 5000 (inferred) | 100 | 19.21 / 19.41 / 19.80 | n12-amedi-subgroup-replication:132 |
| ADR-0019-multi-seed Q3 retract, pooled-v2 subgroups, `reports/v0.5_amedi_*.json` | db0fc80 (underlying grids) | inherits Q3 / Q3+ / N12 | 100 | 19.21 / 19.41 / 19.80 | ADR-0019-multi-seed:20; `scripts/analyse_amedi_pooled_subgroup.py:23` |
| ADR-0019-b2 dimensional MINE E1.3a/c/d (`e1_b2_dimensional_mine.py`) | 78c7a86 / fb7af00 | 100 (override, = lesion steps) | 100 | single schedule, no 3-SNR factor: init 0 dB, floor -20 dB, **-19.80** at step 99 | `scripts/spikes/e1_b2_dimensional_mine.py:98, 175` |
| ADR-0019-b2 full-sonde replication (`e1_b2_full_sonde.py`) | 78c7a86 | 40 (override) | 100 (default) | init 1.0 dB; nominal floors 0.0 / 0.1 / 0.3 dB **reached at step 40**; floors differ by at most 0.3 dB by design | `scripts/spikes/e1_b2_full_sonde.py:40, 79, 125` |
| Paper numbers (`docs/paper/*`, `main.tex`, README, CHANGELOG) | derived | as the grids above | 100 | as the grids above | none states the schedule; `main.tex:50,107,432` say "3 SNRs" |
| Wrappers `v01_smoke.sh`, `reproduce_paper_v01.sh`, `run_grid_with_partition_control.sh` | HEAD | 5000 (code default) | 100 | smoke (single run, floor -20): 19.21; grid wrappers: 19.21 / 19.41 / 19.80 | `v01_smoke.sh:8-9, 21-24`; `reproduce_paper_v01.sh:53`; `run_grid_with_partition_control.sh:54-55` |
| Phase 1 pretrain (200 steps) | all | n/a | n/a | no SNR schedule applied | `loop.py:110-130` |

At the artefact level the schedule is "unknown" for every row (no
artefact records it); the values above are reconstructions. Only the
two E1 spikes used a ramp that reaches its floor.

## What this does and does not change

It does **not** change any published number, ADR verdict or locked
pre-registration, and this ADR edits none of them. The runs are valid
descriptions of what was executed: 5 seeds x 5 modalities x 2 timings x
3 labelled SNR cells.

It does change how the SNR factor may be described:

- The 3-level SNR factor provided almost no manipulation. Every
  150-cell grid exercised one mild-noise regime three times, with at
  most 0.6 dB between the extreme labels. The labels `floor`,
  `minus10`, `plus10` name nominal floors that were not reached.
- Any statement of robustness or invariance "across 3 SNRs" (e.g.
  `main.tex` "3 SNRs", subgroup language such as "tactile+floor" and
  "force+plus10", and B-1/B-2/B-3 medians computed per SNR) must be
  qualified: the three SNR cells are near-replicates, differing in the
  lesion seed (`seed * 10000 + count` in `run_grid.sh`) and a schedule
  offset of at most 0.6 dB, not distinct severities. Per-SNR subgroups
  were closer to extra draws than to a dose factor, which also bears on
  the effective-N reading of the 15-subgroup analyses.
- B-1 "at SNR_floor" (spec, Me7) was evaluated at about 19.2 to 19.8 dB
  instead of the floor named in the spec. A null there says nothing
  about lesions reaching -20 dB, and no conclusion about severe lesions
  can be drawn from these grids.
- The spec's K = 5000 ramp assumed lesion phases of at least K steps;
  the grids used 100.

Unchanged: comparisons between T1 and T2, worlds and LOCK_AFTER values
at fixed schedule are internally consistent (same schedule on every
arm). Only conclusions that depend on the SNR dimension need the
qualification.

## Proposed follow-ups (none done here)

1. Read `metadata.json` / `config.yaml` (`k_steps`, `snr_floor`) from
   the original run directories (Studio `runs/`, kx6tm-23 Q3 and N12
   runs) to replace "inferred" by measured values, and archive them.
2. Code: make the default schedule reach the floor within the lesion
   phase (for instance `k_steps` defaulting to a fraction of `--steps`,
   or an error/warning when `k_steps > steps`), and record the
   effective SNR reached (step 0 and last step) in `metadata.json`.
   Keep the historical behaviour reproducible behind an explicit flag.
3. A guard test asserting that the schedule at the last lesion step
   equals the nominal floor (within tolerance) for the grid defaults.
4. New amendments (v0.7, v0.8, v0.9 drafts on
   `docs/osf-prereg-2026-10`) specify a schedule that reaches each
   floor, flagged as a change versus the historical grids.
5. A dedicated PI pass to decide whether any historical claim needs an
   erratum, or whether a re-run (new pre-registration, not a
   replacement of existing results) is wanted. Out of scope here since
   it requires experiments.
6. Editorial PR qualifying "across 3 SNRs" wording in `main.tex`,
   README and milestone documents, once the PI approves.

## References

ADR-0003 to ADR-0019; spec
`docs/superpowers/specs/2026-04-20-bouba-sens-design.md` §4.1;
`scripts/run_grid.sh`; `src/bouba_sens/{lesion,cli,loop}.py`.
