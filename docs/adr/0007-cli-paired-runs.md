# ADR-0007 — CLI-level paired-run wiring (`eval --t1-ckpt/--t2-ckpt`)

**Status:** Accepted
**Date:** 2026-04-20
**Accepted:** 2026-04-20
**Supersedes:** —
**Superseded by:** —
**Related:** ADR-0003 (proposal), ADR-0004 (aggregator pivot), ADR-0006 (critical-validation, Sprint 7)

## Context

Sprint 5 chose aggregator-level pairing (ADR-0004) to compute Me7 /
congenital gap : the aggregator scans `runs/**/metrics.json`,
groups cells by `(seed, modality, snr)` and subtracts T1 from T2.
This heuristic tuple-matching is sufficient for the pre-registered
5x5x2x3 grids of Sprints 5-6 but becomes the wrong surface for
three concrete empirical needs that surfaced while drafting the
Sprint 7 / Paper v0.1 plan.

Those three motivations are :

1. **Targeted debug of the Sinusoid sign-flip** — ADR-0005 : 67-78
   logged Me7 = +0.0125 on SinusoidWorld vs -0.0063 on Gaussian /
   XOR. Reproducing a single suspect pair today requires re-
   launching the 150-cell grid, wasting ~17 min of wall time per
   iteration.
2. **Reproducibility for Paper v0.1** — explicit CLI-level pairing
   is citable in supplementary material as a reproducer recipe ;
   heuristic tuple-matching inside the aggregator is not.
3. **Standalone comparisons outside the grid** — e.g. seed-0 T1 vs
   seed-1 T2, or a future cross-world pair (Gaussian T1 vs
   Sinusoid T2). The aggregator pins `(seed, modality, snr)` as
   the matching key and cannot express these.

## Me1 observable clarification (Phase B finding)

During Phase B implementation, two distinct Me1 observables were
found to coexist in the codebase under the same name. This ADR
formalises the distinction and keeps both alive additively.

- **`me1` (primary, frozen by ADR-0004 / ADR-0005).** Computed
  via `me1_accuracy(report)` — mean of the last 10 % of
  `report.accuracy_curve` during Phase 2 adaptation, all 5
  modalities active with the lesion applied. Source of truth
  for paper verdicts B-1 / B-2 / B-3. **Never modified by ADR-0007.**
- **`me1_probe` (reproducibility auxiliary, introduced v0.4.0).**
  Computed via `query_accuracy("audio", seed=seed+777)` on the
  model rebuilt from `model.pt` + `config.yaml` — frozen probe
  pass in post-adaptation state, only the audio modality signal-
  bearing. Distinct observable by design; validates model
  reload fidelity.

The two are semantically different (adaptation-curve tail vs.
probe pass) and generally numerically distinct (|Δ| ≈ 0.2 in
representative 3-step smoke runs). They coexist additively in
`eval_report.json` as of v0.4.0. The paired-run CLI consumes the
**probe observable** (`_compute_me1` internally); therefore the
equivalence invariant below is stated against `me1_probe`, not
against the legacy `me1`.

## Decision

- Add a CLI entry point (sub-command or flag on `eval`) of the
  form :
  ```
  eval --t1-ckpt PATH --t2-ckpt PATH --out PATH \
       [--modality M] [--snr S]
  ```
- **Checkpoint contract.** `PATH` désigne le répertoire d'une cell
  produit par `train` ou `lesion` à partir de l'ADR-0007 Phase A,
  contenant `model.pt` (`torch.save` des 4 state_dicts
  `{mux, nerve, head, sensory}`) + `config.yaml` (OmegaConf dump
  avec `phase`, `seed`, `world`, `steps`, modality/timing/snr_*
  pour lesion). Ad-hoc usage : un répertoire minimal avec ces 2
  fichiers est suffisant, aucune structure grid ancestor requise.
- Output schema (single JSON per invocation) :
  ```json
  {
    "pair": {
      "seed_t1": 0,
      "seed_t2": 1,
      "cell_id": "gaussian-s0-vision-snr10",
      "modality": "vision",
      "snr": 10.0
    },
    "me1_t1": 0.71,
    "me1_t2": 0.68,
    "me7": 0.03,
    "timestamp": "2026-04-20T21:00:00Z"
  }
  ```
  `cell_id` is nullable when the pair is ad-hoc (not derived from
  a grid cell).
- **Additive only.** No change to `scripts/aggregate_grid.py`, no
  change to the v0.3.0 `reports/*.json` artefacts. The CLI is a
  new, optional emission channel.
- **Coexistence with `--emit-raw-pairs`** (Sprint 7 Task 7.2) :
  the aggregator's `raw_me7_pairs` payload will be upgraded to the
  enriched `list[dict]` format mirroring the CLI schema above, so
  that `scripts/bootstrap_me7.py` can consume either a directory
  of CLI-emitted JSON files or the aggregator's `raw_me7_pairs`
  list without branching on source.

## Consequences

**Positive.**
- Targeted single-pair debug of Sinusoid / XOR anomalies
  (one-shot reruns instead of 150-cell grid reruns).
- Paper v0.1 supplementary gains a minimal reproducer snippet
  that does not require the full grid runner.
- Ad-hoc tooling (notebooks, Studio sessions) can build on a
  stable pairing surface without scraping `runs/`.

**Negative.**
- Widens the `eval` CLI surface ; adds one more code path that
  computes Me7. Mitigated by routing both paths through the same
  `me7_congenital_gap` implementation in `src/bouba_sens/metrics`.
- Possible divergence between `me7` emitted by the CLI and `me7`
  computed by the aggregator on the same triplet. Mitigated by
  the equivalence test below.

**Tests required.**
- (a) Unit test on `eval --t1-ckpt/--t2-ckpt` with fixture
  checkpoints, asserting the output schema.
- (b) Probe-observable equivalence (Phase B, Option 4). The
  CLI-emitted `me7` (paired-run) must agree within float64
  epsilon (|Δ| < 1e-12) with the `me7` derived by subtracting
  the `me1_probe` fields of the T1 / T2 `eval_report.json`. Both
  paths invoke `_compute_me1` → `query_accuracy`; divergence is
  bounded by reload determinism, not JSON precision. `_compute_me1`
  forces `torch.manual_seed` / `numpy.random.seed` / `random.seed`
  at call time to isolate from caller RNG context. **The legacy
  `me1` field (primary paper observable) is NOT subject to this
  equivalence** — it is a structurally different measurement
  (adaptation curve tail vs. probe pass).
- (c) Determinism test : two successive `_load_cell(dir)` calls
  return identical `(me1, seed, cell_name)` tuples within 1e-12.
  This is the invariant the paired-run CLI depends on for stable
  Me7 output; it subsumes Phase B's rebuild-fidelity claim.

## Pre-registration fidelity

No threshold changes (0.05 / 0.10 / 0.02 unchanged vs ADR-0003 /
ADR-0004). No metric implementation changes — `me7_congenital_gap`
stays the single source of truth. Pure tooling / emission surface ;
no p-hacking vector introduced (the CLI cannot cherry-pick seeds
in a way the aggregator cannot already ; both consume fully-
frozen checkpoint paths at call time).

**Scope of CLI-emitted pairs.** CLI-emitted pairs go into *sup-
plementary / exploratory* material only. Primary paper verdicts
(B-1, B-2, B-3) continue to cite **only** paires issues de
l'aggregator grid fidèle au 5x5x2x3 pré-enregistré. Cross-world
or cross-seed pairs (e.g. Gaussian T1 vs Sinusoid T2) enabled by
the CLI are flagged as ad-hoc and never substituted for the pre-
registered invariant tests.

Version bump recommended : **v0.4.0** (additive feature, not a
fix).

## Implementation note

ADR-0007 **must not block Sprint 7**. Sprint 7 (ADR-0006) owns
null-model B-3, bootstrap B-1 and estimator robustness Me3 and
its timeline is independent. Recommended order :

1. Sprint 7 Task 7.2 lands `--emit-raw-pairs` with the enriched
   `list[dict]` format (already aligned with this ADR).
2. Sprint 7 closes / tags v0.3.0.
3. ADR-0007 implementation opens as a separate feature branch
   cutting v0.4.0.

This sequencing keeps v0.3.0 scope limited to critical
validation and pushes CLI pairing to a clean follow-up release.
