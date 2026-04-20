# bouba_sens — Sprint 6 Implementation Plan (v0.3: cross-world replication)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal.** Triangulate the directional inversion of B-1 observed on GaussianWorld in ADR-0004 by replicating the 150-cell grid on XOR-world and Sinusoid-world. Produce a single aggregate artifact per world and a cross-world ADR-0005 comparing the three verdict triples.

**Hypothesis to test.** If B-1 inversion (T2 beats T1) holds across all three worlds -> the cross-modal architecture is structurally incompatible with the congenital-blindness hypothesis as formulated. If B-1 flips back to >0.05 on XOR or Sinusoid -> the inversion is GaussianWorld-specific, and the architecture is salvaged.

**Architectural insight.** Both `XORWorld` and `SinusoidWorld` already exist under `bouba_sens.world.*` (landed in Sprint 1) with the same `WorldSimulator.sample(batch_size, seed)` API as `GaussianWorld`. The only CLI gap is that `train` and `lesion` hardcode `GaussianWorld`; fixing this is ~10 lines.

**Tech Stack:** unchanged.

**Parent spec + ADRs:** ADR-0004 Next Steps #2 calls for exactly this replication pass. Spec sections 4.5 (replication) and 1.2 (invariants) unchanged.

**Sprint 6 scope:** Tasks 6.1 -> 6.4. Paper draft is deferred to Sprint 7 to consume the full cross-world evidence.

**Compute target:** GrosMac for 6.1 and 6.3 (code + aggregation). Studio for 6.2 (the two replication grids).

---

## File structure touched in Sprint 6

```
bouba_sens/
|-- src/bouba_sens/
|   |-- cli.py                          [Task 6.1]  --world flag on train + lesion
|   `-- _version.py                     [Task 6.4]  bump to 0.3.0
|-- scripts/
|   `-- run_grid.sh                     [Task 6.1]  WORLD env var
|-- docs/adr/
|   `-- 0005-v03-cross-world-verdicts.md  [Task 6.3]
`-- CHANGELOG.md                        [Task 6.4]  v0.3.0 entry
```

---

## Tasks

### Task 6.1 - Parametrise the world in train + lesion CLI + run_grid.sh

- [ ] `src/bouba_sens/cli.py::train`: add `--world` option (default `gaussian`) and dispatch to `XORWorld` or `SinusoidWorld` by name, mirroring the existing `sim` command.
- [ ] `src/bouba_sens/cli.py::lesion`: same. The hardcoded `GaussianWorld(seed=0)` becomes world-dispatched.
- [ ] `scripts/run_grid.sh`: add `WORLD="${WORLD:-gaussian}"` env; thread it into both CLI invocations. Default preserves v0.2 behaviour.
- [ ] Unit test covers all three world choices through a smoke grid call (`--world xor` + `--world sinusoid`).

### Task 6.2 - Studio replication runs (XOR + Sinusoid)

- [ ] `ssh studio "cd ~/Projets/bouba_sens && git pull && uv sync --all-extras"`.
- [ ] Launch two background grids (serial or parallel depending on disk pressure):
  - `nohup WORLD=xor OUT_ROOT=runs/v03_xor_grid STEPS_TRAIN=200 STEPS_LESION=100 METRICS='Me1,Me2,Me3' bash scripts/run_grid.sh > logs/grid-v03-xor-...log 2>&1 &`
  - Same for `WORLD=sinusoid` with `OUT_ROOT=runs/v03_sinusoid_grid`.
- [ ] Expected wall time: ~17 min per world, ~35 min serial or ~17 min parallel on M3 Ultra.
- [ ] Aggregate each: `reports/v0.3_xor_aggregate.json` + `reports/v0.3_sinusoid_aggregate.json`.
- [ ] Scp both back to GrosMac.

### Task 6.3 - ADR-0005 cross-world verdicts

- [ ] `docs/adr/0005-v03-cross-world-verdicts.md`: 3x3 verdict table (world x invariant) with medians, cells_counted, passes. Add an "inversion tracker" column for B-1 showing whether each world retains the v0.1 direction or the v0.2 inversion.
- [ ] Call out whether B-3 (PASS on Gaussian) survives cross-world. If yes, it is a strong architectural invariant; if no, it was Gaussian-specific.
- [ ] Record honest decision on whether to mutate the paper v0.1 headline or stand by the Gaussian-only verdict.
- [ ] No threshold changes.

### Task 6.4 - Sprint 6 close + v0.3.0 release

- [ ] Bump `_version.py` + `pyproject.toml` to `0.3.0`.
- [ ] `CHANGELOG.md` v0.3.0 entry with the cross-world triplet.
- [ ] Version-pinned smoke tests updated.
- [ ] `git tag -a v0.3.0 -m "v0.3.0 Sprint 6 close - cross-world replication"`.
- [ ] Push main + tag.
- [ ] Update memory.

---

## Risks + mitigations

| Risk | Mitigation |
|------|------------|
| R-sprint6-1: XOR / Sinusoid have structurally different MI floors | Me3_delta is world-comparative within itself; no need to re-calibrate threshold. |
| R-sprint6-2: parallel runs OOM on Studio | Run serially if memory climbs over 50%; M3 Ultra has 512 GB. |
| R-sprint6-3: Task 6.1 regression on GaussianWorld baseline | Default `WORLD=gaussian` preserves v0.2 behaviour byte-for-byte; one regression test confirms. |
| R-sprint6-4: XOR world has fewer lesion-recoverable patterns | If all three worlds fail B-2, revisit Me3 Kraskov calibration; otherwise move on. |

---

## Exit criteria

1. 156 existing tests + new world-dispatch tests green.
2. Two Studio grids complete, two aggregate JSONs downloaded.
3. ADR-0005 committed with cross-world verdict triple.
4. Tag `v0.3.0` pushed.
