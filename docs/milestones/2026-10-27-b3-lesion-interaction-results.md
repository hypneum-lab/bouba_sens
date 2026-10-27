# B-3 lesion × query interaction — preregistered results (2026-10-27)

**Preregistration:** `docs/prereg/2026-10-26-b3-lesion-interaction.md`,
annotated tag `prereg-b3-lesion-interaction-20261027` (2026-10-27
03:48:14 UTC = 04:48 CET, commit `37810b5`).
**Data:** `reports/b3_lesion_interaction_gaussian.json`.

## Timeline

| event | time |
|---|---|
| ADR-0024 erratum (PR #44) and the `--emission straight-through` option | 2026-10-27 |
| pilot criterion committed (`b89b9f4`) | 01:32 UTC |
| pilot, seeds 110–111 (73 s); result committed (`5d447f7`) | ≈ 01:35 UTC |
| prereg finalised and tagged | 03:48 UTC |
| confirmatory Gaussian grid, seeds 300–309, 300 cells, 0 failure | 06:05–08:25 CET (2 h 20, competing load 10–13) |

## Regime check (the network learns)

- Phase-1 loss over the last 20 steps on the 10 confirmatory seeds:
  0.81–1.13, against ln 4 = 1.386.
- Mean Me1: **0.467**, against 0.257 for the default emission on the
  ADR-0023 grids.

## Verdict per criterion

| criterion | Gaussian | mock |
|---|---|---|
| precondition: pilot learns | yes (loss 0.993, margin 0.439) | **no** (loss 1.3006 ≥ 1.30) |
| positive control (vision margin ≥ 0.10) | 0.495, reached | — |
| complete grid (60 trios) | yes | — |
| statistic S (observed) | 3.90·10⁻⁴ | — |
| exact seed-level null (1 024 patterns): mean / P99 | 4.10·10⁻⁴ / 9.57·10⁻⁴ | — |
| p | **0.494** | — |
| **preregistered verdict** | **compatible avec l'absence d'interaction** | **non concluant (pas de réseau qui apprend)** |

## Exploratory (not preregistered)

- Mean P by query, off-diagonal cells: audio 0.661, vision 0.676,
  tactile 0.612, gravity 0.281, force 0.333. The query main effect of
  ADR-0023 is still there in a network that learns. The largest seed-mean
  interaction residuals are below 0.01.
- The secondary block permutation (block 1) rejects the stronger H0 of
  no lesion effect on off-diagonal cells: T = 0.0341 vs null mean
  0.0080, p = 1·10⁻⁴. Within a query row, entries differ by about
  0.01–0.03 across lesioned modalities.
- This item-level permutation treats every lesion cell as the same
  model. In fact each cell is a separate Phase-2 run with its own lesion
  seed, so run-to-run variability is enough to produce this rejection.
  That is why the seed-level test is the primary one. The secondary
  result is not read as a lesion effect.
