# Pilot criterion — straight-through emission "learns" (fixed before the pilot)

**Date:** 2026-10-26. Committed and pushed **before** the pilot is run.
**Decision:** PI, 2026-10-26 22:28, point 3.
**Option tested:** `--emission straight-through` (default `argmax-bridge`
unchanged).

## Pilot

- Seeds **110, 111**. They are distinct from 300–309 (lesion-interaction
  confirmatory set), from 100–101 and 200–204 (B-3 v2) and from 0–1
  (collapse diagnosis).
- Worlds: `gaussian` and `studyforrest` without a cache (mock).
- Phase 1 only, run as `scripts/run_grid_probe_v2.sh` runs it:
  `bouba-sens train --steps 200 --batch-size 16 --transducer-gating hard
  --gumbel-tau 1.0 --probe-v2 --probe-v2-seed seed·10000+5099 --emission
  straight-through`. Host electron-server, nice 19.
- Check: `scripts/emission_pilot_check.py`.

## Criterion (per world)

The network **learns** on a world if both hold:

1. **Loss:** the mean over the two pilot seeds of the mean Phase-1
   training loss over the last 20 steps is **< 1.30** (ln 4 = 1.386 is
   chance; the default emission gives 1.385 and 1.390).
2. **Positive control:** the mean over the two seeds of the v2 probe's
   vision margin over the majority baseline is **≥ 0.10**.

Both thresholds were set from information already seen in the collapse
diagnosis (seeds 0–1, straight-through: 0.963 Gaussian, 1.276 mock). The
1.30 loss threshold was already written in the draft preregistration #43
before this pilot.

## Consequence

The lesion × query interaction test (`docs/prereg/2026-10-26-b3-lesion-interaction.md`)
is run with `--emission straight-through` **only on the worlds that
pass**. A world that fails is reported as « non concluant (pas de
réseau qui apprend) ». If neither world passes, nothing is run.
