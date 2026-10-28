# Pilot record — B-3 v2 block length (2026-10-25)

Recorded **before** the confirmatory grids (seeds 200–204) start, as
required by `docs/prereg/2026-10-25-b3-probe-v2.md` §5.1.

- Code: tag `prereg-b3-probe-v2-20261025` (commit `1d96a65`), cloned on
  electron-server, `scripts/run_grid_probe_v2.sh`, four cells at a time,
  nice 19.
- Pilot seeds: 100, 101; 60 cells per world, 12 complete trios each,
  0 failure.
- Gaussian: 2026-10-25 19:14–20:00 (46 min, competing load ≈ 14–17);
  mock (`studyforrest`, no cache): 20:00–20:32 (31 min).
- Files: `reports/b3_probe_v2_pilot_{gaussian,mock}.json`.
- No B-3 v2 statistic, null or p-value was computed on the pilot.

## Block length (preregistered rule)

| world | ACF(d) lag 1 / 2 / 4 / 6 / 8 | k* | block |
|---|---|---:|---:|
| Gaussian | −0.001 / −0.006 / −0.010 / −0.012 / −0.005 | — | **1** (iid by design) |
| mock | 0.255 / 0.170 / 0.129 / 0.087 / 0.031 | 6 | **8** |

ACF(d) = lag-k autocorrelation of the per-item paired differences of the
v2 probe, mean over series with non-zero variance. Mock ACF of the
per-item outcomes: 0.420 (lag 1), 0.130 (6), 0.058 (8).

**Fixed for the confirmatory reading: block 1 (Gaussian), block 8
(mock).**

## Positive control (descriptive on the pilot; criterion applies to
## the confirmatory seeds)

Mean margin of the vision probe over the majority baseline on the intact
networks: 0.463 (Gaussian), 0.373 (mock); threshold 0.10. Untrained
reference (same probe, initial weights): 0.348 (Gaussian), 0.377 (mock)
— the information is largely already present in the random encoders, as
ADR-0023 anticipated.
