# Me6 large-n — mock pilot and block lengths (ADR-0026)

**Status:** pilot record, committed and pushed before any confirmatory
cell. No Me6 statistic or paired null was computed on the pilot.
**Freeze:** tag `prereg-me6-large-n-20261031` on `b790db2`
(`swh:1:rev:b790db272fede343fa55c996709fe39b31f4f9dd`). The file name
keeps the date planned in ADR-0026; the pilot ran on 2026-10-31.

## Run

- Host: electron-server (8-core Xeon X5680, CPU torch), free at the
  tag (load 1.45, no other bouba_sens or NH-01 job). Run log line
  `nproc=2` is `nproc` reading `OMP_NUM_THREADS=2`; the host has
  8 cores.
- Clone at the tag, commit `b790db272fede343fa55c996709fe39b31f4f9dd`,
  `uv.lock` sha256
  `551f04cff304e2df0aa4738d501fef60df40b5389fbc45a56abcaf3bc7768d29`,
  nerve-wml v1.8.1, Python 3.14.4, numpy 2.4.4, torch 2.11.0+cu130.
- `scripts/run_grid_probe_v2.sh`, world `studyforrest` without data
  cache, `BOUBA_SENS_STUDYFORREST_MOCK_SAMPLES=65536`,
  `PROBE_SIZES=64,256,1024,4096`, argmax-bridge, seeds 100–101,
  `PAR=4`, `OMP_NUM_THREADS=2`, nice 19.
- `cell` arm 2026-10-31 08:31:35 → 08:53:26 CET, then `legacy` arm
  08:53:27 → 09:14:45; 60 cells + 2 phase 1 per arm, all complete
  (60 `eval_report.json`, 60 `per_query_successes_n4096.json` each).

## Block lengths (`scripts/me6_large_n.py blocks`, 12 trios per arm)

| Arm | n = 64 | 256 | 1024 | 4096 | historical file |
|---|---|---|---|---|---|
| `cell` (k*) | 6 | 12 | 13 | 12 | 7 |
| `cell` (block) | **8** | **16** | **16** | **16** | **8** |
| `legacy` (k*) | 4 | 8 | 7 | 7 | 6 |
| `legacy` (block) | **4** | **8** | **8** | **8** | **8** |

Gaussian: block 1 at every n in both arms, by design (no pilot).

These blocks are fixed for the confirmatory `report` and empirical
`power` runs of each arm:

- `cell` mock: `--blocks 64:8,256:16,1024:16,4096:16,legacy:8`
- `legacy` mock: `--blocks 64:4,256:8,1024:8,4096:8,legacy:8`
- both arms, Gaussian: `--blocks 64:1,256:1,1024:1,4096:1,legacy:1`

## Files

| File | sha256 |
|---|---|
| `reports/me6_large_n_pilot_mock_cell.json` | `08cd8ad406b566d1f0b82f659420517545c4c89f2bbc6995c21e0c27b4808a4a` |
| `reports/me6_large_n_pilot_mock_legacy.json` | `b21c853858bd207c2a0fcf0929aa4aab03751b527ec06e945801e8c84e7475a4` |
