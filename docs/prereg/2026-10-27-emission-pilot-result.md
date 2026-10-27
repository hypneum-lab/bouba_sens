# Pilot result — straight-through emission (2026-10-27)

Criterion fixed in `docs/prereg/2026-10-26-emission-pilot-criterion.md`
(commit `b89b9f4`, pushed 2026-10-27 01:32 UTC before the pilot). Pilot run on
electron-server at commit `b89b9f4`, Phase 1 only, seeds 110–111, nice 19,
73 s. Files: `reports/emission_pilot_{gaussian,studyforrest}.json`.

| world | Phase-1 loss, last 20 steps (seed 110 / 111 / mean) | < 1.30 | vision margin, mean | ≥ 0.10 | learns |
|---|---|---|---:|---|---|
| Gaussian | 1.003 / 0.983 / **0.993** | yes | 0.439 | yes | **yes** |
| mock | 1.348 / 1.253 / **1.3006** | **no** | 0.295 | yes | **no** |

The mock world misses the loss threshold by 0.0006. The rule is applied
as written, with no rerun and no change of threshold. **The lesion ×
query interaction test runs on the Gaussian world only.** The mock world
is « non concluant (pas de réseau qui apprend) ».

Active argmax codes on the intact networks are still low (1–8 per
modality); the network learns with few codes.
