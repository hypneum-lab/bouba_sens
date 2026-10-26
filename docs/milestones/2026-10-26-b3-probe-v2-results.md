# B-3 v2 — preregistered results (2026-10-26)

Preregistration: `docs/prereg/2026-10-25-b3-probe-v2.md`, tag
`prereg-b3-probe-v2-20261025` (2026-10-25 19:14:09 +01:00). Design:
ADR-0023 (Proposed). Pilot: `docs/prereg/2026-10-25-b3-probe-v2-pilot.md`.

## Timeline

| event | time (+01:00) |
|---|---|
| tag pushed (commit `1d96a65`) | 2026-10-25 19:14:09 |
| pilot grids, seeds 100–101 | 19:14 – 20:32 |
| pilot record committed (`fd15deb`): block 1 Gaussian, 8 mock | 20:32:53 |
| confirmatory Gaussian, seeds 200–204 | 20:32:58 – 21:26:35 |
| confirmatory mock | 21:26:35 – 22:02:10 |
| analysis (`scripts/b3_probe_v2.py report`) | 2026-10-26 |

Host electron-server (8-core Xeon X5680, CPU torch 2.11), nice 19, four
cells at a time, competing load 9–17. Cost: pilot 77 min, confirmatory
89 min (300 cells, 0 failure); paired null 10 s / 5 s.

## Verdict per criterion

| criterion | Gaussian | mock |
|---|---|---|
| positive control: mean vision margin ≥ 0.10 | 0.485 — **reached** | 0.282 — **reached** |
| complete grid (30 trios) | yes | yes |
| block (pilot rule) | 1 | 8 |
| B-3 v2 median (observed) | 0.4141 | 0.3320 |
| paired null mean / P99 | 0.0794 / 0.0918 | 0.1363 / 0.1602 |
| p (B = 20 000) | 5·10⁻⁵ (0 exceedances) | 5·10⁻⁵ (0 exceedances) |
| **preregistered verdict** | **asymmetry present** | **asymmetry present** |

## Reading (exploratory, see ADR-0023 §Results)

The rejection is carried entirely by the query-modality main effect:
query accuracy differs by modality (Gaussian audio 0.69 / vision 0.68 /
tactile 0.57 / gravity 0.29 / force 0.33; mock live 0.55 vs zeroed
0.23), lesioned-modality means are flat, and the asymmetry left after
removing the additive query effect is at its null (Gaussian p ≈ 0.55).
B-3 v2 thus detects the unequal informativeness of the modalities built
into the worlds, not a lesion-induced asymmetry. The code channel is
collapsed (one code per modality on every intact network), so the probe
reads the encoders' hidden state, which an untrained network already
makes partly decodable.

## Not done

Real-ECG world (no data cache). XOR / Sinusoid. A preregistered test of
a lesion-specific effect (H0 conditional on the query main effect).
