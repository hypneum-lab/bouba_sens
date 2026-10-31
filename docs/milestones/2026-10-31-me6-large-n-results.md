# Me6 at large probe sizes — preregistered results (2026-10-31)

**Preregistration:** `docs/adr/0026-me6-large-n-prereg.md`, annotated tag
`prereg-me6-large-n-20261031` (2026-10-31 08:27:58 CET, squash merge
`b790db2`, `swh:1:rev:b790db272fede343fa55c996709fe39b31f4f9dd`), freeze
manifest `docs/prereg/me6-large-n/freeze-manifest.md`. Pilot and block
lengths: `docs/prereg/2026-10-30-me6-large-n-pilot.md` (`9b39c20`, pushed
before any confirmatory cell).
**Data:** `reports/me6_large_n_{cell,legacy}_{gaussian,mock}.json` (paired
nulls), `reports/me6_large_n_{cell,legacy}_{gaussian,mock}_power_empirical.json`,
`reports/me6_large_n_power_iid_p025.json` and
`reports/me6_large_n_power_iid_{cell,legacy}_{gaussian,mock}.json`. The ADR
named one `…_power.json` per arm; the power outputs are split per world
and per reference instead, with the same content.

## Timeline (electron-server, nice 19, PAR = 4, OMP 2)

| event | time (CET) |
|---|---|
| tag pushed, Software Heritage save requested (07:28 UTC) | 08:27:58 |
| mock pilot, seeds 100–101, `cell` then `legacy` arm | 08:31:35 – 09:14:45 |
| block lengths pushed (`9b39c20`) | ≈ 09:19 |
| `cell` Gaussian, `cell` mock, `legacy` Gaussian, `legacy` mock (seeds 400–404, 150 cells each, 0 failure) | 09:19:53 – 12:37:13 |
| paired nulls, empirical power, iid power | 12:42 – 13:41 |

Environment: commit `b790db2`, `uv.lock`
`551f04cf…`, nerve-wml v1.8.1, Python 3.14.4, numpy 2.4.4,
torch 2.11.0+cu130. Competing load on the host: ≈ 2 at the start,
≈ 10 around 12:00 (another job beside the grid's 8 threads).

## Regime check

The argmax-bridge network does not learn (ADR-0024): grand-mean v1
accuracy 0.238–0.269 over the 20 rows, against 1/4. These grids test
the statistic, its null and the frozen rule; **no row is a B-3 reading**
(ADR-0022 precondition, restated in ADR-0026).

## Preregistered readings (16 primary rows)

Unadjusted p of the paired sign-flip null (B = 20 000, seed 25102026);
Bonferroni level 0.000625; smallest attainable p ≈ 5·10⁻⁵. Blocks of the
pilot record: Gaussian 1; mock `cell` 8/16/16/16, mock `legacy` 4/8/8/8.

| arm | world | n | observed median | null median / P99 | p | reading |
|---|---|---|---|---|---|---|
| cell | Gaussian | 64 | 0.1563 | 0.1250 / 0.1563 | 0.0124 | non concluant |
| cell | Gaussian | 256 | 0.0938 | 0.0645 / 0.0762 | 5·10⁻⁵ | excès sur le nul apparié |
| cell | Gaussian | 1024 | 0.0693 | 0.0327 / 0.0386 | 5·10⁻⁵ | excès sur le nul apparié |
| cell | Gaussian | 4096 | 0.0585 | 0.0164 / 0.0193 | 5·10⁻⁵ | excès sur le nul apparié |
| cell | mock | 64 | 0.0625 | 0.1406 / 0.1875 | 0.9999 | compatible avec le nul apparié |
| cell | mock | 256 | 0.0527 | 0.0859 / 0.1191 | 0.9960 | compatible avec le nul apparié |
| cell | mock | 1024 | 0.0249 | 0.0439 / 0.0615 | 0.9980 | compatible avec le nul apparié |
| cell | mock | 4096 | 0.0088 | 0.0223 / 0.0313 | 1.0000 | compatible avec le nul apparié |
| legacy | Gaussian | 64 | 0.1328 | 0.1250 / 0.1484 | 0.2367 | compatible avec le nul apparié |
| legacy | Gaussian | 256 | 0.0781 | 0.0625 / 0.0742 | 0.0037 | excès sur le nul apparié |
| legacy | Gaussian | 1024 | 0.0488 | 0.0313 / 0.0371 | 5·10⁻⁵ | excès sur le nul apparié |
| legacy | Gaussian | 4096 | 0.0282 | 0.0157 / 0.0187 | 5·10⁻⁵ | excès sur le nul apparié |
| legacy | mock | 64 | 0.2109 | 0.1641 / 0.2031 | 0.0018 | excès sur le nul apparié |
| legacy | mock | 256 | 0.0664 | 0.1016 / 0.1270 | 1.0000 | compatible avec le nul apparié |
| legacy | mock | 1024 | 0.0435 | 0.0503 / 0.0630 | 0.9263 | compatible avec le nul apparié |
| legacy | mock | 4096 | 0.0263 | 0.0248 / 0.0310 | 0.2891 | compatible avec le nul apparié |

Historical file (v1 probe of ADR-0023, n = 64), same grids: Gaussian
compatible in both arms (p = 0.128 `cell`, 0.290 `legacy`); mock in
excess in both arms (p = 0.0047 `cell`, 5·10⁻⁵ `legacy`).

## Against the expected report

- **Not met:** the ADR expected the Gaussian rows to be compatible with
  the paired null at every n. Six of the eight Gaussian rows are in
  excess (n ≥ 256, both arms); five of them stay below the Bonferroni
  level.
- **Met:** the paired null stays close to the ADR-0021 iid reference
  (null median at n = 4096: 0.0164 / 0.0157 against the iid 0.0176).
- **Frozen 0.02 rule, rate under the paired null:** 1.0 at n ≤ 1024 in
  every row; at n = 4096, 0.0028 (`cell`) and 0.0009 (`legacy`) on the
  Gaussian, as expected, but 0.748 (`cell`) and 0.981 (`legacy`) on the
  mock. The rule fires on the observed median in 19 of the 20 rows (not
  on `cell` mock n = 4096). It decides nothing.
- No prediction was made for the mock rows nor for the difference
  between arms; both are described above, not tested.

## Power

- **iid reference** (p = 0.25, 30 trios, α = 0.01): power of the paired
  test against δ = 0.02 / 0.05 / 0.10 / 0.15 is 0.012 / 0.022 / 0.18 /
  0.84 at n = 64, 0.019 / 0.20 / 1 / 1 at n = 256, 0.083 / 1 / 1 / 1 at
  n = 1024, and 0.92 / 1 / 1 / 1 at n = 4096. The curves at each row's
  observed accuracy (0.249–0.251) are in the per-arm files.
- **Empirical** (50 plantings, audio | vision and gravity | force): on
  the rows already in excess unplanted, power is 1 trivially and says
  nothing about the planted δ. On the compatible mock rows, δ = 0.10 is
  detected from n = 1024 and δ = 0.05 at n = 4096 in both arms; δ = 0.02
  is never detected. The planting ignores the item order (declared in
  the ADR).

## Status

Preregistered readings as above. Why the Gaussian rows, where the v1
matrix was assumed to have no structure, exceed the paired null at large
n is open. It is either a dependence that the item-level sign flip does
not cover, or a real asymmetry between masking a and masking b in a
network that does not learn. An exploratory diagnostic is written as an
amendment before it is run; it changes none of the readings above.
