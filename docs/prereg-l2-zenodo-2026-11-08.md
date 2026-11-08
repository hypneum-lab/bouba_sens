# bouba_sens — Zenodo L2 record for frozen pre-registrations (2026-11-08)

This file registers the restricted Zenodo deposit that provides level L2
(verifiable deposit date) for the four frozen pre-registrations of this
repository. The record at <https://zenodo.org/records/24054109> (DOI
[10.5281/zenodo.24054109](https://doi.org/10.5281/zenodo.24054109),
concept DOI
[10.5281/zenodo.24054108](https://doi.org/10.5281/zenodo.24054108)) is
restricted: metadata are public, files are restricted, and authenticated
Zenodo users may request access from the PI. The record is published
2026-11-08 under CC-BY-4.0 as a voluntary PI act to fix a verifiable
deposit date for pre-registrations that were previously L1 only. It
belongs to the Hypneum Lab lot of 2026-11-08, which covers all five
Hypneum repositories in one batch.

The L1/L2 terminology follows
`hypneum-lab/.github` `docs/PROJECT-MANAGEMENT.md` section 4: L1 is a
local annotated git tag archived by Software Heritage; L2 adds a dated
deposit with a persistent public identifier (here: a restricted Zenodo
DOI). See also `hypneum-lab/.github#5`.

## Frozen pre-registrations in this deposit

| Study | Tag | Commit | SWHID | Zip | SHA-256 | Documents in the zip |
|---|---|---|---|---|---|---|
| B-3 probe v2 | `prereg-b3-probe-v2-20261025` | `1d96a656fdef6b0a2671832b8709c3b2e82be556` | `swh:1:rev:1d96a656fdef6b0a2671832b8709c3b2e82be556` | `prereg-b3-probe-v2-20261025.zip` | `179d2cc07cda4cba15fa8d8cbdc9e9477fc1516d8332fa273137c86b579995a5` | `docs/adr/0023-b3-probe-v2.md`, `docs/prereg/2026-10-25-b3-probe-v2.md` |
| B-3 lesion × query interaction | `prereg-b3-lesion-interaction-20261027` | `37810b5a25f643248f7759442e6ad2a56febe0be` | `swh:1:rev:37810b5a25f643248f7759442e6ad2a56febe0be` | `prereg-b3-lesion-interaction-20261027.zip` | `aa7b69ce7e098b603c6e9acb6fe08b89af44ca33dc6fee46c4673455c8c10d39` | `docs/prereg/2026-10-26-b3-lesion-interaction.md`, `docs/prereg/2026-10-26-emission-pilot-criterion.md` |
| Me6 large-n paired nulls | `prereg-me6-large-n-20261031` | `b790db272fede343fa55c996709fe39b31f4f9dd` | `swh:1:rev:b790db272fede343fa55c996709fe39b31f4f9dd` | `prereg-me6-large-n-20261031.zip` | `65f75c298e61f069f3414846d8d2f77c0c99ba9172b8d98663a13eb29d09afe5` | `docs/adr/0026-me6-large-n-prereg.md`, `docs/prereg/me6-large-n/freeze-manifest.md` |
| v0.7 ActionSense | `bouba_sens/v0.7-actionsense-prereg` | `02c24073d3a04e1db3d3adf2ee77e545c0bdde17` | `swh:1:rev:02c24073d3a04e1db3d3adf2ee77e545c0bdde17` | `bouba_sens_v0.7-actionsense-prereg.zip` | `1ad90e105533ca2010604574075636c5eeadee3ddc5deaed1f422fb952c764b0` | `docs/adr/0027-erratum-world-seed-train-lesion-eval.md`, `docs/osf/amendment-v0.7-actionsense.md`, `docs/prereg/actionsense-v0.7/freeze-manifest.md` |

## Files not in the deposit (created after the tag)

The following files exist on `main` but were committed after the freeze
tag; they are not part of the deposit and are not covered by the L2 date.

| Tag | File |
|---|---|
| `prereg-b3-probe-v2-20261025` | `docs/prereg/2026-10-25-b3-probe-v2-pilot.md` |
| `prereg-me6-large-n-20261031` | `docs/prereg/2026-10-30-me6-large-n-pilot.md` |
| `prereg-me6-large-n-20261031` | `docs/prereg/me6-large-n/amendment-2026-10-31-diagnostic.md` |
| `bouba_sens/v0.7-actionsense-prereg` | `docs/osf/erratum-2026-10-31-v0.7-seeds.md` |
