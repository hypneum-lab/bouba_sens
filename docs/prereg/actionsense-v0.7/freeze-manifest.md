# ActionSense v0.7 — pre-registration freeze manifest

**Study:** bouba_sens v0.7, ActionSense world (amendment
`docs/osf/amendment-v0.7-actionsense.md`).
**PI sign-off:** 2026-10-30, comment 6532469131 on PR #55 (amendment)
and comment 6532468639 on PR #59 (ADR-0027).
**Tag:** `bouba_sens/v0.7-actionsense-prereg` (annotated, on the
squash merge of PR #55). No ActionSense grid has run before this tag.
**Extraction code:** `a1fd6ff` (PR #57; includes PR #56).

## Protocol and code files (sha256)

| File | sha256 |
|---|---|
| `docs/osf/amendment-v0.7-actionsense.md` | `f20be89ebe280331f091d7d0e05454740aad318ad59fb441dbb56600d18f89a5` |
| `docs/superpowers/specs/2026-10-28-actionsense-world-design.md` | `932701cd688eada2eb9fc8a5889f807f484d4bc9c390af61eb8789c8214ef602` |
| `docs/superpowers/plans/2026-10-28-actionsense-world.md` | `1b3b78bebed31ee09186b34ab0f9a6d79ee8df626e6b011de8dead35b2f3b23a` |
| `docs/adr/0021-me6-null-calibration.md` | `f1e03d9e12bd7b273612962b15ba44aa00aee717e7de829c9b468c2b7daf40de` |
| `docs/adr/0022-me6-paired-null.md` | `0411635960ef29861fe29a6ba324268e5b8399e7deabb5783a4a3a10733924d9` |
| `docs/adr/0023-b3-probe-v2.md` | `8b9c6468c0a55a688ca2b4238d6db8bcab50e8c813796ef5cdfa22cf738d4d64` |
| `docs/adr/0027-erratum-world-seed-train-lesion-eval.md` | `243ad1d3248907a3b5aedbdf47f9172e9d1fc96de500b6943e0b5f9ac5cdd309` |
| `src/bouba_sens/world/actionsense.py` | `c53b3a63c8d2fc595bc8a7597433731d37267cb1dba7fd4a3d0dd6f4885ab69b` |
| `src/bouba_sens/world/actionsense_features.py` | `6d950c2491b349da35114788f03d51be252ba2c0798abb46783c04ac13512492` |
| `src/bouba_sens/cli.py` | `eb920106e6536dfdd53515e1f0389008cd8101191524a46ddc851c81a96c77cc` |
| `src/bouba_sens/loop.py` | `47eea148f4cfc41150a3570ed45600dab32ba88dc8b9411ad6fcf5135d0278a5` |
| `scripts/fetch_actionsense.py` | `3c196c4b4f0a63df936e6b7261fc0a548b1107eb0967019c123d1343e80295c6` |
| `scripts/extract_actionsense.py` | `6751283bf9cc14b31eb2ba146d0dc1ae4de8c329a24017d6176a2fef8f2485ad` |
| `scripts/run_grid.sh` | `d3c1a497e3fb622e7705c6dc3bbaad910b36ece593997009e80985c94118bad1` |
| `scripts/aggregate_grid.py` | `a773153ba9f60fdb2bf343aa5ce5b115a3ce964e58cddf8e2adb1659df3929b5` |
| `scripts/me6_paired_null.py` | `47160d6aa1bfcf674ee5272aabd01bcff07f9aa7a91e8e20361fa437fe57378f` |
| `uv.lock` | `551f04cff304e2df0aa4738d501fef60df40b5389fbc45a56abcaf3bc7768d29` |

## Code trees (git tree ids at the tagged commit)

| Tree | git tree id |
|---|---|
| `src/` | `7961b185125e5148929d066a641cb326d8485e57` |
| `scripts/` | `619c5c8366c5c13850d5f719e6695e717df2de61` |
| `tests/` | `d1b5ed48ffeca8d7e67be65c30054b4379a8200e` |

## Data

| Item | sha256 |
|---|---|
| Raw manifest `actionsense_raw/MANIFEST.json` (35 files, S00–S05, size and sha256 each; download verified) | `b801042416796af2431c7271bcae7fcb2a3521428718467392c327da22f82e95` |
| S00 extraction at `a1fd6ff`, output `MANIFEST.md` (13 062 rows, macM1, ffmpeg 9.0.2) | `87ac0f7be7150814a26c756317aaea80fab3b24e5ef3558276757f033d1789c3` |
| S00 `audio.pt` | `cf6f95771aa380426e6aebd607ff669a388d5466f86163005b3eaf57036f7bd5` |
| S00 `force.pt` | `7a8ec62dffdbfd14d2906b027172a3879fc517c8f9e88695731d16ec504beb26` |
| S00 `gravity.pt` | `130186a39042b4c9ad820a4af28090f1d411619a3086bf732f4287428c1021bb` |
| S00 `labels.pt` | `461ac91875b1efeb278880a47d5ed2fb8489165f823e7a854e293bac319ffd14` |
| S00 `rows.pt` | `0b5c4f6a7821c055216469317be1b610890229d3cf844b9f100aeade5a760ed1` |
| S00 `subject.pt` | `63bf868d24d0077f934e73e8f7996df58cd3f3a31a6ac13d5aa7efb61582030e` |
| S00 `tactile.pt` | `b806fa36e020fe1eeca63ab61c29eb29ffe0184056b2c738e84e9af7b862a646` |
| S00 `vision.pt` | `c73f9a4382f07488145504aa6c4bd37d3a225c4b259885315e44729b5d56aba1` |

The sha256 of this manifest is written in the tag message.
