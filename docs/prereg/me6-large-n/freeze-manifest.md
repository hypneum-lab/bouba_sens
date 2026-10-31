# Me6 large-n — pre-registration freeze manifest

**Study:** Me6 at large probe sizes, paired nulls, n ∈ {64, 256, 1024,
4096} (`docs/adr/0026-me6-large-n-prereg.md`).
**PI sign-off:** comment 6540527463 on PR #58 (on `ec101c5`),
posted 2026-10-31 07:18 UTC.
**Tag:** `prereg-me6-large-n-20261031` (annotated, on the squash
merge of PR #58). No Me6 large-n pilot or confirmatory cell has run
before this tag.
**Code base:** main `1295160` (PR #59, ADR-0027) plus PR #58.

## Protocol and code files (sha256)

| File | sha256 |
|---|---|
| `docs/adr/0026-me6-large-n-prereg.md` | `9b104990fd76ed4f4214ef89d05f6ce8db432f84d99e0df4680ac7201e0f31b4` |
| `docs/adr/0021-me6-null-calibration.md` | `f1e03d9e12bd7b273612962b15ba44aa00aee717e7de829c9b468c2b7daf40de` |
| `docs/adr/0022-me6-paired-null.md` | `0411635960ef29861fe29a6ba324268e5b8399e7deabb5783a4a3a10733924d9` |
| `docs/adr/0023-b3-probe-v2.md` | `8b9c6468c0a55a688ca2b4238d6db8bcab50e8c813796ef5cdfa22cf738d4d64` |
| `docs/adr/0027-erratum-world-seed-train-lesion-eval.md` | `243ad1d3248907a3b5aedbdf47f9172e9d1fc96de500b6943e0b5f9ac5cdd309` |
| `src/bouba_sens/cli.py` | `f4bb8cf23b8e6f61868c1fe2bb97542d387beca916d446ceb0680f264b539dd3` |
| `src/bouba_sens/loop.py` | `47eea148f4cfc41150a3570ed45600dab32ba88dc8b9411ad6fcf5135d0278a5` |
| `src/bouba_sens/lesion.py` | `ef2c0c2a5a1aab210e17ab709ec66a56359e870b14033240370dc1eef10bb3fe` |
| `src/bouba_sens/linear_probe.py` | `e85b80b5e4fe1a8484832f8eec2c7c04b83e11895b2b8df3a8928e3e435f65ff` |
| `src/bouba_sens/world/studyforrest.py` | `3b780f3abaff591687c851a13d7f28728d887c54b8a9554e4c2dd58c0b2d5b27` |
| `scripts/run_grid_probe_v2.sh` | `4a5eba8254d5929180ab56ac0d9732119364b7e0498ee83ca7b3c9c3fdc47a2d` |
| `scripts/run_grid.sh` | `1ae12c6e7a1bd0b61eaef5d1ed00a34736d4c23202910907a74c02c3a9dcade5` |
| `scripts/b3_probe_v2.py` | `c2d39c979b13db7de48bf6f3b0b5789beda052043f655837c0013efc3cf27a87` |
| `scripts/me6_paired_null.py` | `47160d6aa1bfcf674ee5272aabd01bcff07f9aa7a91e8e20361fa437fe57378f` |
| `scripts/me6_large_n.py` | `f5ddbaf484587876aa99aeb1bcfe036e6768e829d63ded6063add71b9334ba9c` |
| `uv.lock` | `551f04cff304e2df0aa4738d501fef60df40b5389fbc45a56abcaf3bc7768d29` |

## Code trees (git tree ids at the tagged commit)

| Tree | git tree id |
|---|---|
| `src/` | `0d7d7465b26bc3a6bea4bc2b40cefcc85baae02c` |
| `scripts/` | `fa5cebf7d4d1f4302707a21b3eab439951f1d076` |
| `tests/` | `bc63e36f65650b1d1ad2eb71e0e597d14bb8d361` |

The sha256 of this manifest is written in the tag message.
