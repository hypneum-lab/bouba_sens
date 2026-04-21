# v0.3 Critical Validation Artefacts — Sprint 7 MANIFEST

Generated 2026-04-21. Reproducibility pinned to commit tree under tag `v0.4.0`.

## Artefacts

| File | SHA256 | Produced by |
|------|--------|-------------|
| `me7_bootstrap.json` | `0473a9c1eb4ab5a27e28598ed01e123e40afc593dcb812386b653148670a787d` | Task 7.2 |
| `mi_estimator_comparison.json` | `0699b6c23dad9bf5f923c454ce22c15c31c8f6fa2ee0f64e3a9b0b24924821fe` | Task 7.3 |
| `null_b3_part_0_partial.json` | `c3cf33a0c6251888e43be1a3eef47d40d618ea52e422d6a148246acc81529612` | Task 7.1 (partial, 1/10) |

## Reproduction commands (Studio)

### Task 7.2 — Me7 bootstrap CI per world

```bash
cd ~/Projets/bouba_sens
git checkout sprint7/critical-validation
uv sync --all-extras

# Re-aggregate the v0.2 grids so raw_me7_pairs is emitted
for w in gaussian xor sinusoid; do
  src=runs/v02_grid
  [ "$w" = xor ]      && src=runs/v02_xor
  [ "$w" = sinusoid ] && src=runs/v02_sinusoid
  uv run python scripts/aggregate_grid.py --root "$src" \
    --out reports/v0.2_aggregate_${w}.json
done

# Bootstrap 95 % CIs (10_000 resamples, seed=0)
uv run python scripts/bootstrap_me7.py \
  --gaussian reports/v0.2_aggregate_gaussian.json \
  --xor      reports/v0.2_aggregate_xor.json \
  --sinusoid reports/v0.2_aggregate_sinusoid.json \
  --out      reports/v0.3_critical_validation/me7_bootstrap.json \
  --n-boot   10000 --seed 0
```

### Task 7.3 — Me3 estimator comparison

```bash
cd ~/Projets/bouba_sens
uv run python scripts/compare_mi_estimators.py gaussian xor sinusoid \
  --runs-root runs \
  --out reports/v0.3_critical_validation/mi_estimator_comparison.json
```

Wall-clock ~5 min on Studio (MINE 300 epochs × 450 cells).

### Task 7.1 — null-model B-3 (partial, 1 / 10 partitions)

```bash
cd ~/Projets/bouba_sens
# Only index=0 completed before Studio branch drift on 2026-04-21 09:24.
env PATH=/opt/homebrew/bin:$PATH \
  PARTITION_SEED=0 PARTITION_INDEX=0 WORLD=gaussian \
  STEPS_TRAIN=200 STEPS_LESION=100 \
  OUT_ROOT=runs/null_b3/part_0 METRICS='Me1,Me2,Me3' \
  bash scripts/run_null_b3.sh
```

Partition index 0 corresponds to `({force, tactile, vision}, {audio, gravity})` per `bouba_sens.metrics.partitions.generate_random_3_2_partitions(n=20, seed=0, unique=True)[0]`.

**Sprint 8 pending** — indices 1-9 + an `--partition-prereg` flag for apples-to-apples comparison with pre-reg.

## Integrity verification

```bash
shasum -a 256 reports/v0.3_critical_validation/*.json
```

Expected output must match the table above.
