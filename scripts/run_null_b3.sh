#!/usr/bin/env bash
# Sprint 7 Task 7.1 — null-model B-3 grid launcher.
# Reuses the Sprint 4 grid logic but injects a random-partition
# label remapping before the aggregator forms the 5x5 perf matrix.
#
# NOTE: SEEDS / MODALITIES / SNR_LEVELS env vars are NOT forwarded to
# run_grid.sh because that script uses hardcoded bash arrays. To run a
# reduced grid for testing, edit run_grid.sh directly or use a separate
# wrapper. On Studio the full 150-cell grid runs with default settings.
set -euo pipefail

WORLD="${WORLD:-gaussian}"
STEPS_TRAIN="${STEPS_TRAIN:-200}"
STEPS_LESION="${STEPS_LESION:-100}"
OUT_ROOT="${OUT_ROOT:-runs/null_b3}"
METRICS="${METRICS:-Me1,Me2,Me3,Me6}"
PARTITION_SEED="${PARTITION_SEED:-0}"
PARTITION_INDEX="${PARTITION_INDEX:-0}"

mkdir -p "${OUT_ROOT}"
export BOUBA_SENS_NULL_PARTITION_SEED="${PARTITION_SEED}"
export BOUBA_SENS_NULL_PARTITION_INDEX="${PARTITION_INDEX}"

PATH=/opt/homebrew/bin:$PATH \
  WORLD="${WORLD}" STEPS_TRAIN="${STEPS_TRAIN}" STEPS_LESION="${STEPS_LESION}" \
  OUT_ROOT="${OUT_ROOT}" METRICS="${METRICS}" \
  bash scripts/run_grid.sh

# Aggregate with the same random partition
PATH=/opt/homebrew/bin:$PATH \
  uv run python scripts/aggregate_grid.py \
    --root "${OUT_ROOT}" \
    --out  "${OUT_ROOT}/aggregate.json" \
    --partition-seed  "${PARTITION_SEED}" \
    --partition-index "${PARTITION_INDEX}"
