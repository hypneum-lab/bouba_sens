#!/bin/bash
# B-1 grid timing / SNR check on seed 0 (historical default grid seed,
# outside every amendment pool): 30 lesion cells, LOCK_AFTER 100, K_STEPS 50.
set -euo pipefail
export PATH=/opt/homebrew/bin:$PATH
export BOUBA_SENS_STUDYFORREST_DATA=$HOME/hypneum-lab/bouba_sens/data/studyforrest_real
cd "$HOME/hypneum-lab/pilots-pr9/bouba_sens"
echo "START $(date -u +%FT%TZ) $(date +%s) commit $(git rev-parse HEAD)"
SEEDS="0" WORLD=studyforrest LOCK_AFTER=100 K_STEPS=50 \
  OUT_ROOT="$HOME/hypneum-lab/pilots-pr9/runs/b1_timing_seed0" \
  nice -n 10 uv run --frozen bash scripts/run_grid.sh
echo "END $(date -u +%FT%TZ) $(date +%s)"
