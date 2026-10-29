#!/bin/bash
# v0.9 pilot (seeds 3101-3105) under the class-stratified held-out split:
# manipulation check, then the main grid and the D variance, then N.
set -euo pipefail
export PATH=/opt/homebrew/bin:$PATH
export BOUBA_SENS_STUDYFORREST_DATA=$HOME/hypneum-lab/bouba_sens/data/studyforrest_real
cd "$HOME/hypneum-lab/pilots-pr9/bouba_sens"
echo "START $(date -u +%FT%TZ) $(date +%s) commit $(git rev-parse HEAD)"
nice -n 10 uv run --frozen python -u scripts/pilots/run_v09_manipulation_check.py \
  --split stratified --out reports/pilots/v09_manipulation_check_stratified_seeds3101-3105.json \
  3101 3102 3103 3104 3105
echo "CHECK-DONE $(date -u +%FT%TZ) $(date +%s)"
nice -n 10 uv run --frozen python -u scripts/run_v09_decorrelation.py \
  --seeds 3101,3102,3103,3104,3105 --out reports/pilots/v09_pilot_main_grid.json
echo "GRID-DONE $(date -u +%FT%TZ) $(date +%s)"
nice -n 10 uv run --frozen python -u scripts/pilots/derive_v09_pilot.py \
  reports/pilots/v09_pilot_main_grid.json reports/pilots/v09_pilot_n_rule.json
echo "END $(date -u +%FT%TZ) $(date +%s)"
