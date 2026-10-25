#!/bin/bash
set -euo pipefail
export PATH=/opt/homebrew/bin:$PATH
export BOUBA_SENS_STUDYFORREST_DATA=$HOME/hypneum-lab/bouba_sens/data/studyforrest_real
cd $HOME/hypneum-lab/pilots-pr9/bouba_sens
echo "START $(date -u +%FT%TZ) $(date +%s) commit $(git rev-parse HEAD)"
nice -n 10 uv run --frozen python -u scripts/spikes/e1_b2_dimensional_mine.py --preset v07-e2 --seed-list 1101,1102,1103,1104,1105 --out reports/pilots/v07_b2_pilot_r3_seeds1101-1105.json 2>&1 | while IFS= read -r l; do echo "$(date +%s) $l"; done
echo "END-RUN $(date -u +%FT%TZ) $(date +%s)"
nice -n 10 uv run --frozen python -u scripts/pilots/derive_v07_pilot.py reports/pilots/v07_b2_pilot_r3_seeds1101-1105.json reports/pilots/v07_b2_pilot_r3_derived.json > logs/v07_r3_derive.log 2>&1
echo "END $(date -u +%FT%TZ) $(date +%s)"
