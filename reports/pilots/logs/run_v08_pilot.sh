#!/bin/bash
# v0.8 pilot (seeds 2101-2110): capture every (seed, lock) block, then k*
# and the section 7 planning inputs, then the N rule.
set -euo pipefail
export PATH=/opt/homebrew/bin:$PATH
export BOUBA_SENS_STUDYFORREST_DATA=$HOME/hypneum-lab/bouba_sens/data/studyforrest_real
cd "$HOME/hypneum-lab/pilots-pr9/bouba_sens"
BLOCKS="$HOME/hypneum-lab/pilots-pr9/runs/v08_pilot_blocks"
echo "START $(date -u +%FT%TZ) $(date +%s) commit $(git rev-parse HEAD)"
nice -n 10 uv run --frozen python -u scripts/run_v08_subspace.py capture \
  --seeds 2101,2102,2103,2104,2105,2106,2107,2108,2109,2110 --blocks-dir "$BLOCKS"
echo "CAPTURE-DONE $(date -u +%FT%TZ) $(date +%s)"
cp "$BLOCKS/capture_log.json" reports/pilots/v08_pilot_capture_log.json
nice -n 10 uv run --frozen python -u scripts/run_v08_subspace.py pilot \
  --blocks-dir "$BLOCKS" --out reports/pilots/v08_pilot_quantities.json
echo "PILOT-DONE $(date -u +%FT%TZ) $(date +%s)"
nice -n 10 uv run --frozen python -u scripts/pilots/derive_v08_pilot.py \
  reports/pilots/v08_pilot_quantities.json reports/pilots/v08_pilot_n_rule.json
echo "END $(date -u +%FT%TZ) $(date +%s)"
