#!/bin/bash
# Resume of the rev. B pilots after the 2026-10-25 interruption: the v0.8
# blocks (seeds 2201-2210) were captured at 680e979; this computes the v0.8
# pilot summary and N rule at the fixed commit, then runs the v0.9 rev. B
# pilot (seeds 3201-3205). Heavy steps run under the shared heavy-job lock.
set -euo pipefail
export PATH=/opt/homebrew/bin:$PATH
export BOUBA_SENS_STUDYFORREST_DATA=$HOME/hypneum-lab/bouba_sens/data/studyforrest_real
cd "$HOME/hypneum-lab/pilots-pr9/bouba_sens"
LOCK="$HOME/scratch/.heavy.lock"
P=reports/pilots
B8="$HOME/hypneum-lab/pilots-pr9/runs/v08_revb_pilot_blocks"
stamp() { echo "$1 $(date -u +%FT%TZ) $(date +%s)"; }
stamp "START commit $(git rev-parse HEAD)"
cp "$B8/capture_log.json" $P/v08_revb_pilot_capture_log.json
/usr/bin/lockf -k "$LOCK" nice -n 10 uv run --frozen python -u scripts/run_v08_subspace.py pilot \
  --blocks-dir "$B8" --out $P/v08_revb_pilot_quantities.json --design revb
nice -n 10 uv run --frozen python -u scripts/pilots/derive_v08_pilot.py \
  $P/v08_revb_pilot_quantities.json $P/v08_revb_pilot_n_rule.json
stamp "V08-DONE"
memory_pressure | tail -1
/usr/bin/lockf -k "$LOCK" nice -n 10 uv run --frozen python -u scripts/pilots/run_v09_manipulation_check.py \
  --design revb --out $P/v09_revb_manipulation_check_seeds3201-3205.json 3201 3202 3203 3204 3205
/usr/bin/lockf -k "$LOCK" nice -n 10 uv run --frozen python -u scripts/run_v09_decorrelation.py \
  --design revb --seeds 3201,3202,3203,3204,3205 --out $P/v09_revb_pilot_main_grid.json
nice -n 10 uv run --frozen python -u scripts/pilots/derive_v09_pilot.py \
  $P/v09_revb_pilot_main_grid.json $P/v09_revb_pilot_n_rule.json
stamp "END"
