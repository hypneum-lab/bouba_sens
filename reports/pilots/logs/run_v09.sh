#!/bin/bash
set -euo pipefail
export PATH=/opt/homebrew/bin:$PATH
export BOUBA_SENS_STUDYFORREST_DATA=$HOME/hypneum-lab/bouba_sens/data/studyforrest_real
cd $HOME/hypneum-lab/pilots-pr9/bouba_sens
echo "START $(date -u +%FT%TZ)"
/usr/bin/time -p nice -n 10 uv run --frozen python -u scripts/pilots/run_v09_manipulation_check.py --out reports/pilots/v09_manipulation_check_seeds3101-3105.json 3101 3102 3103 3104 3105
echo "END $(date -u +%FT%TZ)"
