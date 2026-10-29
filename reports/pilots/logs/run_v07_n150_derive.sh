#!/bin/bash
# v0.7 pilot-derived power at N = 150 (PI decision 2026-10-25 13:32):
# B-2 robust rule from pilot round 3, then B-1 robust rule from Q3/Q3+.
set -euo pipefail
export PATH=/opt/homebrew/bin:$PATH
cd "$HOME/hypneum-lab/pilots-pr9/bouba_sens"
echo "START $(date -u +%FT%TZ) $(date +%s) commit $(git rev-parse HEAD)"
nice -n 10 uv run --frozen python -u scripts/pilots/derive_v07_pilot.py \
  reports/pilots/v07_b2_pilot_r3_seeds1101-1105.json \
  reports/pilots/v07_b2_pilot_r3_derived_n150.json 150 > logs/v07_r3_derive_n150.log 2>&1
echo "B2-DONE $(date -u +%FT%TZ) $(date +%s)"
nice -n 10 uv run --frozen python -u scripts/pilots/derive_v07_b1_robust.py \
  reports/pilots/v07_b1_robust_power_n150.json > logs/v07_b1_robust_n150.log 2>&1
echo "END $(date -u +%FT%TZ) $(date +%s)"
