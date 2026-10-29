#!/bin/bash
# Pilots after the PI decisions of 2026-10-25 (19:47), sequential on macM1:
#  1. v0.7 B-2 pilot round 4 = round 3 config in the v07-b2-pilot namespace
#     (seeds 1101-1105), then its N = 150 derivations (robust and sign);
#  2. v0.7 B-1 SNR / world-class pilot cell, seed 4201, v07-b1-pilot namespace;
#  3. v0.8 rev. B pilot, seeds 2201-2210 (v08-pilot namespace);
#  4. v0.9 rev. B pilot, seeds 3201-3205 (v09-pilot namespace).
set -euo pipefail
export PATH=/opt/homebrew/bin:$PATH
export BOUBA_SENS_STUDYFORREST_DATA=$HOME/hypneum-lab/bouba_sens/data/studyforrest_real
cd "$HOME/hypneum-lab/pilots-pr9/bouba_sens"
RUNS="$HOME/hypneum-lab/pilots-pr9/runs"
P=reports/pilots
stamp() { echo "$1 $(date -u +%FT%TZ) $(date +%s)"; }
stamp "START commit $(git rev-parse HEAD)"

nice -n 10 uv run --frozen python -u scripts/spikes/e1_b2_dimensional_mine.py --preset v07-e2 \
  --seed-list 1101,1102,1103,1104,1105 --out $P/v07_b2_pilot_r4_seeds1101-1105.json
stamp "B2-DONE"
nice -n 10 uv run --frozen python -u scripts/pilots/derive_v07_pilot.py \
  $P/v07_b2_pilot_r4_seeds1101-1105.json $P/v07_b2_pilot_r4_derived_n150.json 150
nice -n 10 uv run --frozen python -u scripts/pilots/derive_v07_sign_power.py \
  $P/v07_b2_pilot_r4_seeds1101-1105.json $P/v07_sign_power_r4_n150.json
stamp "B2-DERIVE-DONE"

GRID="$RUNS/v07_snr_pilot_ns_seed4201"
SEEDS="4201" WORLD=studyforrest LOCK_AFTER=100 K_STEPS=50 OUT_ROOT="$GRID" \
  nice -n 10 uv run --frozen bash scripts/run_grid.sh
nice -n 10 uv run --frozen python scripts/aggregate_grid.py --root "$GRID" \
  --out $P/v07_snr_pilot_ns_seed4201_aggregate.json --require-world StudyforrestRealWorld
nice -n 10 uv run --frozen python - "$GRID" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
out = {}
for d in sorted(root.iterdir()):
    m = d / "metadata.json"
    if m.exists():
        meta = json.loads(m.read_text())
        rec = {k: meta.get(k) for k in ("phase", "world_class", "seed_namespace", "snr_floor", "k_steps")}
        traj = meta.get("snr_trajectory_db")
        if traj:
            rec.update(snr_step0=traj[0], snr_step50=traj[50], snr_step99=traj[99],
                       snr_mean=sum(traj) / len(traj))
        out[d.name] = rec
Path("reports/pilots/v07_snr_pilot_ns_seed4201_cells.json").write_text(json.dumps(out, indent=2) + "\n")
print(len(out), "outputs")
PY
stamp "SNR-DONE"

B8="$RUNS/v08_revb_pilot_blocks"
nice -n 10 uv run --frozen python -u scripts/run_v08_subspace.py capture \
  --seeds 2201,2202,2203,2204,2205,2206,2207,2208,2209,2210 --blocks-dir "$B8" --design revb
cp "$B8/capture_log.json" $P/v08_revb_pilot_capture_log.json
nice -n 10 uv run --frozen python -u scripts/run_v08_subspace.py pilot \
  --blocks-dir "$B8" --out $P/v08_revb_pilot_quantities.json --design revb
nice -n 10 uv run --frozen python -u scripts/pilots/derive_v08_pilot.py \
  $P/v08_revb_pilot_quantities.json $P/v08_revb_pilot_n_rule.json
stamp "V08-DONE"

nice -n 10 uv run --frozen python -u scripts/pilots/run_v09_manipulation_check.py --design revb \
  --out $P/v09_revb_manipulation_check_seeds3201-3205.json 3201 3202 3203 3204 3205
nice -n 10 uv run --frozen python -u scripts/run_v09_decorrelation.py --design revb \
  --seeds 3201,3202,3203,3204,3205 --out $P/v09_revb_pilot_main_grid.json
nice -n 10 uv run --frozen python -u scripts/pilots/derive_v09_pilot.py \
  $P/v09_revb_pilot_main_grid.json $P/v09_revb_pilot_n_rule.json
stamp "END"
