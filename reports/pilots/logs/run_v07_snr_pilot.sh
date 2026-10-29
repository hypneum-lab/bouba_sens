#!/bin/bash
# v0.7 B-1 SNR / world-class pilot cell (PI decision 2026-10-25 19:04):
# fresh pilot seed 4201 (outside every seed pool), v0.7 B-1 parameters
# (Studyforrest, 200 / 100 steps, batch 16, HARD gating, LOCK_AFTER 100,
# K_STEPS 50); then the aggregate with the world-class check; then the
# sign-test power at N = 150.
set -euo pipefail
export PATH=/opt/homebrew/bin:$PATH
export BOUBA_SENS_STUDYFORREST_DATA=$HOME/hypneum-lab/bouba_sens/data/studyforrest_real
cd "$HOME/hypneum-lab/pilots-pr9/bouba_sens"
GRID="$HOME/hypneum-lab/pilots-pr9/runs/v07_snr_pilot_seed4201"
echo "START $(date -u +%FT%TZ) $(date +%s) commit $(git rev-parse HEAD)"
SEEDS="4201" WORLD=studyforrest LOCK_AFTER=100 K_STEPS=50 OUT_ROOT="$GRID" \
  nice -n 10 uv run --frozen bash scripts/run_grid.sh
echo "GRID-DONE $(date -u +%FT%TZ) $(date +%s)"
nice -n 10 uv run --frozen python scripts/aggregate_grid.py --root "$GRID" \
  --out reports/pilots/v07_snr_pilot_seed4201_aggregate.json --require-world StudyforrestRealWorld
nice -n 10 uv run --frozen python - "$GRID" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
out = {}
for d in sorted(root.iterdir()):
    m = d / "metadata.json"
    if m.exists():
        meta = json.loads(m.read_text())
        rec = {k: meta.get(k) for k in ("phase", "world_class", "snr_init", "snr_floor", "k_steps", "steps")}
        traj = meta.get("snr_trajectory_db")
        if traj:
            rec["snr_step0"], rec["snr_step49"], rec["snr_step50"], rec["snr_step99"] = (
                traj[0], traj[49], traj[50], traj[99])
            rec["snr_mean"] = sum(traj) / len(traj)
        out[d.name] = rec
Path("reports/pilots/v07_snr_pilot_seed4201_cells.json").write_text(json.dumps(out, indent=2) + "\n")
print(len(out), "outputs")
PY
echo "CHECK-DONE $(date -u +%FT%TZ) $(date +%s)"
nice -n 10 uv run --frozen python -u scripts/pilots/derive_v07_sign_power.py \
  reports/pilots/v07_b2_pilot_r3_seeds1101-1105.json reports/pilots/v07_sign_power_n150.json
echo "END $(date -u +%FT%TZ) $(date +%s)"
