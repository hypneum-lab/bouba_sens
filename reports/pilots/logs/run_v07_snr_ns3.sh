#!/bin/bash
# v0.7 B-1 SNR / world-class pilot cell, seed 4201, namespace 7101, after
# the Codex r7 fix (namespaced eval --run probe).
set -euo pipefail
export PATH=/opt/homebrew/bin:$PATH
export BOUBA_SENS_STUDYFORREST_DATA=$HOME/hypneum-lab/bouba_sens/data/studyforrest_real
cd "$HOME/hypneum-lab/pilots-pr9/bouba_sens"
GRID="$HOME/hypneum-lab/pilots-pr9/runs/v07_snr_pilot_ns3_seed4201"
echo "START $(date -u +%FT%TZ) $(date +%s) commit $(git rev-parse HEAD)"
SEEDS="4201" WORLD=studyforrest LOCK_AFTER=100 K_STEPS=50 OUT_ROOT="$GRID" \
  nice -n 10 uv run --frozen bash scripts/run_grid.sh
nice -n 10 uv run --frozen python scripts/aggregate_grid.py --root "$GRID" \
  --out reports/pilots/v07_snr_pilot_ns3_seed4201_aggregate.json --require-world StudyforrestRealWorld
nice -n 10 uv run --frozen python - "$GRID" <<'PY'
import json, sys
from pathlib import Path
from omegaconf import OmegaConf
root = Path(sys.argv[1])
out = {}
for d in sorted(root.iterdir()):
    m = d / "metadata.json"
    if m.exists():
        meta = json.loads(m.read_text())
        rec = {k: meta.get(k) for k in ("phase", "seed", "world_class", "seed_namespace", "snr_floor", "k_steps")}
        traj = meta.get("snr_trajectory_db")
        if traj:
            rec.update(snr_step0=traj[0], snr_step50=traj[50], snr_step99=traj[99],
                       snr_mean=sum(traj) / len(traj))
        cfg = d / "config.yaml"
        if cfg.exists() and rec["phase"] == "lesion":
            rec["config_seed_namespace"] = OmegaConf.load(cfg).get("seed_namespace")
        ev = d / "eval_report.json"
        if ev.exists():
            e = json.loads(ev.read_text())
            rec.update(me1=e.get("me1"), me1_probe=e.get("me1_probe"))
        out[d.name] = rec
Path("reports/pilots/v07_snr_pilot_ns3_seed4201_cells.json").write_text(json.dumps(out, indent=2) + "\n")
print(len(out), "outputs")
PY
echo "END $(date -u +%FT%TZ) $(date +%s)"
