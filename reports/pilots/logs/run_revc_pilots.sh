#!/bin/bash
# Rev. C pilots (PI decisions 2026-10-28, about 19:05), macM1, after the
# Codex r8 review of the implementation. Every heavy step runs under the
# shared heavy-job lock and only when at least 30 % of memory is free.
# Order: v0.8 (2211-2220), v0.9 CC-film (primary), v0.9 Studyforrest.
set -euo pipefail
export PATH=/opt/homebrew/bin:$PATH
export BOUBA_SENS_STUDYFORREST_DATA=$HOME/hypneum-lab/bouba_sens/data/studyforrest_real
export BOUBA_SENS_CCFILM_DATA=$HOME/hypneum-lab/bouba_sens/data/ccfilm_real_10fps
cd "$HOME/hypneum-lab/pilots-pr9/bouba_sens"
LOCK="$HOME/scratch/.heavy.lock"
P=reports/pilots
B8="$HOME/hypneum-lab/pilots-pr9/runs/v08_revc_pilot_blocks"
stamp() { echo "$1 $(date -u +%FT%TZ) $(date +%s)"; }
mem_ok() {
  until [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -ge 30 ]; do
    echo "waiting for memory $(date -u +%T)"; sleep 60
  done
  memory_pressure | tail -1
}
heavy() { mem_ok; /usr/bin/lockf -k "$LOCK" nice -n 10 "$@"; }
stamp "START commit $(git rev-parse HEAD)"
uv run --frozen python -c "import torch, sklearn, numpy; print('torch', torch.__version__, 'sklearn', sklearn.__version__, 'numpy', numpy.__version__)"
S8=2211,2212,2213,2214,2215,2216,2217,2218,2219,2220
heavy uv run --frozen python -u scripts/run_v08_subspace.py gate --seeds $S8 \
  --out $P/v08_revc_learning_gate.json || true
stamp "V08-GATE-DONE"
if uv run --frozen python -c "import json,sys; sys.exit(0 if json.load(open('$P/v08_revc_learning_gate.json'))['passed'] else 1)"; then
  heavy uv run --frozen python -u scripts/run_v08_subspace.py capture --design revc \
    --seeds $S8 --blocks-dir "$B8" --gate-file $P/v08_revc_learning_gate.json
  heavy uv run --frozen python -u scripts/run_v08_subspace.py pilot --design revc \
    --blocks-dir "$B8" --out $P/v08_revc_pilot_quantities.json
  cp "$B8/capture_log.json" $P/v08_revc_pilot_capture_log.json
  heavy uv run --frozen python -u scripts/pilots/derive_v08_pilot.py \
    $P/v08_revc_pilot_quantities.json $P/v08_revc_pilot_n_rule.json
  heavy uv run --frozen python -u scripts/pilots/derive_v08_pilot.py \
    $P/v08_revc_pilot_quantities_secondary_planning.json $P/v08_revc_pilot_n_rule_secondary.json
else
  echo "v0.8 learning gate failed: no capture (stop and report)"
fi
stamp "V08-DONE"
heavy uv run --frozen python -u scripts/run_v09_revc.py --world ccfilm \
  --out $P/v09_revc_ccfilm_pilot.json
heavy uv run --frozen python -u scripts/pilots/derive_v09_pilot.py \
  $P/v09_revc_ccfilm_pilot.json $P/v09_revc_ccfilm_n_rule.json
stamp "V09-CCFILM-DONE"
heavy uv run --frozen python -u scripts/run_v09_revc.py --world studyforrest \
  --out $P/v09_revc_studyforrest_pilot.json
heavy uv run --frozen python -u scripts/pilots/derive_v09_pilot.py \
  $P/v09_revc_studyforrest_pilot.json $P/v09_revc_studyforrest_n_rule.json
stamp "END"
