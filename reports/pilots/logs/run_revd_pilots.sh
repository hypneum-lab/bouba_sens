#!/bin/bash
# v0.9 rev. D pilots (PI decision 2026-10-29, 06:10), macM1: same rule as
# rev. C, recovery R = 1,750, fresh seeds 3211-3215 (namespace 9101).
# CC-film (primary) first, then Studyforrest (secondary). Each world holds
# the shared heavy-job lock for its whole run (lockf -k), and starts only
# with >= 30 % memory free.
set -euo pipefail
export PATH=/opt/homebrew/bin:$PATH
export BOUBA_SENS_STUDYFORREST_DATA=$HOME/hypneum-lab/bouba_sens/data/studyforrest_real
export BOUBA_SENS_CCFILM_DATA=$HOME/hypneum-lab/bouba_sens/data/ccfilm_real_10fps
cd "$HOME/hypneum-lab/pilots-pr9/bouba_sens"
LOCK="$HOME/scratch/.heavy.lock"
P=reports/pilots
stamp() { echo "$1 $(date -u +%FT%TZ) $(date +%s)"; }
mem_ok() {
  until [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -ge 30 ]; do
    echo "waiting for memory $(date -u +%T)"; sleep 60
  done
  memory_pressure | tail -1
}
stamp "START commit $(git rev-parse HEAD)"
uv run --frozen python -c "import torch, sklearn, numpy; print('torch', torch.__version__, 'sklearn', sklearn.__version__, 'numpy', numpy.__version__)"
for W in ccfilm studyforrest; do
  mem_ok
  /usr/bin/lockf -k "$LOCK" nice -n 10 bash -c "
    uv run --frozen python -u scripts/run_v09_revc.py --budget revd --world $W \
      --out $P/v09_revd_${W}_pilot.json &&
    uv run --frozen python -u scripts/pilots/derive_v09_pilot.py \
      $P/v09_revd_${W}_pilot.json $P/v09_revd_${W}_n_rule.json"
  stamp "V09-REVD-${W}-DONE"
done
stamp "END"
