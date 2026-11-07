#!/bin/sh
# ADR-0019 amendment 2026-11-06: stratified B-2 probe, then the paired
# null, on the corrected (newcache) and first (oldcache) Studyforrest
# caches. Run on macM1 under /usr/bin/lockf -k ~/scratch/.heavy.lock.
export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
cd ~/scratch/bouba-b2-0611/bouba_sens || exit 1
L=../logs
O=../out
for tag in newcache oldcache; do
  if [ $tag = oldcache ]; then D=$HOME/hypneum-lab/bouba_sens/data/studyforrest_real; else D=$HOME/scratch/bouba-b2-0611/data/studyforrest_real; fi
  echo "== b2 $tag start $(date -u +%FT%TZ) data=$D commit=$(git rev-parse --short HEAD)" >> $L/strat_b2_$tag.log
  BOUBA_SENS_STUDYFORREST_DATA=$D /usr/bin/time -p uv run python scripts/spikes/e1_b2_dimensional_mine.py \
    --world studyforrest --seeds 5 --lock-after 100 --steps-train 200 --steps-lesion 100 \
    --batch-size 512 --bootstrap 500 --probe-sampling stratified \
    --out $O/e1_b2_dimensional_mine_studyforrest_stratified_$tag.json >> $L/strat_b2_$tag.log 2>&1
  echo "== b2 $tag rc=$? end $(date -u +%FT%TZ)" >> $L/strat_b2_$tag.log
done
for tag in newcache oldcache; do
  if [ $tag = oldcache ]; then D=$HOME/hypneum-lab/bouba_sens/data/studyforrest_real; else D=$HOME/scratch/bouba-b2-0611/data/studyforrest_real; fi
  echo "== null $tag start $(date -u +%FT%TZ) data=$D commit=$(git rev-parse --short HEAD)" >> $L/strat_null_$tag.log
  BOUBA_SENS_STUDYFORREST_DATA=$D /usr/bin/time -p uv run python scripts/spikes/e1_b2_paired_null.py \
    --world studyforrest --seeds 5 --n-null 1000 --probe-sampling stratified \
    --out $O/e1_b2_paired_null_studyforrest_stratified_$tag.json >> $L/strat_null_$tag.log 2>&1
  echo "== null $tag rc=$? end $(date -u +%FT%TZ)" >> $L/strat_null_$tag.log
done
