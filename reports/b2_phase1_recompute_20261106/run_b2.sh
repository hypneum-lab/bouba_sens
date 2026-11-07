#!/bin/sh
# B-2 replay (ADR-0019 E1.3c defaults): old cache first (reproduction check), then corrected cache.
export PATH=/opt/homebrew/bin:$PATH
cd ~/scratch/bouba-b2-0611/bouba_sens || exit 1
L=../logs
for tag in oldcache newcache; do
  if [ $tag = oldcache ]; then D=$HOME/hypneum-lab/bouba_sens/data/studyforrest_real; else D=$HOME/scratch/bouba-b2-0611/data/studyforrest_real; fi
  echo "== $tag start $(date -u +%FT%TZ) data=$D" >> $L/b2_$tag.log
  BOUBA_SENS_STUDYFORREST_DATA=$D /usr/bin/time -p uv run python scripts/spikes/e1_b2_dimensional_mine.py \
    --world studyforrest --seeds 5 --lock-after 100 --steps-train 200 --steps-lesion 100 \
    --batch-size 512 --bootstrap 500 --out ../out/e1_b2_dimensional_mine_studyforrest_$tag.json >> $L/b2_$tag.log 2>&1
  echo "== $tag rc=$? end $(date -u +%FT%TZ)" >> $L/b2_$tag.log
done
