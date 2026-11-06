#!/bin/sh
export PATH=/opt/homebrew/bin:$PATH
cd ~/scratch/bouba-b2-0611/bouba_sens || exit 1
for tag in oldcache newcache; do
  if [ $tag = oldcache ]; then D=$HOME/hypneum-lab/bouba_sens/data/studyforrest_real; else D=$HOME/scratch/bouba-b2-0611/data/studyforrest_real; fi
  echo "== $tag start $(date -u +%FT%TZ) data=$D" >> ../logs/null_$tag.log
  BOUBA_SENS_STUDYFORREST_DATA=$D /usr/bin/time -p uv run python scripts/spikes/e1_b2_paired_null.py \
    --world studyforrest --seeds 5 --n-null 1000 --out ../out/e1_b2_paired_null_studyforrest_$tag.json >> ../logs/null_$tag.log 2>&1
  echo "== $tag rc=$? end $(date -u +%FT%TZ)" >> ../logs/null_$tag.log
done
