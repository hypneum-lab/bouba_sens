#!/usr/bin/env bash
# Sprint 3 Task 3.10 — one-liner wrapper around the 5-command CLI pipeline.
# Ends in a Jinja2 HTML summary for a single-seed audio-M2-T2 smoke run.
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-./runs/v01_smoke_$(date +%s)}"
SEED="${SEED:-0}"
STEPS_TRAIN="${STEPS_TRAIN:-100}"
STEPS_LESION="${STEPS_LESION:-100}"

echo "v01 smoke run -> ${OUT_ROOT}"

mkdir -p "${OUT_ROOT}"

uv run bouba-sens train \
    --steps "${STEPS_TRAIN}" \
    --batch-size 16 \
    --seed "${SEED}" \
    --out "${OUT_ROOT}/phase1"

uv run bouba-sens lesion \
    --ckpt "${OUT_ROOT}/phase1" \
    --modality audio \
    --steps "${STEPS_LESION}" \
    --seed 10000 \
    --out "${OUT_ROOT}/phase2_t2"

uv run bouba-sens eval \
    --run "${OUT_ROOT}/phase2_t2" \
    --metrics "Me1,Me2,Me7" \
    --out "${OUT_ROOT}/phase2_t2/eval_report.json"

uv run bouba-sens aggregate \
    --glob "${OUT_ROOT}/phase2_t2" \
    --out "${OUT_ROOT}/summary.html"

echo "v01 smoke done; open ${OUT_ROOT}/summary.html"
