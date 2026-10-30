#!/usr/bin/env bash
# ADR-0023 — parallel 150-cell grid with the B-3 probe v2.
#
# Same cells, names, CLI flags and per-cell lesion seeds as
# scripts/run_grid.sh (seed * 10000 + count, count in run_grid.sh order;
# STEPS_TRAIN 200, STEPS_LESION 100, METRICS Me1,Me2,Me3, gating hard,
# gumbel tau 1.0), run PAR cells at a time (the electron-server twin of
# ADR-0022, ~/scratch/bouba-paired/run_grid_parallel.sh). Two additions,
# fixed by the preregistration docs/prereg/2026-10-25-b3-probe-v2.md:
#
# - the five cells of a (seed, timing, snr) trio share ONE v2 probe seed,
#   seed * 10000 + 5000 + trio, trio = 3 * timing_index + snr_index
#   (T1 = 0, T2 = 1; floor = 0, minus10 = 1, plus10 = 2), so P[q, m] and
#   P[m, q] are scored on the same held-out world items;
# - the intact network of each seed (phase 1) is probed with seed
#   seed * 10000 + 5099 (positive control), and so is its untrained
#   initial state (descriptive reference).
#
# Idempotent: cells with eval_report.json are skipped.
# Usage: SEEDS="200 201" WORLD=gaussian OUT_ROOT=... PAR=4 scripts/run_grid_probe_v2.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT_ROOT="${OUT_ROOT:?set OUT_ROOT}"
LOG_DIR="${LOG_DIR:-${OUT_ROOT}_logs}"
PAR="${PAR:-4}"
STEPS_TRAIN="${STEPS_TRAIN:-200}"
STEPS_LESION="${STEPS_LESION:-100}"
METRICS="${METRICS:-Me1,Me2,Me3}"
WORLD="${WORLD:-gaussian}"
# ADR-0024 follow-up: SensoryWML emission (argmax-bridge = historical default).
EMISSION="${EMISSION:-argmax-bridge}"
# ADR-0027: world seed of lesion / eval (legacy = historical, cell = phase-1 world).
WORLD_SEED_MODE="${WORLD_SEED_MODE:-legacy}"
# shellcheck disable=SC2206
SEEDS=(${SEEDS:?set SEEDS, e.g. "200 201 202 203 204"})
MODALITIES=(audio vision tactile gravity force)
TIMINGS=(T1 T2)
SNR_CELLS=("floor:-20.0" "minus10:-10.0" "plus10:10.0")

export REPO OUT_ROOT LOG_DIR STEPS_TRAIN STEPS_LESION METRICS WORLD EMISSION WORLD_SEED_MODE
mkdir -p "$OUT_ROOT" "$LOG_DIR"
cd "$REPO"
total=$(( ${#SEEDS[@]} * ${#MODALITIES[@]} * ${#TIMINGS[@]} * ${#SNR_CELLS[@]} ))
echo "[$(date -Is)] grid start: $(git rev-parse --short HEAD 2>/dev/null || echo nogit) world=$WORLD emission=$EMISSION world_seed_mode=$WORLD_SEED_MODE seeds=${SEEDS[*]} cells=$total PAR=$PAR -> $OUT_ROOT"

phase1_job() {
    local seed="$1"
    local dir="$OUT_ROOT/phase1_seed${seed}"
    if [[ -f "$dir/checkpoint.pkl" ]]; then
        echo "[phase1 seed $seed] skip"
        return 0
    fi
    local t0; t0=$(date +%s)
    uv run bouba-sens train --steps "$STEPS_TRAIN" --batch-size 16 --seed "$seed" \
        --world "$WORLD" --out "$dir" --transducer-gating hard --gumbel-tau 1.0 \
        --probe-v2 --probe-v2-seed $(( seed * 10000 + 5099 )) --emission "$EMISSION" \
        > "$LOG_DIR/phase1_seed${seed}.log" 2>&1
    echo "[phase1 seed $seed] done in $(( $(date +%s) - t0 )) s"
}

cell_job() {
    local seed="$1" modality="$2" timing="$3" snr_label="$4" snr_floor="$5" count="$6" trio="$7"
    local cell_name="seed${seed}_${timing}_${modality}_${snr_label}"
    local cell_dir="$OUT_ROOT/$cell_name"
    local eval_json="$cell_dir/eval_report.json"
    if [[ -f "$eval_json" ]]; then
        echo "[$count] $cell_name skip"
        return 0
    fi
    local t0; t0=$(date +%s)
    local args=(--modality "$modality" --timing "$timing" --steps "$STEPS_LESION"
        --seed $(( seed * 10000 + count )) --snr-floor "$snr_floor" --world "$WORLD"
        --out "$cell_dir" --transducer-gating hard --gumbel-tau 1.0
        --probe-v2 --probe-v2-seed $(( seed * 10000 + 5000 + trio )) --emission "$EMISSION")
    if [[ "$WORLD_SEED_MODE" != "legacy" ]]; then
        args+=(--world-seed-mode "$WORLD_SEED_MODE" --world-seed "$seed")
    fi
    {
        if [[ "$timing" == "T1" ]]; then
            uv run bouba-sens lesion "${args[@]}"
        else
            uv run bouba-sens lesion --ckpt "$OUT_ROOT/phase1_seed${seed}" "${args[@]}"
        fi
        uv run bouba-sens eval --run "$cell_dir" --metrics "$METRICS" --out "$eval_json"
    } > "$LOG_DIR/${cell_name}.log" 2>&1
    echo "[$count] $cell_name done in $(( $(date +%s) - t0 )) s"
}
export -f phase1_job cell_job

printf '%s\n' "${SEEDS[@]}" | xargs -P "$PAR" -I{} bash -c 'phase1_job "$@"' _ {}

count=0
for seed in "${SEEDS[@]}"; do
    for modality in "${MODALITIES[@]}"; do
        for t_idx in "${!TIMINGS[@]}"; do
            for s_idx in "${!SNR_CELLS[@]}"; do
                snr_cell="${SNR_CELLS[$s_idx]}"
                count=$(( count + 1 ))
                echo "$seed $modality ${TIMINGS[$t_idx]} ${snr_cell%%:*} ${snr_cell#*:} $count $(( 3 * t_idx + s_idx ))"
            done
        done
    done
done | xargs -P "$PAR" -L 1 bash -c 'cell_job "$@"' _

echo "[$(date -Is)] grid done"
