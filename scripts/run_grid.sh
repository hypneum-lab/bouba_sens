#!/usr/bin/env bash
# Sprint 4 Task 4.2 — 150-run grid orchestrator.
# 5 seeds x 5 modalities x 2 timings x 3 SNR = 150 cells.
# Idempotent: cells with an existing eval_report.json are skipped.
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-./runs/v01_grid}"
STEPS_TRAIN="${STEPS_TRAIN:-200}"
STEPS_LESION="${STEPS_LESION:-100}"
METRICS="${METRICS:-Me1,Me2,Me3}"
WORLD="${WORLD:-gaussian}"
# nerve-wml#4 critical-period lock. When set (e.g. LOCK_AFTER=STEPS_TRAIN),
# T2 cells inherit a frozen constellation at Phase 2 entry (mux was trained
# through Phase 1 and crossed the lock threshold); T1 cells receive a fresh
# mux that locks mid-Phase-2 OR never locks if the threshold exceeds
# STEPS_LESION. Empty = legacy v1.3 behaviour (no lock).
LOCK_AFTER="${LOCK_AFTER:-}"
# Sprint 11 — transducer gating. "hard" (default) = v0.5 binary rule;
# "gumbel" = sigmoid-soft activation per nerve-wml#5 for compound
# critical-period experiments.
TRANSDUCER_GATING="${TRANSDUCER_GATING:-hard}"
# Sprint 12 — gumbel sigmoid temperature. Only read when
# TRANSDUCER_GATING=gumbel. Lower = tighter toward hard; higher = flatter.
GUMBEL_TAU="${GUMBEL_TAU:-1.0}"
# Sprint 13 — AdaptiveCodebook freeze after N training steps. Empty = no freeze.
CODEBOOK_LOCK="${CODEBOOK_LOCK:-}"
# Sprint 14 — transducer gating phase-transition schedule. Empty = no schedule.
# GATING_SCHEDULE=200 with GATING_TARGET=gumbel + TRANSDUCER_GATING=hard =
# HARD during Phase 1 (0..200), GUMBEL during Phase 2 (200..300) for T2 cells.
GATING_SCHEDULE="${GATING_SCHEDULE:-}"
GATING_TARGET="${GATING_TARGET:-}"

SEEDS=(0 1 2 3 4)
MODALITIES=(audio vision tactile gravity force)
TIMINGS=(T1 T2)

# (label floor_db)
SNR_CELLS=(
    "floor:-20.0"
    "minus10:-10.0"
    "plus10:10.0"
)

mkdir -p "${OUT_ROOT}"
total=$(( ${#SEEDS[@]} * ${#MODALITIES[@]} * ${#TIMINGS[@]} * ${#SNR_CELLS[@]} ))
echo "grid: ${total} cells -> ${OUT_ROOT}"

count=0
skipped=0
for seed in "${SEEDS[@]}"; do
    # One Phase 1 checkpoint per seed (reused across T2 cells for that seed).
    phase1_dir="${OUT_ROOT}/phase1_seed${seed}"
    if [[ ! -f "${phase1_dir}/checkpoint.pkl" ]]; then
        echo "[seed ${seed}] phase 1 pretrain ${STEPS_TRAIN} steps -> ${phase1_dir}"
        train_args=(
            --steps "${STEPS_TRAIN}"
            --batch-size 16
            --seed "${seed}"
            --world "${WORLD}"
            --out "${phase1_dir}"
        )
        [[ -n "${LOCK_AFTER}" ]] && train_args+=(--constellation-lock-after "${LOCK_AFTER}")
        train_args+=(--transducer-gating "${TRANSDUCER_GATING}")
        train_args+=(--gumbel-tau "${GUMBEL_TAU}")
        [[ -n "${CODEBOOK_LOCK}" ]] && train_args+=(--codebook-lock-after "${CODEBOOK_LOCK}")
        [[ -n "${GATING_SCHEDULE}" ]] && train_args+=(--transducer-gating-schedule "${GATING_SCHEDULE}")
        [[ -n "${GATING_TARGET}" ]] && train_args+=(--transducer-gating-target "${GATING_TARGET}")
        uv run bouba-sens train "${train_args[@]}"
    fi

    for modality in "${MODALITIES[@]}"; do
        for timing in "${TIMINGS[@]}"; do
            for snr_cell in "${SNR_CELLS[@]}"; do
                snr_label="${snr_cell%%:*}"
                snr_floor="${snr_cell#*:}"

                cell_name="seed${seed}_${timing}_${modality}_${snr_label}"
                cell_dir="${OUT_ROOT}/${cell_name}"
                eval_json="${cell_dir}/eval_report.json"

                count=$(( count + 1 ))

                if [[ -f "${eval_json}" ]]; then
                    skipped=$(( skipped + 1 ))
                    continue
                fi

                echo "[${count}/${total}] ${cell_name}"

                lesion_args=(
                    --modality "${modality}"
                    --timing "${timing}"
                    --steps "${STEPS_LESION}"
                    --seed $(( seed * 10000 + count ))
                    --snr-floor "${snr_floor}"
                    --world "${WORLD}"
                    --out "${cell_dir}"
                )
                [[ -n "${LOCK_AFTER}" ]] && lesion_args+=(--constellation-lock-after "${LOCK_AFTER}")
                lesion_args+=(--transducer-gating "${TRANSDUCER_GATING}")
                lesion_args+=(--gumbel-tau "${GUMBEL_TAU}")
                [[ -n "${CODEBOOK_LOCK}" ]] && lesion_args+=(--codebook-lock-after "${CODEBOOK_LOCK}")
                [[ -n "${GATING_SCHEDULE}" ]] && lesion_args+=(--transducer-gating-schedule "${GATING_SCHEDULE}")
                [[ -n "${GATING_TARGET}" ]] && lesion_args+=(--transducer-gating-target "${GATING_TARGET}")
                if [[ "${timing}" == "T1" ]]; then
                    # Congenital: skip pretrain, no checkpoint.
                    uv run bouba-sens lesion "${lesion_args[@]}"
                else
                    # Late-acquired: restore phase 1 checkpoint.
                    uv run bouba-sens lesion --ckpt "${phase1_dir}" "${lesion_args[@]}"
                fi

                uv run bouba-sens eval \
                    --run "${cell_dir}" \
                    --metrics "${METRICS}" \
                    --out "${eval_json}"
            done
        done
    done
done

echo "grid done; ${count} cells processed, ${skipped} skipped (already had eval_report.json)"
echo "next: python scripts/aggregate_grid.py --root ${OUT_ROOT} --out reports/v0.1_aggregate.json"
