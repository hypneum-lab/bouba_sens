#!/usr/bin/env bash
# Sprint 4 Task 4.2 — 150-run grid orchestrator.
# 5 seeds x 5 modalities x 2 timings x 3 SNR = 150 cells.
# Idempotent: cells with an existing eval_report.json are skipped.
#
# WARNING (v0.7): the seed rule of the v2 probe changed in this script
# (ADR-0023 trio rule, seed*10000 + 5000 + trio, instead of the CLI
# default seed + 777). A grid started before this change must NOT be
# resumed. For v0.7 a fresh OUT_ROOT is mandatory. A guard below refuses
# to resume a cell whose probe_v2_meta.json is missing or has another seed.
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
# ADR-0024 follow-up — SensoryWML code emission: "argmax-bridge" (historical
# default) or "straight-through".
EMISSION="${EMISSION:-argmax-bridge}"
# ADR-0027 — world seed of lesion / eval: "legacy" (default, historical:
# lesion world seed 0, eval world seed = lesion seed) or "cell" (the
# phase-1 world, seed = the grid seed, for T1 and T2 cells alike).
WORLD_SEED_MODE="${WORLD_SEED_MODE:-legacy}"

# Seeds default to the 5-seed replication set used in paper v0.5. Q3
# (`scripts/replay_amedi_seeds.py`) overrides this via the `SEEDS` env
# var with a space-separated list to sweep custom seed sets without
# editing this file.
if [[ -n "${SEEDS:-}" ]]; then
    # shellcheck disable=SC2206
    SEEDS=(${SEEDS})
else
    SEEDS=(0 1 2 3 4)
fi
MODALITIES=(audio vision tactile gravity force)
TIMINGS=(T1 T2)

# (label floor_db)
SNR_CELLS=(
    "floor:-20.0"
    "minus10:-10.0"
    "plus10:10.0"
)

# Refuse to resume a finished cell that lacks probe_v2_meta.json or was
# probed with another seed (grid started under the old seed rule).
# usage: check_probe_seed <dir> <expected_seed> <done_marker_file>
check_probe_seed() {
    local dir="$1" want="$2" marker="$3"
    [[ -f "${marker}" ]] || return 0
    local meta="${dir}/probe_v2_meta.json"
    if [[ ! -f "${meta}" ]] || ! grep -Eq "\"probe_seed\": *${want}([^0-9]|$)" "${meta}"; then
        echo "ERROR: ${dir} has no probe_v2_meta.json with probe_seed ${want}." >&2
        echo "Grid predates the ADR-0023 seed rule; use a fresh OUT_ROOT." >&2
        exit 1
    fi
}

# ADR-0027: refuse to resume a finished cell that was run in another
# world-seed mode (mode read from its config.yaml; legacy if absent).
# Phase 1 needs no guard: `train` does not read the mode.
# usage: check_world_mode <cell_dir> <done_marker_file>
check_world_mode() {
    local dir="$1" marker="$2" have="legacy" line
    [[ -f "${marker}" ]] || return 0
    if [[ -f "${dir}/config.yaml" ]]; then
        line=$(grep -E '^world_seed_mode:' "${dir}/config.yaml" || true)
        if [[ -n "${line}" ]]; then
            have=$(printf '%s' "${line#world_seed_mode:}" | tr -d " '\"")
        fi
    fi
    if [[ "${have}" != "${WORLD_SEED_MODE}" ]]; then
        echo "ERROR: ${dir} was run with world_seed_mode ${have}," >&2
        echo "but WORLD_SEED_MODE=${WORLD_SEED_MODE}. Use one OUT_ROOT per arm (ADR-0027)." >&2
        exit 1
    fi
}

mkdir -p "${OUT_ROOT}"
total=$(( ${#SEEDS[@]} * ${#MODALITIES[@]} * ${#TIMINGS[@]} * ${#SNR_CELLS[@]} ))
echo "grid: ${total} cells -> ${OUT_ROOT}"

count=0
skipped=0
for seed in "${SEEDS[@]}"; do
    # One Phase 1 checkpoint per seed (reused across T2 cells for that seed).
    phase1_dir="${OUT_ROOT}/phase1_seed${seed}"
    check_probe_seed "${phase1_dir}" "$(( seed * 10000 + 5099 ))" "${phase1_dir}/checkpoint.pkl"
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
        train_args+=(--emission "${EMISSION}")
        train_args+=(--probe-v2 --probe-v2-seed $(( seed * 10000 + 5099 )))
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
                    timing_idx=0; [[ "${timing}" == "T2" ]] && timing_idx=1
                    case "${snr_label}" in floor) snr_idx=0 ;; minus10) snr_idx=1 ;; *) snr_idx=2 ;; esac
                    check_probe_seed "${cell_dir}" "$(( seed * 10000 + 5000 + 3 * timing_idx + snr_idx ))" "${eval_json}"
                    check_world_mode "${cell_dir}" "${eval_json}"
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
                lesion_args+=(--emission "${EMISSION}")
                if [[ "${WORLD_SEED_MODE}" != "legacy" ]]; then
                    lesion_args+=(--world-seed-mode "${WORLD_SEED_MODE}" --world-seed "${seed}")
                fi
                # ADR-0023: v2 probe, seed shared per (seed, timing, snr) trio,
                # same rule as run_grid_probe_v2.sh.
                timing_idx=0; [[ "${timing}" == "T2" ]] && timing_idx=1
                case "${snr_label}" in floor) snr_idx=0 ;; minus10) snr_idx=1 ;; *) snr_idx=2 ;; esac
                trio=$(( 3 * timing_idx + snr_idx ))
                lesion_args+=(--probe-v2 --probe-v2-seed $(( seed * 10000 + 5000 + trio )))
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
