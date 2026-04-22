#!/usr/bin/env bash
# Sprint 16 Task 4 — one-command reproduction of every grid cited
# in paper v0.1 (§4 main results + §5 mechanism + §5.9 Sprint 14).
#
# Regenerates `reports/*_aggregate.json` from scratch using the
# pipeline at tag `v0.5.5`. Total compute budget ~8 h on an
# M3 Ultra with 5-way seed concurrency, OR run individual blocks
# via env CHOOSE=<block>.
#
# Usage::
#
#     uv sync --all-extras
#     bash scripts/reproduce_paper_v01.sh                 # full pipeline
#     CHOOSE=v0.5_s10_dr bash scripts/reproduce_paper_v01.sh  # one block
#
# Idempotent: cells with an existing `eval_report.json` are skipped
# by `run_grid.sh`, so a re-run only fills in gaps.
set -euo pipefail

CHOOSE="${CHOOSE:-all}"
OUT_BASE="${OUT_BASE:-runs}"
REPORTS="${REPORTS:-reports}"
STUDYFORREST_DATA="${BOUBA_SENS_STUDYFORREST_DATA:-data/sf_phase2_adapted}"

mkdir -p "${OUT_BASE}" "${REPORTS}"

# Helper: run_grid.sh wrapper that passes through its env and then
# aggregates the grid into a named JSON. Honours CHOOSE gating.
_run() {
    local label="$1"; shift
    local out_json="$1"; shift
    if [[ "${CHOOSE}" != "all" && "${CHOOSE}" != "${label}" ]]; then
        return 0
    fi
    echo "=== ${label} -> ${out_json} ==="
    env "$@" bash scripts/run_grid.sh
    # Recover OUT_ROOT from the environment vars passed in.
    local out_root
    out_root=$(env "$@" bash -c 'echo "${OUT_ROOT:-runs/v01_grid}"')
    uv run python scripts/aggregate_grid.py \
        --root "${out_root}" --out "${out_json}"
}

# -- §4 main results : unlocked grids ---------------------------------
_run v0.1_smoke "${REPORTS}/v0.1_aggregate.json" \
    OUT_ROOT="${OUT_BASE}/v01_gauss_nolock" WORLD=gaussian
_run v0.3_xor "${REPORTS}/v0.3_xor_aggregate.json" \
    OUT_ROOT="${OUT_BASE}/v03_xor" WORLD=xor
_run v0.3_sinusoid "${REPORTS}/v0.3_sinusoid_aggregate.json" \
    OUT_ROOT="${OUT_BASE}/v03_sinusoid" WORLD=sinusoid
_run v0.4_sfmock "${REPORTS}/v0.4_studyforrest_aggregate.json" \
    OUT_ROOT="${OUT_BASE}/v04_sfmock" WORLD=studyforrest
_run v0.4_sf_real "${REPORTS}/v0.4_studyforrest_real_aggregate.json" \
    OUT_ROOT="${OUT_BASE}/v04_sf_real" WORLD=studyforrest \
    BOUBA_SENS_STUDYFORREST_DATA="${STUDYFORREST_DATA}"

# -- §5.2 cross-world lock matrix (ADR-0011) --------------------------
_run v0.4_recovery "${REPORTS}/v0.4_b1_recovery_aggregate.json" \
    OUT_ROOT="${OUT_BASE}/v04_gauss_lock" WORLD=gaussian LOCK_AFTER=200
_run v0.4_xor_lock "${REPORTS}/v0.4_b1_xor_lock_aggregate.json" \
    OUT_ROOT="${OUT_BASE}/v04_xor_lock" WORLD=xor LOCK_AFTER=200
_run v0.4_sinusoid_lock "${REPORTS}/v0.4_b1_sinusoid_lock_aggregate.json" \
    OUT_ROOT="${OUT_BASE}/v04_sinusoid_lock" WORLD=sinusoid LOCK_AFTER=200
_run v0.4_ecg_lock "${REPORTS}/v0.4_b1_ecg_lock_aggregate.json" \
    OUT_ROOT="${OUT_BASE}/v04_ecg_lock" WORLD=studyforrest LOCK_AFTER=200

# -- §5.5 Amedi dose-response (ADR-0013) ------------------------------
for L in 50 100 200 400 800; do
    _run v0.5_s10_dr "${REPORTS}/v0.5_dr_lock${L}_aggregate.json" \
        OUT_ROOT="${OUT_BASE}/v05_dr_lock${L}" WORLD=studyforrest \
        BOUBA_SENS_STUDYFORREST_DATA="${STUDYFORREST_DATA}" LOCK_AFTER="${L}"
done

# -- §5.6 compound lock + gumbel (ADR-0014) ---------------------------
_run v0.5_s11 "${REPORTS}/v0.5_s11_compound_aggregate.json" \
    OUT_ROOT="${OUT_BASE}/v05_s11_compound" WORLD=studyforrest \
    BOUBA_SENS_STUDYFORREST_DATA="${STUDYFORREST_DATA}" LOCK_AFTER=100 \
    TRANSDUCER_GATING=gumbel GUMBEL_TAU=1.0

# -- §5.7 gumbel tau scan (ADR-0015) ----------------------------------
for TAU in 0.1 0.3 0.5 2.0; do
    tag="${TAU/./_}"
    _run "v0.5_s12_tau${tag}" "${REPORTS}/v0.5_s12_tau${tag}_aggregate.json" \
        OUT_ROOT="${OUT_BASE}/v05_s12_tau${tag}" WORLD=studyforrest \
        BOUBA_SENS_STUDYFORREST_DATA="${STUDYFORREST_DATA}" LOCK_AFTER=100 \
        TRANSDUCER_GATING=gumbel GUMBEL_TAU="${TAU}"
done

# -- §5.8 finer tau + codebook freeze (ADR-0016) ----------------------
for TAU in 0.2 0.25 0.35 0.4; do
    tag="${TAU/./_}"
    _run "v0.5_s13_tau${tag}" "${REPORTS}/v0.5_s13_tau${tag}_aggregate.json" \
        OUT_ROOT="${OUT_BASE}/v05_s13_tau${tag}" WORLD=studyforrest \
        BOUBA_SENS_STUDYFORREST_DATA="${STUDYFORREST_DATA}" LOCK_AFTER=100 \
        TRANSDUCER_GATING=gumbel GUMBEL_TAU="${TAU}"
done
_run v0.5_s13_cbfreeze "${REPORTS}/v0.5_s13_cbfreeze_aggregate.json" \
    OUT_ROOT="${OUT_BASE}/v05_s13_cbfreeze" WORLD=studyforrest \
    BOUBA_SENS_STUDYFORREST_DATA="${STUDYFORREST_DATA}" LOCK_AFTER=100 \
    CODEBOOK_LOCK=100 TRANSDUCER_GATING=hard

# -- §5.9 phase-transition schedule (ADR-0017) ------------------------
_run v0.5_s14_joint "${REPORTS}/v0.5_s14c_joint_aggregate.json" \
    OUT_ROOT="${OUT_BASE}/v05_s14_joint" WORLD=studyforrest \
    BOUBA_SENS_STUDYFORREST_DATA="${STUDYFORREST_DATA}" LOCK_AFTER=100 \
    TRANSDUCER_GATING=hard GUMBEL_TAU=0.30 \
    GATING_SCHEDULE=200 GATING_TARGET=gumbel

# -- Per-seed re-aggregations (ADR-0017 §14a + §14c) ------------------
if [[ "${CHOOSE}" == "all" || "${CHOOSE}" == "per_seed" ]]; then
    uv run python scripts/aggregate_grid_per_seed.py \
        --root "${OUT_BASE}/v05_s12_tau0_3" \
        --out "${REPORTS}/v0.5_s14a_tau0_3_per_seed.json"
    uv run python scripts/aggregate_grid_per_seed.py \
        --root "${OUT_BASE}/v05_s14_joint" \
        --out "${REPORTS}/v0.5_s14c_joint_per_seed.json"
fi

# -- Sprint 17 Kraskov vs MINE (ADR-0017 §6.3) ------------------------
if [[ "${CHOOSE}" == "all" || "${CHOOSE}" == "sprint17" ]]; then
    uv run python scripts/sprint17_compare_kraskov_mine.py \
        --root "${OUT_BASE}/v05_dr_lock100" \
        --out "${REPORTS}/sprint17_kraskov_vs_mine.csv"
fi

# -- Final manifest ---------------------------------------------------
uv run python scripts/sha256_manifest.py \
    --root "${REPORTS}" --out "${REPORTS}/MANIFEST.json"

echo "reproduction complete; verify sha256 against reports/MANIFEST.json"
