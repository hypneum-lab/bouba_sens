#!/usr/bin/env bash
# Sprint 9 wrapper — run_grid.sh + critical_validation_pipeline.sh chained.
#
# Usage :
#   scripts/run_grid_with_partition_control.sh
#
# Or via env vars (matching run_grid.sh + a new TAG/WORLD pair) :
#   OUT_ROOT=runs/v0.5_studyforrest_5modal \
#     TAG=v0.5-studyforrest-5modal \
#     WORLD=studyforrest \
#     scripts/run_grid_with_partition_control.sh
#
# Why this exists : ADR-0014 (2026-04-24) closed issue #4 with a 3/3
# partition-control failure across cluster (XOR), mock (AR(1)), and
# real (ECG) grids. ADR-0012 acceptance gate now requires that any
# grid run report partition control alongside its raw aggregates.
# Forgetting to chain the two scripts at launch time was the failure
# mode that motivated this wrapper.
#
# Behaviour :
#   1. Runs scripts/run_grid.sh with the env vars passed in.
#   2. If the grid run exits 0 AND scripts/critical_validation_pipeline.sh
#      exists in the working tree (it lives on sprint9/critical-pipeline
#      branch as of 2026-04-24), runs it on the freshly produced grid.
#   3. Surfaces both exit codes ; partition-control failure does NOT
#      mask grid-run success.
#
# Exit codes :
#   0 = grid + partition control both succeeded
#   2 = grid run failed (partition control skipped)
#   3 = grid run succeeded, partition control script not found
#   4 = grid run succeeded, partition control failed
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-./runs/v0.5_grid}"
TAG="${TAG:-$(basename "${OUT_ROOT}" | sed 's/^runs_//')}"
WORLD="${WORLD:-gaussian}"

PIPELINE_SCRIPT="scripts/critical_validation_pipeline.sh"

echo "[wrapper] phase 1/2 — grid run via run_grid.sh"
echo "[wrapper]   OUT_ROOT=${OUT_ROOT}"
echo "[wrapper]   WORLD=${WORLD}"
echo "[wrapper]   TAG=${TAG}"

# Forward all relevant env vars to run_grid.sh. We pass them explicitly
# rather than relying on inheritance so a typo in the parent shell
# environment becomes a visible default rather than a silent override.
# `set -e` would abort on grid failure before we can capture $? and
# emit our custom exit code 2. Wrap the call in `if !` so failure is
# observable and routable.
if ! OUT_ROOT="${OUT_ROOT}" \
     WORLD="${WORLD}" \
     STEPS_TRAIN="${STEPS_TRAIN:-200}" \
     STEPS_LESION="${STEPS_LESION:-100}" \
     METRICS="${METRICS:-Me1,Me2,Me3}" \
     LOCK_AFTER="${LOCK_AFTER:-}" \
     TRANSDUCER_GATING="${TRANSDUCER_GATING:-hard}" \
     GUMBEL_TAU="${GUMBEL_TAU:-1.0}" \
     CODEBOOK_LOCK="${CODEBOOK_LOCK:-}" \
     GATING_SCHEDULE="${GATING_SCHEDULE:-}" \
     GATING_TARGET="${GATING_TARGET:-}" \
        bash scripts/run_grid.sh; then
    echo "[wrapper] grid run failed — skipping partition control" >&2
    exit 2
fi

echo "[wrapper] phase 2/2 — partition control via ${PIPELINE_SCRIPT}"

if [[ ! -f "${PIPELINE_SCRIPT}" ]]; then
    echo "[wrapper] WARN ${PIPELINE_SCRIPT} not found in working tree." >&2
    echo "[wrapper] WARN The pipeline lives on sprint9/critical-pipeline branch" >&2
    echo "[wrapper] WARN (commit 858ce51 as of 2026-04-24). Either checkout that" >&2
    echo "[wrapper] WARN branch or copy the script in. Grid run was OK ; partition" >&2
    echo "[wrapper] WARN control was the gate that failed." >&2
    exit 3
fi

# Same pattern for the partition-control pipeline.
if ! bash "${PIPELINE_SCRIPT}" \
        --grid-root "${OUT_ROOT}" \
        --tag       "${TAG}" \
        --world     "${WORLD}"; then
    echo "[wrapper] partition-control pipeline failed" >&2
    exit 4
fi

echo "[wrapper] DONE — grid + partition control both succeeded."
echo "[wrapper] Read the verdict in :"
echo "[wrapper]   reports/${TAG}_critical_validation/null_b3_analysis.json"
echo "[wrapper] If passes_95pct=false, ADR-0014 dichotomous framework applies."
