#!/usr/bin/env bash
# ActionSense S00 pilot, the go/no-go gate (spec 2026-10-28, section 7).
#
# Usage: ./scripts/pilot_actionsense.sh
#   RAW=data/actionsense_raw   raw files (fetch_actionsense.py --subjects S00)
#   WORK=runs/pilot_actionsense  out_a/, out_b/, smoke runs, summary
#   FIG=docs/pilot/actionsense-s00-xcorr.png
#   SKIP_EXTRACT=1             reuse an existing out_a/ and out_b/
#
# Check 1: extraction does not abort (check_aligned, #45).
# Check 2: read from out_a/MANIFEST.md (spans, backwards steps, drops).
# Check 3: scripts/pilot_actionsense_xcorr.py on out_a (slicing frames).
# Check 4: sha256 of out_a/*.pt == out_b/*.pt.
# Check 5: train 20 / lesion T2 20 / eval, twice; four sha256 of #35.
# Exit code: 0 when checks 1, 3, 4 and 5 pass, 1 otherwise.
set -euo pipefail

RAW="${RAW:-data/actionsense_raw}"
WORK="${WORK:-runs/pilot_actionsense}"
FIG="${FIG:-docs/pilot/actionsense-s00-xcorr.png}"
SUMMARY="${WORK}/summary.txt"

for tool in uv shasum ffmpeg; do
    command -v "${tool}" >/dev/null || { echo "missing tool: ${tool}" >&2; exit 2; }
done
mkdir -p "${WORK}"
: >"${SUMMARY}"
say() { echo "$*" | tee -a "${SUMMARY}"; }
fail=0

# ---- Checks 1 and 4: two extractions -----------------------------------
for tag in a b; do
    out="${WORK}/out_${tag}"
    if [[ "${SKIP_EXTRACT:-0}" == "1" && -f "${out}/MANIFEST.md" ]]; then
        say "extract ${tag}: reused ${out}"
        continue
    fi
    rm -rf "${out}"
    t0=$(date +%s)
    if uv run python scripts/extract_actionsense.py --raw "${RAW}" --out "${out}" \
        --subjects S00 >"${WORK}/extract_${tag}.log" 2>&1; then
        say "extract ${tag}: ok in $(( $(date +%s) - t0 )) s"
    else
        say "extract ${tag}: ABORTED (exit $?) after $(( $(date +%s) - t0 )) s"
        say "check 1: FAIL"
        exit 1
    fi
done
say "check 1: PASS (no abort)"

hashes() { (cd "$1" && shasum -a 256 ./*.pt); }
hashes "${WORK}/out_a" >"${WORK}/sha_out_a.txt"
hashes "${WORK}/out_b" >"${WORK}/sha_out_b.txt"
cat "${WORK}/sha_out_a.txt" >>"${SUMMARY}"
if diff -q "${WORK}/sha_out_a.txt" "${WORK}/sha_out_b.txt" >/dev/null; then
    say "check 4: PASS ($(wc -l <"${WORK}/sha_out_a.txt" | tr -d ' ') files identical)"
else
    say "check 4: FAIL"
    diff "${WORK}/sha_out_a.txt" "${WORK}/sha_out_b.txt" | tee -a "${SUMMARY}" || true
    fail=1
fi

# ---- Check 3: onset cross-correlation on slicing frames ----------------
if uv run python scripts/pilot_actionsense_xcorr.py --data "${WORK}/out_a" \
    --fig "${FIG}" --json "${WORK}/xcorr.json" >"${WORK}/xcorr.log" 2>&1; then
    say "check 3: PASS ($(grep '"peak_lag_ms"' "${WORK}/xcorr.json" | tr -d ' ,'))"
else
    say "check 3: FAIL (see ${WORK}/xcorr.log)"
    fail=1
fi

# ---- Check 5: smoke cell twice on ActionSenseWorld ---------------------
export BOUBA_SENS_ACTIONSENSE_DATA="${WORK}/out_a"
cell() {
    local root="$1"
    rm -rf "${root}"
    uv run bouba-sens train --world actionsense --steps 20 --batch-size 16 \
        --seed 0 --out "${root}/phase1"
    uv run bouba-sens lesion --ckpt "${root}/phase1" --world actionsense \
        --modality audio --timing T2 --steps 20 --seed 10000 \
        --out "${root}/phase2_t2"
    uv run bouba-sens eval --run "${root}/phase2_t2" --metrics Me1,Me2 \
        --out "${root}/phase2_t2/eval_report.json"
}
SMOKE_FILES=(phase1/model.pt phase2_t2/model.pt phase2_t2/per_query_me1.json
    phase2_t2/eval_report.json)
ok5=1
for tag in a b; do
    if ! cell "${WORK}/smoke_${tag}" >"${WORK}/smoke_${tag}.log" 2>&1; then
        say "smoke ${tag}: command failed (see ${WORK}/smoke_${tag}.log)"
        ok5=0
    fi
done
if [[ "${ok5}" == "1" ]]; then
    for rel in "${SMOKE_FILES[@]}"; do
        ha=$(shasum -a 256 "${WORK}/smoke_a/${rel}" | cut -d' ' -f1)
        hb=$(shasum -a 256 "${WORK}/smoke_b/${rel}" | cut -d' ' -f1)
        say "smoke ${rel}: a=${ha} b=${hb}"
        [[ "${ha}" == "${hb}" ]] || ok5=0
    done
fi
if [[ "${ok5}" == "1" ]]; then say "check 5: PASS"; else say "check 5: FAIL"; fail=1; fi

say "check 2: read ${WORK}/out_a/MANIFEST.md (stream spans and clock checks)"
exit "${fail}"
