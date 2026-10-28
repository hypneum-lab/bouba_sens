#!/usr/bin/env bash
# Sprint 9 / Task 9.1 — fetch the Studyforrest phase-2 subset needed
# for the 5-modality bridge. Idempotent: re-runs do nothing when the
# target subtree already exists.
#
# Source: the `///studyforrest-data-phase2` handle no longer resolves on
# datasets.datalad.org (checked 2026-10-25, issue #38); the dataset is
# installed from its GitHub mirror instead (override with
# STUDYFORREST_PHASE2_URL).
#
# REQUIRED files are exactly the two that scripts/extract_phase2_adapted.py
# reads. The ses-movie files of the original Sprint 9 plan are kept as
# OPTIONAL (several no longer exist in phase-2, see ADR-0012).
#
# Note (issue #45): the public movie_localizer.mkv lasts 12 s, so the
# extraction cannot produce the 1800-frame vision tensor used in April
# 2026; extract_phase2_adapted.py now stops with an explicit error.
set -euo pipefail

DEST="${1:-data/studyforrest_phase2}"
SUBJECT="${SUBJECT:-sub-01}"
RUN="${RUN:-run-1}"
SOURCE_URL="${STUDYFORREST_PHASE2_URL:-https://github.com/psychoinformatics-de/studyforrest-data-phase2.git}"

# 1. Require datalad + git-annex. Nothing is installed automatically:
#    every third-party tool goes through human review first (lab
#    supply-chain policy), so a missing tool is a hard stop.
missing=()
for tool in datalad git-annex; do
    command -v "${tool}" >/dev/null || missing+=("${tool}")
done
if (( ${#missing[@]} > 0 )); then
    echo "[fetch] ERROR: missing tool(s): ${missing[*]}" >&2
    echo "[fetch]        This script never installs software. Install a" >&2
    echo "[fetch]        reviewed, pinned version yourself, for example:" >&2
    echo "[fetch]          macOS : brew install datalad git-annex" >&2
    echo "[fetch]          Debian/Ubuntu : apt install datalad git-annex" >&2
    echo "[fetch]          pip : pip install datalad (git-annex separately)" >&2
    echo "[fetch]        See https://handbook.datalad.org/en/latest/intro/installation.html" >&2
    echo "[fetch]        then re-run this script." >&2
    exit 1
fi

# Detect the SHA256 tool early so MANIFEST emission cannot silently fail.
if command -v shasum >/dev/null; then
    SHA_CMD=(shasum -a 256)
elif command -v sha256sum >/dev/null; then
    SHA_CMD=(sha256sum)
else
    echo "[fetch] ERROR: neither shasum nor sha256sum available" >&2
    exit 1
fi

# 2. Install the dataset handle (no payload yet).
if [[ ! -d "${DEST}" ]]; then
    echo "[fetch] datalad install -s ${SOURCE_URL} -> ${DEST}"
    datalad install -s "${SOURCE_URL}" "${DEST}"
fi

cd "${DEST}"

# 3. Get only what we need. File paths follow BIDS conventions.
#    REQUIRED = the inputs of scripts/extract_phase2_adapted.py.
#    OPTIONAL items may not be resolvable on every dataset snapshot;
#    failures are non-fatal.
REQUIRED=(
    "code/stimulus/movie_localizer/videos/movie_localizer.mkv"
    "${SUBJECT}/ses-localizer/func/${SUBJECT}_ses-localizer_task-movielocalizer_${RUN}_recording-cardresp_physio.tsv.gz"
)
OPTIONAL=(
    "stimuli/soundtrack/fg_av_ger_stereo.mp3"
    "${SUBJECT}/ses-movie/func/${SUBJECT}_ses-movie_task-movie_${RUN}_bold.nii.gz"
    "${SUBJECT}/ses-movie/func/${SUBJECT}_ses-movie_task-movie_${RUN}_physio.tsv.gz"
    "stimuli/annotations/movie_motion-locations.csv"
    "stimuli/movie/fg_av_ger_stereo.mkv"
    "${SUBJECT}/ses-movie/func/${SUBJECT}_ses-movie_task-movie_${RUN}_recording-eyegaze_physio.tsv.gz"
)

for f in "${REQUIRED[@]}"; do
    if [[ ! -e "${f}" ]]; then
        echo "[fetch] datalad get (required) ${f}"
        attempt=0
        until datalad get "${f}"; do
            attempt=$(( attempt + 1 ))
            if (( attempt >= 3 )); then
                echo "[fetch] FATAL: failed to fetch required ${f} after 3 attempts" >&2
                exit 1
            fi
            echo "[fetch] retry ${attempt}/3 for ${f}"
            sleep 5
        done
    fi
done

for f in "${OPTIONAL[@]}"; do
    if [[ ! -e "${f}" ]]; then
        echo "[fetch] datalad get (optional) ${f}"
        datalad get "${f}" || echo "[fetch] WARN: could not fetch ${f}, skipping"
    fi
done

# 4. Emit MANIFEST with SHA256s for reproducibility.
cat > MANIFEST.md <<EOF
# Studyforrest phase-2 fetch manifest

- Source: \`${SOURCE_URL}\` (Hanke et al. 2014, Scientific Data)
- Subject: ${SUBJECT}
- Run: ${RUN}
- Fetched: $(date -u +%Y-%m-%dT%H:%M:%SZ)

## SHA256
EOF
for f in "${REQUIRED[@]}" "${OPTIONAL[@]}"; do
    if [[ -e "${f}" ]]; then
        h=$("${SHA_CMD[@]}" "${f}" | awk '{print $1}')
        echo "- \`${f}\`: \`${h}\`" >> MANIFEST.md
    fi
done

# Count how many of each category actually landed on disk.
req_ok=0
for f in "${REQUIRED[@]}"; do
    [[ -e "${f}" ]] && req_ok=$(( req_ok + 1 ))
done
opt_ok=0
for f in "${OPTIONAL[@]}"; do
    [[ -e "${f}" ]] && opt_ok=$(( opt_ok + 1 ))
done
echo "[fetch] done: ${req_ok}/${#REQUIRED[@]} required, ${opt_ok}/${#OPTIONAL[@]} optional fetched"
echo "[fetch] manifest at ${DEST}/MANIFEST.md"
