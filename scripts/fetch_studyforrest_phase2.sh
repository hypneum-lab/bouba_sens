#!/usr/bin/env bash
# Sprint 9 / Task 9.1 — fetch the Studyforrest phase-2 subset needed
# for the 5-modality bridge. Idempotent: re-runs do nothing when the
# target subtree already exists.
set -euo pipefail

DEST="${1:-data/studyforrest_phase2}"
SUBJECT="${SUBJECT:-sub-01}"
RUN="${RUN:-run-1}"

# 1. Install datalad + git-annex if missing (Homebrew on macOS only).
if ! command -v datalad >/dev/null; then
    if command -v brew >/dev/null; then
        echo "[fetch] installing datalad via Homebrew"
        brew install datalad git-annex
    else
        echo "[fetch] ERROR: datalad not found and brew unavailable." >&2
        echo "[fetch]        Install datalad manually (https://www.datalad.org/)" >&2
        echo "[fetch]        then re-run this script." >&2
        exit 1
    fi
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
    echo "[fetch] datalad install ///studyforrest-data-phase2 -> ${DEST}"
    datalad install -r "///studyforrest-data-phase2" "${DEST}"
fi

cd "${DEST}"

# 3. Get only what we need. File paths follow BIDS conventions.
#    OPTIONAL items (e.g. video.mkv) may not be resolvable via datalad
#    on all dataset snapshots; failures are non-fatal.
REQUIRED=(
    "stimuli/soundtrack/fg_av_ger_stereo.mp3"
    "${SUBJECT}/ses-movie/func/${SUBJECT}_ses-movie_task-movie_${RUN}_bold.nii.gz"
    "${SUBJECT}/ses-movie/func/${SUBJECT}_ses-movie_task-movie_${RUN}_physio.tsv.gz"
    "stimuli/annotations/movie_motion-locations.csv"
)
OPTIONAL=(
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

- Source: \`///studyforrest-data-phase2\` (Hanke et al. 2014, Scientific Data)
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
