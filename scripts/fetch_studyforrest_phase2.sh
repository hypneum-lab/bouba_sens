#!/usr/bin/env bash
# Sprint 9 / Task 9.1 — fetch the Studyforrest phase-2 subset needed
# for the 5-modality bridge. Idempotent: re-runs do nothing when the
# target subtree already exists.
set -euo pipefail

DEST="${1:-data/studyforrest_phase2}"
SUBJECT="${SUBJECT:-sub-01}"
RUN="${RUN:-run-1}"

# 1. Install datalad + git-annex if missing (Homebrew on macOS).
if ! command -v datalad >/dev/null; then
    echo "[fetch] installing datalad via Homebrew"
    brew install datalad git-annex
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
    if [[ ! -e "$(realpath "${f}")" ]]; then
        echo "[fetch] datalad get (required) ${f}"
        datalad get "${f}"
    fi
done

for f in "${OPTIONAL[@]}"; do
    if [[ ! -e "$(realpath "${f}")" ]]; then
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
        h=$(shasum -a 256 "${f}" | awk '{print $1}')
        echo "- \`${f}\`: \`${h}\`" >> MANIFEST.md
    fi
done

echo "[fetch] done; manifest at ${DEST}/MANIFEST.md"
