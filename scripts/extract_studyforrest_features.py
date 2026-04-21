"""Sprint 9 / Task 9.2 — orchestrate the 5-modality extraction.

Reads the Studyforrest phase-2 fetched tree, runs the 5 extractors,
writes 6 pytorch tensors + a MANIFEST with SHA256s. Pass the output
dir to `StudyforrestRealWorld(data_dir=...)` (Task 9.3).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from bouba_sens.world.studyforrest_features import (
    TARGET_HZ,
    extract_audio,
    extract_force,
    extract_gravity,
    extract_labels,
    extract_tactile,
    extract_vision,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Studyforrest phase-2 fetched root (output of fetch_studyforrest_phase2.sh)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Target dir for the 6 tensors + MANIFEST.md",
    )
    parser.add_argument(
        "--subject",
        default="sub-01",
        help="BIDS subject label",
    )
    parser.add_argument(
        "--run",
        default="run-1",
        help="BIDS run label",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=900,
        help="Trim to first N seconds (default 15 min = 9000 frames @ 10 Hz)",
    )
    args = parser.parse_args()

    n_frames = args.duration_seconds * TARGET_HZ

    audio_src = args.src / "stimuli/soundtrack/fg_av_ger_stereo.mp3"
    video_src = args.src / "stimuli/movie/fg_av_ger_stereo.mkv"
    annot_src = args.src / "stimuli/annotations/movie_motion-locations.csv"
    bold_src = (
        args.src
        / args.subject
        / "ses-movie/func"
        / f"{args.subject}_ses-movie_task-movie_{args.run}_bold.nii.gz"
    )
    physio_src = (
        args.src
        / args.subject
        / "ses-movie/func"
        / f"{args.subject}_ses-movie_task-movie_{args.run}_physio.tsv.gz"
    )

    print(f"[extract] audio <- {audio_src}")
    audio = extract_audio(audio_src, n_frames)
    print(f"[extract] vision <- {video_src}")
    vision = extract_vision(video_src, n_frames)
    print(f"[extract] tactile <- {annot_src}")
    tactile = extract_tactile(annot_src, n_frames)
    print(f"[extract] gravity <- {bold_src}")
    gravity = extract_gravity(bold_src, n_frames)
    print(f"[extract] force <- {physio_src}")
    force = extract_force(physio_src, n_frames)
    print(f"[extract] labels <- {annot_src}")
    labels = extract_labels(annot_src, n_frames)

    args.out.mkdir(parents=True, exist_ok=True)
    tensors = {
        "audio.pt": audio,
        "vision.pt": vision,
        "tactile.pt": tactile,
        "gravity.pt": gravity,
        "force.pt": force,
        "labels.pt": labels,
    }
    for name, tensor in tensors.items():
        torch.save(tensor, args.out / name)

    lines = [
        "# Studyforrest phase-2 5-modality feature manifest",
        "",
        f"- Source: `{args.src}` (datalad `///studyforrest-data-phase2`)",
        f"- Subject: {args.subject}",
        f"- Run: {args.run}",
        f"- Frames: {n_frames} at {TARGET_HZ} Hz ({args.duration_seconds} s)",
        "",
        "## Tensor shapes",
        "",
    ]
    for name, tensor in tensors.items():
        lines.append(f"- `{name}`: shape {tuple(tensor.shape)}, dtype {tensor.dtype}")
    lines.append("")
    lines.append("## SHA256")
    lines.append("")
    for name in tensors:
        lines.append(f"- `{name}`: `{_sha256(args.out / name)}`")
    (args.out / "MANIFEST.md").write_text("\n".join(lines) + "\n")
    print(f"[extract] wrote {len(tensors)} tensors + MANIFEST to {args.out}")


if __name__ == "__main__":
    main()
