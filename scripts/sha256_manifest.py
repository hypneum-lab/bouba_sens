"""Sprint 16 Task 6 — SHA256 manifest for every `reports/` artefact
cited in the paper.

Emits `reports/MANIFEST.json` listing sha256 and size for each
`.json`, `.csv`, `.png` under `reports/`. Appendix C of the paper
references this file so a reviewer can verify byte-identity between
the released repository and any re-run.

Usage::

    # Write (re)generate manifest
    uv run python scripts/sha256_manifest.py --root reports \
        --out reports/MANIFEST.json

    # Check existing manifest against the tree (read-only, exits 1 on diff)
    uv run python scripts/sha256_manifest.py --check \
        --root reports --out reports/MANIFEST.json

Deterministic: entries are sorted by relative path. Rerunning on the
same tree produces byte-identical output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256(path: Path, *, buf_size: int = 65_536) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(buf_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path) -> dict[str, dict[str, int | str]]:
    """Collect a mapping `relative_path -> {size, sha256}` under `root`."""
    entries: dict[str, dict[str, int | str]] = {}
    for ext in ("*.json", "*.csv", "*.png"):
        for path in sorted(root.rglob(ext)):
            if path.name == "MANIFEST.json":
                continue
            rel = path.relative_to(root).as_posix()
            entries[rel] = {"size": path.stat().st_size, "sha256": _sha256(path)}
    return dict(sorted(entries.items()))


def check_manifest(root: Path, out: Path) -> int:
    """Compare the tree under `root` with the existing manifest at `out`.

    Prints added / removed / modified paths and returns 1 if any
    difference is found, 0 otherwise. Never writes to disk.
    """
    if not out.exists():
        print(f"[MISSING] manifest file not found: {out}", file=sys.stderr)
        return 1

    committed: dict[str, dict[str, int | str]] = json.loads(out.read_text())
    current = build_manifest(root)

    committed_keys = set(committed)
    current_keys = set(current)

    added = sorted(current_keys - committed_keys)
    removed = sorted(committed_keys - current_keys)
    modified = sorted(
        k
        for k in committed_keys & current_keys
        if committed[k]["sha256"] != current[k]["sha256"]
        or committed[k]["size"] != current[k]["size"]
    )

    if added:
        for p in added:
            print(f"[ADDED]    {p}")
    if removed:
        for p in removed:
            print(f"[REMOVED]  {p}")
    if modified:
        for p in modified:
            print(f"[MODIFIED] {p}")

    if added or removed or modified:
        total = len(added) + len(removed) + len(modified)
        print(f"\nmanifest mismatch: {total} difference(s)")
        return 1

    print(f"manifest OK: {len(current)} artefact(s) match {out}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("reports"))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Destination path for the generated manifest (required unless"
            " --check is given). No default: the script never overwrites"
            " reports/MANIFEST.json without an explicit --out."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Read-only: compare the tree against the existing manifest and"
            " exit 1 on any difference. Never writes to disk."
        ),
    )
    args = parser.parse_args()

    if args.check:
        out = args.out if args.out is not None else Path("reports/MANIFEST.json")
        sys.exit(check_manifest(args.root, out))

    if args.out is None:
        parser.error(
            "--out is required when writing. "
            "Example: --out reports/MANIFEST.json\n"
            "Use --check to verify an existing manifest without writing."
        )

    manifest = build_manifest(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest: {len(manifest)} artefact(s) written to {args.out}")


if __name__ == "__main__":
    main()
