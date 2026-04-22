"""Sprint 16 Task 6 — SHA256 manifest for every `reports/` artefact
cited in the paper.

Emits `reports/MANIFEST.json` listing sha256, size, and mtime for
each `.json`, `.csv`, `.png` under `reports/`. Appendix C of the
paper references this file so a reviewer can verify byte-identity
between the released repository and any re-run.

Usage::

    uv run python scripts/sha256_manifest.py --root reports \
        --out reports/MANIFEST.json

Deterministic: entries are sorted by relative path. Rerunning on the
same tree produces byte-identical output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("reports"))
    parser.add_argument("--out", type=Path, default=Path("reports/MANIFEST.json"))
    args = parser.parse_args()

    manifest = build_manifest(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest: {len(manifest)} artefact(s) -> {args.out}")


if __name__ == "__main__":
    main()
