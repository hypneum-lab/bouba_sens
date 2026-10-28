"""Fetch the ActionSense raw files needed by extract_actionsense.py.

The directory listing of https://data.csail.mit.edu/ActionNet/ answers 403,
so the per-file links are resolved from the project page
https://action-net.csail.mit.edu/data.html, which links every file as
``https://data.csail.mit.edu/ActionNet/{wearable_data,external_data}/
<date>_experiment_<S>/…``. Per subject and recording split we take:

  *_streamLog_actionNet-wearables_<S>.hdf5            (wearable streams)
  *_<S>_eye-tracking-video-world_frame.mp4            (world video, no gaze)
  *_streamLog_actionNet-microphone_<S>_sink_chunked_data.wav
  *_streamLog_actionNet-microphone_<S>.hdf5           (microphone timing)

Every file is HEAD-ed first. The run aborts if free disk space minus the bytes
still to download is below 20 GB. Downloads resume from partial ``.part``
files. ``MANIFEST.json`` maps ``<subject>/<file>`` to ``{url, bytes, sha256}``.
Data licence: CC BY-NC-SA 4.0 (ActionSense, DelPreto et al., NeurIPS 2022).

Usage:
    uv run python scripts/fetch_actionsense.py --subjects S00 --out data/actionsense_raw
    uv run python scripts/fetch_actionsense.py --out data/actionsense_raw --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path

PAGE_URL = "https://action-net.csail.mit.edu/data.html"
URL_RE = re.compile(r"https://data\.csail\.mit\.edu/ActionNet/[^\"'\s<>]+")
MIN_FREE_AFTER_BYTES = 20 * 10**9
UA = {"User-Agent": "Mozilla/5.0 (bouba_sens fetch_actionsense)"}
DEFAULT_SUBJECTS = ["S00", "S02", "S03", "S04", "S05"]


def wanted_patterns(subject: str) -> tuple[str, ...]:
    return (
        f"_streamLog_actionNet-wearables_{subject}.hdf5",
        f"_{subject}_eye-tracking-video-world_frame.mp4",
        f"_streamLog_actionNet-microphone_{subject}_sink_chunked_data.wav",
        f"_streamLog_actionNet-microphone_{subject}.hdf5",
    )


def resolve_urls(page_html: str, subjects: Sequence[str]) -> dict[str, list[str]]:
    """Per-subject sorted, de-duplicated list of the URLs to fetch."""
    all_urls = sorted(set(URL_RE.findall(page_html)))
    out: dict[str, list[str]] = {}
    for sid in subjects:
        pats = wanted_patterns(sid)
        out[sid] = [u for u in all_urls if f"_experiment_{sid}/" in u and u.endswith(pats)]
    return out


def _head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Length"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _download(url: str, dest: Path, size: int) -> None:
    """Download with resume into ``dest.part``, then rename."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    if have < size:
        headers = dict(UA)
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            if have and resp.status != 206:
                have = 0  # server ignored Range: restart
            with part.open("ab" if have else "wb") as fh:
                shutil.copyfileobj(resp, fh, 1 << 20)
    if part.stat().st_size != size:
        raise OSError(f"{dest.name}: got {part.stat().st_size} bytes, expected {size}")
    part.rename(dest)


def verify(out: Path) -> int:
    man_path = out / "MANIFEST.json"
    manifest: dict[str, dict[str, object]] = json.loads(man_path.read_text())
    bad = 0
    for rel, entry in sorted(manifest.items()):
        p = out / rel
        if not p.exists():
            print(f"MISSING {rel}", file=sys.stderr)
            bad += 1
        elif p.stat().st_size != entry["bytes"] or _sha256(p) != entry["sha256"]:
            print(f"MISMATCH {rel}", file=sys.stderr)
            bad += 1
        else:
            print(f"ok {rel}", flush=True)
    return 1 if bad else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--out", type=Path, default=Path("data/actionsense_raw"))
    ap.add_argument("--verify", action="store_true", help="re-hash against MANIFEST.json")
    ap.add_argument("--dry-run", action="store_true", help="HEAD every file, download nothing")
    args = ap.parse_args(argv)

    if args.verify:
        return verify(args.out)

    req = urllib.request.Request(PAGE_URL, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        page = resp.read().decode("utf-8", errors="replace")
    urls = resolve_urls(page, args.subjects)
    sizes: dict[str, int] = {}
    for sid, lst in urls.items():
        if not lst:
            print(f"ABORT: no links found for {sid}", file=sys.stderr)
            return 1
        for u in lst:
            sizes[u] = _head_size(u)
            print(f"HEAD {sizes[u] / 1e6:10.1f} MB  {u.rsplit('/', 1)[-1]}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    todo = 0
    for sid, lst in urls.items():
        for u in lst:
            dest = args.out / sid / u.rsplit("/", 1)[-1]
            if not dest.exists():
                part = dest.with_name(dest.name + ".part")
                todo += sizes[u] - (part.stat().st_size if part.exists() else 0)
    free = shutil.disk_usage(args.out).free
    print(f"to download {todo / 1e9:.1f} GB, free {free / 1e9:.1f} GB", flush=True)
    if free - todo < MIN_FREE_AFTER_BYTES:
        print("ABORT: free disk after download would be below 20 GB", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0

    man_path = args.out / "MANIFEST.json"
    manifest: dict[str, dict[str, object]] = (
        json.loads(man_path.read_text()) if man_path.exists() else {}
    )
    for sid, lst in urls.items():
        (args.out / sid).mkdir(exist_ok=True)
        for u in lst:
            rel = f"{sid}/{u.rsplit('/', 1)[-1]}"
            dest = args.out / rel
            if not dest.exists():
                print(f"GET  {rel}", flush=True)
                _download(u, dest, sizes[u])
            entry = manifest.get(rel)
            if not entry or entry["bytes"] != sizes[u]:
                entry = {"url": u, "bytes": sizes[u], "sha256": _sha256(dest)}
            manifest[rel] = entry
            man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {man_path} ({len(manifest)} files)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
