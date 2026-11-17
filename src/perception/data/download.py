"""Manifest-driven, resumable dataset download and safe extraction (spec 01.1)."""

from __future__ import annotations

import hashlib
import json
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

CHUNK = 1 << 20  # 1 MiB
LOCK_NAME = "_manifest_lock.json"


@dataclass(frozen=True)
class ManifestEntry:
    key: str
    url: str
    size_mb: float
    dest: str
    extract: bool

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


def parse_manifest(cfg: dict[str, Any]) -> list[ManifestEntry]:
    return [
        ManifestEntry(key=k, url=v["url"], size_mb=float(v["size_mb"]), dest=v["dest"], extract=bool(v["extract"]))
        for k, v in cfg["manifest"].items()
    ]


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        n = r.headers.get("Content-Length")
        return int(n) if n is not None else None


def download(entry: ManifestEntry, archive_dir: Path, log=print) -> Path:
    """Download one entry into archive_dir, resuming a partial file if present."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    out = archive_dir / entry.filename
    part = out.with_suffix(out.suffix + ".part")
    total = _remote_size(entry.url)

    if out.exists() and (total is None or out.stat().st_size == total):
        log(f"[skip] {entry.filename} already complete")
        return out

    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    req = urllib.request.Request(entry.url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        if start:
            # Never truncate a partial file: if the server ignored Range, fail so the caller retries later.
            content_range = r.headers.get("Content-Range", "")
            if r.status != 206 or not content_range.startswith(f"bytes {start}-"):
                raise IOError(f"{entry.filename}: server ignored Range (status {r.status}); keeping {start} bytes")
        mode = "ab" if start else "wb"
        done, t0, last = start, time.time(), 0.0
        with open(part, mode) as f:
            while chunk := r.read(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 5:
                    last = time.time()
                    pct = f"{100 * done / total:5.1f}%" if total else ""
                    rate = (done - start) / max(last - t0, 1e-6) / 1e6
                    log(f"  {entry.filename}: {done / 1e6:,.0f} MB {pct} ({rate:.1f} MB/s)")
    if total is not None and part.stat().st_size != total:
        raise IOError(f"{entry.filename}: size {part.stat().st_size} != expected {total}")
    part.replace(out)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _is_safe_member(name: str) -> bool:
    p = PurePosixPath(name.replace("\\", "/"))
    return not p.is_absolute() and ".." not in p.parts and not (p.parts and p.parts[0].endswith(":"))


def safe_extract(archive: Path, dest: Path) -> None:
    """Extract a .tgz/.tar/.zip, refusing members that would escape dest."""
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith((".tgz", ".tar.gz", ".tar")):
        with tarfile.open(archive) as tf:
            bad = [m.name for m in tf.getmembers() if not _is_safe_member(m.name) or m.issym() or m.islnk()]
            if bad:
                raise ValueError(f"unsafe members in {archive.name}: {bad[:5]}")
            tf.extractall(dest, filter="data")
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            bad = [n for n in zf.namelist() if not _is_safe_member(n)]
            if bad:
                raise ValueError(f"unsafe members in {archive.name}: {bad[:5]}")
            zf.extractall(dest)
    else:
        raise ValueError(f"unsupported archive type: {archive.name}")


def update_lock(root: Path, entry: ManifestEntry, path: Path) -> None:
    lock_path = root / LOCK_NAME
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    lock[entry.key] = {
        "url": entry.url,
        "file": str(path.relative_to(root)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True))
