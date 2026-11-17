"""Download and extract approved datasets (spec 01.1).

Examples:
    python scripts/download_data.py --dry-run
    python scripts/download_data.py tum_fr1_desk
    python scripts/download_data.py sunrgbd_meta3d sunrgbd_meta2d sunrgbd_toolbox sunrgbd_images
"""

from __future__ import annotations

import argparse
import sys

from perception.config import data_root, load_config
from perception.data.download import download, parse_manifest, safe_extract, update_lock


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="*", help="manifest keys to fetch (default: none; use --all)")
    ap.add_argument("--all", action="store_true", help="fetch every manifest entry")
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched, download nothing")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config("dataset")
    root = data_root(cfg)
    entries = {e.key: e for e in parse_manifest(cfg)}

    unknown = [k for k in args.keys if k not in entries]
    if unknown:
        print(f"unknown keys: {unknown}. Known: {sorted(entries)}", file=sys.stderr)
        return 2
    selected = list(entries.values()) if args.all or args.dry_run and not args.keys else [entries[k] for k in args.keys]

    if args.dry_run:
        for e in selected:
            print(f"{e.key:22s} {e.size_mb:>8,.0f} MB  -> {root / e.dest}  ({e.url})")
        print(f"{'TOTAL':22s} {sum(e.size_mb for e in selected):>8,.0f} MB")
        return 0
    if not selected:
        ap.print_help()
        return 2

    for e in selected:
        print(f"[get] {e.key} ({e.size_mb:,.0f} MB)")
        path = download(e, root / "_archives")
        update_lock(root, e, path)
        if e.extract and not args.no_extract:
            print(f"[extract] {path.name} -> {root / e.dest}")
            safe_extract(path, root / e.dest)
        elif not e.extract:
            target = root / e.dest / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(path.read_bytes())
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
