"""Stage downloaded archives as Kaggle dataset folders (private mirrors for Kaggle GPU phases).

Hard-links archives from <data_root>/_archives into <data_root>/_kaggle/<slug>/ (no copy on the same drive)
and writes dataset-metadata.json. Upload afterwards with:
    kaggle datasets create -p <data_root>/_kaggle/<slug> -r skip
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

from perception.config import data_root, load_config
from perception.data.download import parse_manifest

DATASETS = {
    "tum-rgbd-p5": {
        "title": "TUM RGB-D subset (Project 5 mirror, private)",
        "keys": ["tum_fr1_desk", "tum_fr3_long_office"],
        "license": "CC-BY-4.0",
        "source": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset (Sturm et al., IROS 2012). CC BY 4.0.",
    },
    "sunrgbd-p5": {
        "title": "SUN RGB-D (Project 5 mirror, private)",
        "keys": ["sunrgbd_meta3d", "sunrgbd_meta2d", "sunrgbd_toolbox", "sunrgbd_images"],
        "license": "other",
        "source": "https://rgbd.cs.princeton.edu/ (Song et al., CVPR 2015). Private mirror for personal research only.",
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="Kaggle username")
    args = ap.parse_args()

    cfg = load_config("dataset")
    root = data_root(cfg)
    entries = {e.key: e for e in parse_manifest(cfg)}
    for slug, spec in DATASETS.items():
        out = root / "_kaggle" / slug
        out.mkdir(parents=True, exist_ok=True)
        for key in spec["keys"]:
            src = root / "_archives" / entries[key].filename
            dst = out / src.name
            if not src.exists():
                print(f"[missing] {src}")
                continue
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            print(f"[staged] {slug}/{src.name}")
        # Same field set as previously accepted uploads on this account. The CLI creates datasets
        # private by default (only `--public` makes them public); `subtitle`/`isPrivate` led to a 400.
        meta = {
            "title": spec["title"],
            "id": f"{args.user}/{slug}",
            "licenses": [{"name": spec["license"]}],
            "description": f"Source: {spec['source']} Raw archives, unmodified. SHA-256 of each archive is recorded in the Project 5 data lock file.",
        }
        (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
