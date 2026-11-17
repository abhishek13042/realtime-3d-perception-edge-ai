"""Parse SUN RGB-D annotations, write the splits and class-frequency tables (spec 01.3, T2/T4, AC3/AC4).

Outputs:
  <data_root>/sunrgbd/_cache/annotations_v1.json        (parsed annotations, outside Git)
  configs/splits/sunrgbd_{train,val,test}.txt          (image ids, one per line)
  benchmarks/data_quality/sunrgbd_annotations.json      (counts + class frequencies)

Usage:  python -I scripts/build_sun_annotations.py [--val-frac 0.1] [--seed 0] [--rebuild]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perception.config import data_root, load_config  # noqa: E402
from perception.data.splits import carve_val, check_no_overlap, load_official, write_split  # noqa: E402
from perception.data.sunrgbd_annotations import load_annotations  # noqa: E402

# The 10 classes used by the common SUN RGB-D 3D detection benchmark (VoteNet et al.)
BENCHMARK_10 = ["bed", "table", "sofa", "chair", "toilet", "desk", "dresser", "night_stand", "bookshelf", "bathtub"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    sun = data_root(load_config("dataset")) / "sunrgbd"
    anns = load_annotations(sun, rebuild=args.rebuild)
    by_seq = {a.sequence.rstrip("/"): a for a in anns}
    groups = {a.image_id: a.sequence for a in anns}

    tr_seq, te_seq = load_official(sun / "SUNRGBDtoolbox" / "traintestSUNRGBD" / "allsplit.mat")
    missing = sorted((tr_seq | te_seq) - set(by_seq))
    if missing:
        raise SystemExit(f"{len(missing)} split entries not in metadata, e.g. {missing[:3]}")
    unassigned = set(by_seq) - tr_seq - te_seq
    official_train = sorted(by_seq[s].image_id for s in tr_seq)
    test = sorted(by_seq[s].image_id for s in te_seq)
    train, val = carve_val(official_train, groups, args.val_frac, args.seed)
    splits = {"train": train, "val": val, "test": test}
    check_no_overlap(splits, groups)

    out_dir = REPO / "configs" / "splits"
    for name, ids in splits.items():
        write_split(out_dir / f"sunrgbd_{name}.txt", ids)

    by_id = {a.image_id: a for a in anns}
    report = {
        "source": "SUNRGBDMeta3DBB_v2.mat, SUNRGBDMeta2DBB_v2.mat, allsplit.mat",
        "val": {"frac": args.val_frac, "seed": args.seed, "carved_from": "official alltrain", "group": "sequenceName"},
        "n_images": len(anns),
        "official": {"train": len(tr_seq), "test": len(te_seq), "unassigned": len(unassigned)},
        "splits": {},
    }
    for name, ids in splits.items():
        c3, c2, img_with = Counter(), Counter(), 0
        for i in ids:
            a = by_id[i]
            c3.update(b.cls for b in a.boxes3d)
            c2.update(b.cls for b in a.boxes2d)
            img_with += bool(a.boxes3d)
        report["splits"][name] = {
            "images": len(ids),
            "images_with_3d_boxes": img_with,
            "boxes3d": sum(c3.values()),
            "boxes2d": sum(c2.values()),
            "n_classes3d": len(c3),
            "benchmark10_boxes3d": {k: c3.get(k, 0) for k in BENCHMARK_10},
            "top50_classes3d": c3.most_common(50),
        }
    sensors = Counter(by_id[i].sensor for i in test)
    report["test_sensors"] = dict(sensors)
    out = REPO / "benchmarks" / "data_quality" / "sunrgbd_annotations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"images {len(anns)} | official train {len(tr_seq)} test {len(te_seq)} unassigned {len(unassigned)}")
    for name, s in report["splits"].items():
        print(f"{name:5s} images {s['images']:5d}  3D boxes {s['boxes3d']:6d}  2D boxes {s['boxes2d']:6d}  classes {s['n_classes3d']}")
    print("benchmark-10 (test):", report["splits"]["test"]["benchmark10_boxes3d"])
    print("overlap check: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
