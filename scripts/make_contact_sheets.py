"""Contact sheets of every 30th TUM frame with operational detections drawn (spec 04.2 R1, T1).

Used to count the distinct static objects per class by eye for docs/tum_object_inventory.yaml.
    python scripts/make_contact_sheets.py
Writes benchmarks/tracking/contact/<seq>_<k>.jpg (4 x 5 thumbnails per sheet).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from perception.config import load_config  # noqa: E402
from perception.data.tum import TUMSequence  # noqa: E402
from perception.detection.classes import COCO_NAMES  # noqa: E402
from perception.detection.runio import read_run  # noqa: E402
from perception.tracking.runner import align_records  # noqa: E402

STEP, COLS, ROWS, TW, TH = 30, 4, 5, 400, 300
SEQS = {"fr1": ("rgbd_dataset_freiburg1_desk", "tum_fr1_desk_full.jsonl"),
        "fr3": ("rgbd_dataset_freiburg3_long_office_household", "tum_fr3_long_office_full.jsonl")}
OUT = REPO / "benchmarks" / "tracking" / "contact"


def colour(c: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(c * 7 + 3)
    return tuple(int(x) for x in rng.integers(60, 255, 3))


def main() -> int:
    dcfg = load_config("dataset")
    conf = float(load_config("detector")["conf_operational"])
    excluded = set(load_config("tracker")["excluded_static_classes"])
    OUT.mkdir(parents=True, exist_ok=True)
    for alias, (name, run) in SEQS.items():
        seq = TUMSequence.from_config(name, dcfg)
        recs = align_records(seq, read_run(REPO / "benchmarks" / "detection" / run)[1])
        idx = list(range(0, len(seq), STEP))
        per = COLS * ROWS
        for k in range(0, len(idx), per):
            sheet = np.zeros((ROWS * TH, COLS * TW, 3), np.uint8)
            for j, i in enumerate(idx[k:k + per]):
                img = cv2.cvtColor(seq[i].rgb, cv2.COLOR_RGB2BGR)
                r = recs[i]
                for b, s, c in zip(r["boxes"], r["scores"], r["class_ids"]):
                    if s < conf or COCO_NAMES[c] in excluded:
                        continue
                    x1, y1, x2, y2 = map(int, b)
                    cv2.rectangle(img, (x1, y1), (x2, y2), colour(c), 2)
                    cv2.putText(img, COCO_NAMES[c], (x1 + 2, max(y1 + 14, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                colour(c), 2)
                th = cv2.resize(img, (TW, TH), interpolation=cv2.INTER_AREA)
                cv2.putText(th, f"#{i}", (5, TH - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                rr, cc = divmod(j, COLS)
                sheet[rr * TH:(rr + 1) * TH, cc * TW:(cc + 1) * TW] = th
            p = OUT / f"{alias}_{k // per:02d}.jpg"
            cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(p.name, len(idx[k:k + per]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
