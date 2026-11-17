"""Write the fixed benchmark frame lists (spec 01.4, R4/R5) to configs/benchmark_frames/.

- sunrgbd_test_500.txt / sunrgbd_val_200.txt : sampled image ids (detection + 3D accuracy dev/benchmark).
- tum_<sequence>_200.txt : sampled frame timestamps (latency benchmarks on real frames).
  Tracking uses the full associated sequences, not these lists.
Also writes an 8-frame contact sheet per TUM sequence for the qualitative object-presence check (R5).

Usage:  python -I scripts/make_benchmark_frames.py [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perception.config import data_root, load_config  # noqa: E402
from perception.data.sampling import sample_ids, write_ids  # noqa: E402
from perception.data.tum import TUMSequence  # noqa: E402

OUT = REPO / "configs" / "benchmark_frames"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config("dataset")

    for split, n in (("sunrgbd_test", 500), ("sunrgbd_val", 200)):
        ids = sample_ids(split, n, args.seed)
        write_ids(OUT / f"{split}_{n}.txt", ids, f"sample_ids('{split}', {n}, seed={args.seed})")
        print(f"{split}_{n}: {len(ids)} ids")

    root = data_root(cfg) / "tum"
    sheet_dir = REPO / "benchmarks" / "data_quality" / "tum_presence"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for name in cfg["tum"]["sequences"]:
        seq = TUMSequence.from_config(name, cfg)
        stamps = [f"{t:.6f}" for t, _, _ in seq.items]
        ids = sample_ids(stamps, 200, args.seed)
        write_ids(OUT / f"tum_{name}_200.txt", ids, f"sample_ids(<{name} associated stamps>, 200, seed={args.seed})")
        print(f"tum_{name}_200: {len(ids)} stamps of {len(stamps)}")
        # 8 evenly spaced frames for the object-presence check
        tiles = []
        for i in np.linspace(0, len(seq) - 1, 8).round().astype(int):
            im = cv2.cvtColor(seq[int(i)].rgb, cv2.COLOR_RGB2BGR)
            im = cv2.resize(im, (320, 240))
            cv2.putText(im, f"{i}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(im)
        cv2.imwrite(str(sheet_dir / f"{name}.jpg"), np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
