"""Depth/RGB alignment check (spec 01.2, T6/AC2/AC3).

For N evenly spaced frames per sequence: overlay depth edges (red) on RGB edges (grey image),
save PNGs, and report median valid depth and valid fraction per frame.

    python scripts/check_alignment.py --dataset tum --n 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from perception.config import REPO_ROOT, data_root, load_config
from perception.data.frame import Frame
from perception.data.tum import TUMSequence

OUT = REPO_ROOT / "benchmarks" / "data_quality" / "alignment"


def depth_edges(depth_m: np.ndarray, valid: np.ndarray, jump_m: float = 0.10) -> np.ndarray:
    """Pixels where depth jumps by more than jump_m to a neighbour (both valid), or valid/invalid borders."""
    d = depth_m
    e = np.zeros_like(valid)
    for dy, dx in ((0, 1), (1, 0)):
        a, b = d[: d.shape[0] - dy, : d.shape[1] - dx], d[dy:, dx:]
        va, vb = valid[: d.shape[0] - dy, : d.shape[1] - dx], valid[dy:, dx:]
        jump = va & vb & (np.abs(a - b) > jump_m)
        e[: d.shape[0] - dy, : d.shape[1] - dx] |= jump
    return e


def edge_agreement(f: Frame, tol_px: int = 3) -> float:
    """Fraction of depth-edge pixels lying within tol_px of an RGB Canny edge (higher = better aligned)."""
    de = depth_edges(f.depth_m, f.valid)
    if de.sum() == 0:
        return float("nan")
    gray = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2GRAY)
    ce = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150) > 0
    near = cv2.dilate(ce.astype(np.uint8), np.ones((2 * tol_px + 1,) * 2, np.uint8)) > 0
    return float((de & near).sum() / de.sum())


def overlay(f: Frame) -> np.ndarray:
    gray = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray // 2 + 64, cv2.COLOR_GRAY2BGR)
    de = cv2.dilate(depth_edges(f.depth_m, f.valid).astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    img[de] = (0, 0, 255)
    img[~f.valid] = (img[~f.valid] * 0.5 + np.array([80, 0, 0]) * 0.5).astype(np.uint8)  # invalid = blue tint
    return img


def run_tum(n: int) -> dict:
    cfg = load_config("dataset")
    root = data_root(cfg) / "tum"
    report = {}
    for seq_name in cfg["tum"]["sequences"]:
        seq = TUMSequence.from_config(seq_name, cfg)
        idxs = np.linspace(0, len(seq) - 1, n).round().astype(int)
        rows = []
        for i in idxs:
            f = seq[int(i)]
            cv2.imwrite(str(OUT / f"tum_{seq_name}_{i:05d}.png"), overlay(f))
            rows.append({
                "index": int(i), "frame_id": f.frame_id,
                "median_depth_m": float(np.median(f.depth_m[f.valid])) if f.valid.any() else None,
                "valid_fraction": float(f.valid.mean()),
                # Diagnostic only: Canny also fires on texture, so this is NOT an acceptance metric (see docs/calibration.md)
                "edge_agreement": edge_agreement(f),
            })
        med = [r["median_depth_m"] for r in rows if r["median_depth_m"] is not None]
        report[seq_name] = {
            "n_rgb": seq.n_rgb, "n_depth": seq.n_depth, "n_associated": seq.n_pairs, "n_with_pose": len(seq),
            "K": seq.K.tolist(),
            "frac_median_depth_in_0p5_6m": float(np.mean([0.5 <= m <= 6 for m in med])),
            "frames": rows,
        }
    return report


def run_sun(n: int, n_stats: int = 200, seed: int = 0) -> dict:
    from perception.data.sunrgbd import SunRGBD

    cfg = load_config("dataset")
    ds = SunRGBD(data_root(cfg) / "sunrgbd", min_m=cfg["depth"]["min_m"], max_m=cfg["depth"]["max_m"])
    rng = np.random.default_rng(seed)
    by_sensor: dict[str, list[int]] = {}
    for r in ds.records:
        by_sensor.setdefault(r.sensor, []).append(r.index)
    report = {"n_records": len(ds), "per_sensor": {}}
    for sensor, idxs in sorted(by_sensor.items()):
        k_vis = max(1, round(n * len(idxs) / len(ds)))
        k_stats = max(10, round(n_stats * len(idxs) / len(ds)))
        pick = rng.choice(idxs, size=min(k_stats, len(idxs)), replace=False)
        rows = []
        for j, i in enumerate(pick):
            f = ds[int(i)]
            if j < k_vis:
                cv2.imwrite(str(OUT / f"sun_{sensor}_{f.frame_id}.png"), overlay(f))
            rows.append({
                "index": int(i), "frame_id": f.frame_id, "hw": list(f.hw),
                "median_depth_m": float(np.median(f.depth_m[f.valid])) if f.valid.any() else None,
                "valid_fraction": float(f.valid.mean()),
            })
        med = [r["median_depth_m"] for r in rows if r["median_depth_m"] is not None]
        report["per_sensor"][sensor] = {
            "n_records": len(idxs), "n_sampled": len(rows), "n_overlays": k_vis,
            "image_sizes": sorted({tuple(r["hw"]) for r in rows}),
            "frac_median_depth_in_0p5_6m": float(np.mean([0.5 <= m <= 6 for m in med])),
            "median_of_median_depth_m": float(np.median(med)),
            "median_valid_fraction": float(np.median([r["valid_fraction"] for r in rows])),
            "frames": rows,
        }
    all_med = [r["median_depth_m"] for s in report["per_sensor"].values() for r in s["frames"] if r["median_depth_m"]]
    report["frac_median_depth_in_0p5_6m"] = float(np.mean([0.5 <= m <= 6 for m in all_med]))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tum", "sun"], default="tum")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.dataset == "sun":
        report = run_sun(args.n)
        out = REPO_ROOT / "benchmarks" / "data_quality" / "alignment_sun.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"records: {report['n_records']}; median depth in 0.5-6 m overall: "
              f"{100 * report['frac_median_depth_in_0p5_6m']:.1f}%")
        for s, r in report["per_sensor"].items():
            print(f"  {s:10s} n={r['n_records']:5d} sampled={r['n_sampled']:3d} sizes={r['image_sizes']} "
                  f"in-range={100 * r['frac_median_depth_in_0p5_6m']:.0f}% median_depth={r['median_of_median_depth_m']:.2f} m "
                  f"valid={r['median_valid_fraction']:.2f}")
        print(f"wrote {out}")
        return 0
    report = run_tum(args.n)
    out = REPO_ROOT / "benchmarks" / "data_quality" / f"alignment_{args.dataset}.json"
    out.write_text(json.dumps(report, indent=2))
    for name, r in report.items():
        fr = r["frames"]
        print(f"{name}: rgb={r['n_rgb']} depth={r['n_depth']} associated={r['n_associated']} with_pose={r['n_with_pose']}")
        print(f"  median depth in 0.5-6 m: {100 * r['frac_median_depth_in_0p5_6m']:.0f}% of {len(fr)} frames")
        print(f"  median depth per frame (m): {[round(x['median_depth_m'], 2) for x in fr]}")
        print(f"  valid fraction: {[round(x['valid_fraction'], 2) for x in fr]}")
        print(f"  edge agreement (diagnostic): {[round(x['edge_agreement'], 2) for x in fr]}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
