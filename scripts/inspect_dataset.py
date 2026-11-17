"""Dataset quality inspection (spec 01.4, R1–R3).

Per frame: valid-depth fraction, depth percentiles (5/50/95) over valid pixels, image size; for SUN RGB-D also
the sensor and the fraction of pixels where raw `depth/` differs from hole-filled `depth_bfx/`.
Aggregated per group (TUM: sequence; SUN: sensor) into benchmarks/data_quality/<dataset>.json; per-frame rows go
to <dataset>_frames.csv; histograms to benchmarks/data_quality/plots/.

Deterministic: same --n/--seed gives identical JSON apart from `generated_utc` (AC1).

Usage:  python -I scripts/inspect_dataset.py --dataset {tum,sunrgbd} [--n 0] [--seed 0]     (n=0: all frames)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from perception.config import CONFIG_DIR, data_root, load_config  # noqa: E402
from perception.data.quality import (DEPTH_HIST_EDGES, depth_histogram, depth_percentiles,  # noqa: E402
                                     differs_fraction, valid_fraction)
from perception.data.sampling import sample_ids  # noqa: E402
from perception.data.splits import read_split  # noqa: E402
from perception.data.sunrgbd_annotations import load_annotations  # noqa: E402
from perception.data.tum import TUMSequence  # noqa: E402
from perception.depth.decode import decode_sun  # noqa: E402

OUT = REPO / "benchmarks" / "data_quality"
ND = 4  # decimals kept in JSON


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", "-q", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return (r.stdout.strip() or None) if r.returncode == 0 else None  # None until the first commit
    except (OSError, subprocess.SubprocessError):
        return None


def config_hash() -> str:
    h = hashlib.sha256()
    for name in ("dataset.yaml", "class_map.yaml"):
        h.update((CONFIG_DIR / name).read_bytes())
    return h.hexdigest()[:16]


def q(a, qs=(5, 50, 95)):
    a = np.asarray(a, dtype=np.float64)
    return [round(float(x), ND) for x in np.percentile(a, qs)] if a.size else None


def frame_row(fid: str, group: str, depth: np.ndarray, valid: np.ndarray) -> dict:
    p = depth_percentiles(depth, valid)
    return {"id": fid, "group": group, "h": int(depth.shape[0]), "w": int(depth.shape[1]),
            "valid_frac": round(valid_fraction(valid), ND),
            "p5": None if p is None else round(p[0], ND),
            "p50": None if p is None else round(p[1], ND),
            "p95": None if p is None else round(p[2], ND)}


# ---------------------------------------------------------------- readers
def iter_tum(cfg, n, seed):
    root = data_root(cfg) / "tum"
    for name in cfg["tum"]["sequences"]:
        seq = TUMSequence.from_config(name, cfg)
        idx = list(range(len(seq)))
        if n:
            keep = set(sample_ids([f"{i:06d}" for i in idx], n, seed))
            idx = [i for i in idx if f"{i:06d}" in keep]
        for i in idx:
            f = seq[i]
            yield f.frame_id, name, f.depth_m, f.valid, None


def iter_sun(cfg, n, seed):
    sun = data_root(cfg) / "sunrgbd"
    anns = load_annotations(sun)
    ids = [a.image_id for a in anns]
    keep = set(sample_ids(ids, n, seed)) if n else set(ids)
    lo, hi = cfg["depth"]["min_m"], cfg["depth"]["max_m"]
    for a in anns:
        if a.image_id not in keep:
            continue
        raw = cv2.imread(str(sun / a.depth_rel), cv2.IMREAD_UNCHANGED)
        bfx_path = sun / a.depth_rel.replace("/depth/", "/depth_bfx/")
        bfx = cv2.imread(str(bfx_path), cv2.IMREAD_UNCHANGED) if bfx_path.exists() else None
        depth, valid = decode_sun(raw, lo, hi)
        diff = differs_fraction(raw, bfx) if bfx is not None and bfx.shape == raw.shape else None
        yield a.image_id, a.sensor, depth, valid, diff


# ---------------------------------------------------------------- aggregation
def aggregate(rows, hists, diffs):
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    out = {}
    for g in sorted(groups):
        rs = groups[g]
        vf = [r["valid_frac"] for r in rs]
        med = [r["p50"] for r in rs if r["p50"] is not None]
        p95 = [r["p95"] for r in rs if r["p95"] is not None]
        h = hists[g]
        cdf = np.cumsum(h) / max(1, h.sum())
        d = {
            "frames": len(rs),
            "image_sizes": {f"{k[0]}x{k[1]}": v for k, v in sorted(Counter((r["h"], r["w"]) for r in rs).items())},
            "valid_frac_mean": round(float(np.mean(vf)), ND),
            "valid_frac_p5_p50_p95": q(vf),
            "frames_valid_lt_0.5": int(sum(v < 0.5 for v in vf)),
            "frames_no_valid_depth": int(sum(r["p50"] is None for r in rs)),
            "frame_median_depth_p5_p50_p95_m": q(med),
            "frame_p95_depth_p50_m": q(p95, (50,))[0] if p95 else None,
            "frames_median_in_0.5_6m": round(float(np.mean([(0.5 <= m <= 6.0) for m in med])), ND) if med else None,
            "pixel_depth_quantiles_m": {f"p{int(p * 100)}": round(float(DEPTH_HIST_EDGES[np.searchsorted(cdf, p) + 1]), 1)
                                        for p in (0.05, 0.5, 0.95, 0.99)},
            "pixel_depth_hist_0.1m": h.tolist(),
        }
        if diffs.get(g):
            dv = diffs[g]
            d["bfx_differs_frac_p50"] = q(dv, (50,))[0]
            d["frames_raw_equals_bfx"] = round(float(np.mean([x == 0.0 for x in dv])), ND)
        out[g] = d
    return out


def class_counts(cfg) -> dict:
    sun = data_root(cfg) / "sunrgbd"
    anns = {a.image_id: a for a in load_annotations(sun)}
    cmap = yaml.safe_load((CONFIG_DIR / "class_map.yaml").read_text(encoding="utf-8"))["sun_to_coco"]
    res = {}
    for split in ("train", "val", "test"):
        ids = read_split(REPO / "configs" / "splits" / f"sunrgbd_{split}.txt")
        coco, total, mapped = Counter(), 0, 0
        for i in ids:
            for b in anns[i].boxes2d:
                total += 1
                c = cmap.get(b.cls.replace(" ", "_"))
                if c:
                    coco[c] += 1
                    mapped += 1
        res[split] = {"boxes2d": total, "boxes2d_mapped_to_coco": mapped,
                      "mapped_share": round(mapped / max(1, total), ND),
                      "coco_counts": dict(sorted(coco.items(), key=lambda kv: (-kv[1], kv[0])))}
    res["images_without_3d_boxes"] = sum(1 for a in anns.values() if not a.boxes3d)
    res["images_without_2d_boxes"] = sum(1 for a in anns.values() if not a.boxes2d)
    return res


def detect_anomalies(dataset: str, groups: dict, extra: dict) -> list[dict]:
    """Rule-based flags; the report generator attaches how each is handled."""
    an = []
    for g, d in groups.items():
        if d["valid_frac_p5_p50_p95"][1] < 0.75:
            an.append({"id": "low_valid_depth", "group": g, "value": d["valid_frac_p5_p50_p95"][1]})
        if d["frames_valid_lt_0.5"] / d["frames"] > 0.05:
            an.append({"id": "many_frames_under_half_valid", "group": g,
                       "value": round(d["frames_valid_lt_0.5"] / d["frames"], ND)})
        if d["frames_no_valid_depth"]:
            an.append({"id": "frames_without_depth", "group": g, "value": d["frames_no_valid_depth"]})
        if d["pixel_depth_quantiles_m"]["p99"] > 7.0:
            an.append({"id": "long_range_depth", "group": g, "value": d["pixel_depth_quantiles_m"]["p99"]})
        if len(d["image_sizes"]) > 1:
            an.append({"id": "mixed_image_sizes", "group": g, "value": d["image_sizes"]})
        if d.get("frames_raw_equals_bfx", 0) > 0.5:
            an.append({"id": "raw_depth_already_filled", "group": g, "value": d["frames_raw_equals_bfx"]})
    if extra.get("images_without_3d_boxes"):
        an.append({"id": "images_without_3d_boxes", "group": "all", "value": extra["images_without_3d_boxes"]})
    return an


def plots(dataset: str, rows: list[dict], groups: dict) -> list[str]:
    pdir = OUT / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    names = sorted(groups)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for g in names:
        rs = [r for r in rows if r["group"] == g]
        ax[0].hist([r["valid_frac"] for r in rs], bins=np.linspace(0, 1, 41), histtype="step", label=g, density=True)
        ax[1].hist([r["p50"] for r in rs if r["p50"] is not None], bins=np.linspace(0, 8, 41), histtype="step",
                   label=g, density=True)
        h = np.asarray(groups[g]["pixel_depth_hist_0.1m"], dtype=float)
        ax[2].plot(DEPTH_HIST_EDGES[:-1] + 0.05, h / max(1.0, h.sum()), label=g)
    ax[0].set(title="Valid-depth fraction per frame", xlabel="fraction", ylabel="density")
    ax[1].set(title="Median depth per frame", xlabel="m", ylabel="density")
    ax[2].set(title="Valid pixel depth distribution", xlabel="m", ylabel="share per 0.1 m")
    for a in ax:
        a.legend(fontsize=8)
    fig.suptitle(f"{dataset}: {len(rows)} frames")
    fig.tight_layout()
    p = pdir / f"{dataset}_distributions.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return [p.relative_to(REPO).as_posix()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tum", "sunrgbd"], required=True)
    ap.add_argument("--n", type=int, default=0, help="frames to sample per dataset (TUM: per sequence); 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config("dataset")

    it = iter_tum(cfg, args.n, args.seed) if args.dataset == "tum" else iter_sun(cfg, args.n, args.seed)
    rows, hists, diffs = [], defaultdict(lambda: np.zeros(len(DEPTH_HIST_EDGES) - 1, np.int64)), defaultdict(list)
    for k, (fid, group, depth, valid, diff) in enumerate(it):
        rows.append(frame_row(fid, group, depth, valid))
        hists[group] += depth_histogram(depth, valid)
        if diff is not None:
            diffs[group].append(round(diff, ND))
        if (k + 1) % 500 == 0:
            print(f"  {k + 1} frames", flush=True)

    groups = aggregate(rows, hists, diffs)
    extra = class_counts(cfg) if args.dataset == "sunrgbd" else {}
    report = {
        "dataset": args.dataset,
        "n_requested": args.n,
        "seed": args.seed,
        "frames": len(rows),
        "git_commit": git_commit(),
        "config_sha256_16": config_hash(),
        "depth_range_m": [cfg["depth"]["min_m"], cfg["depth"]["max_m"]],
        "group_by": "sequence" if args.dataset == "tum" else "sensor",
        "groups": groups,
        "classes": extra,
        "anomalies": detect_anomalies(args.dataset, groups, extra),
        "plots": plots(args.dataset, rows, groups),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.dataset}.json").write_text(json.dumps(report, indent=1, sort_keys=False), encoding="utf-8")
    with open(OUT / f"{args.dataset}_frames.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{args.dataset}: {len(rows)} frames")
    for g, d in groups.items():
        print(f"  {g:45s} frames {d['frames']:5d} valid p50 {d['valid_frac_p5_p50_p95'][1]:.3f} "
              f"median depth p50 {d['frame_median_depth_p5_p50_p95_m'][1]:.2f} m "
              f"pixel p99 {d['pixel_depth_quantiles_m']['p99']} m"
              + (f" raw==bfx {d['frames_raw_equals_bfx']}" if "frames_raw_equals_bfx" in d else ""))
    for a in report["anomalies"]:
        print("  anomaly:", a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
