"""Generate docs/data_quality_report.md from benchmarks/data_quality/{tum,sunrgbd}.json (spec 01.4, R3/T3).

The numbers come only from the JSON. Hand-written text is limited to HANDLING (how each anomaly type is
handled) and OBSERVATIONS (visual findings that cannot be measured automatically), both clearly labelled.

Usage:  python -I scripts/make_quality_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DQ = REPO / "benchmarks" / "data_quality"

HANDLING = {
    "low_valid_depth": "Per-object depth uses only valid pixels, and boxes with too few valid pixels are flagged rather than given a guessed depth (Phase 03). Results are reported per sensor or sequence.",
    "many_frames_under_half_valid": "Same as low_valid_depth. These frames stay in the benchmarks, so the reported numbers include hard cases.",
    "frames_without_depth": "Excluded from 3D metrics and counted as a 3D failure; still used for 2D detection.",
    "long_range_depth": "Depth beyond the sensor's reliable range is still kept up to the 10 m cutoff. Phase 03 reports 3D error binned by distance, so far objects cannot hide inside an average.",
    "mixed_image_sizes": "Every frame carries its own K; the readers never assume a fixed size.",
    "raw_depth_already_filled": "For this sensor, raw `depth/` equals hole-filled `depth_bfx/`, so 'raw' depth was already processed upstream. Its valid fraction overstates true sensor coverage, so the report compares sensors with that in mind.",
    "images_without_3d_boxes": "Kept as negatives for 2D detection, and skipped in 3D box metrics.",
}

OBSERVATIONS = [
    "TUM fr1_desk has strong motion blur in several frames (e.g. 0, 163 and 245 of the 573 associated frames). This makes it a hard case for detection and tracking, and is a reason to keep it. Contact sheets: `benchmarks/data_quality/tum_presence/`.",
    "SUN `depth_bfx/` differs from raw `depth/` in a median 92–96% of pixels for every sensor, including pixels where raw depth is valid. It is a re-processed map, not just raw with holes filled, which supports using raw `depth/`. As a result, the 'raw = bfx' column cannot show whether Xtion/kv1 raw depth (nearly hole-free) was pre-filled upstream; that question stays open.",
    "TUM depth has a ~30–40 px registration border with no depth on the right and top edges (spec 01.2).",
]


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def group_table(d: dict) -> list[str]:
    has_bfx = any("frames_raw_equals_bfx" in g for g in d["groups"].values())
    head = ["Group", "Frames", "Image sizes", "Valid fraction p5 / p50 / p95", "Frames < 50% valid",
            "Median depth per frame p5 / p50 / p95 (m)", "Pixel depth p95 / p99 (m)"]
    if has_bfx:
        head.append("Frames with raw = bfx")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, g in d["groups"].items():
        vf = g["valid_frac_p5_p50_p95"]
        md = g["frame_median_depth_p5_p50_p95_m"]
        row = [f"`{name}`", f"{g['frames']:,}", ", ".join(g["image_sizes"]),
               " / ".join(f"{v:.2f}" for v in vf), f"{g['frames_valid_lt_0.5']} ({pct(g['frames_valid_lt_0.5'] / g['frames'])})",
               " / ".join(f"{v:.2f}" for v in md) if md else "n/a",
               f"{g['pixel_depth_quantiles_m']['p95']} / {g['pixel_depth_quantiles_m']['p99']}"]
        if has_bfx:
            row.append(pct(g.get("frames_raw_equals_bfx")))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def presence_section() -> list[str]:
    """R5 object presence, measured by the Phase 02 detector (spec 02.2 R4). Replaces the earlier qualitative note."""
    p = REPO / "benchmarks" / "detection" / "tum_presence.json"
    if not p.exists():
        return ["## TUM object presence (R5)", "",
                "Not yet measured. Run `scripts/detection_presence_gallery.py` after the Phase 02 runs.", ""]
    d = json.loads(p.read_text(encoding="utf-8"))
    L = ["## TUM object presence (R5, measured)", "",
         f"Measured with YOLO26-n at the operational threshold conf {d['conf_operational']} (spec 02.2 R4, "
         "`benchmarks/detection/tum_presence.json`). The counts are detector output, not ground truth: TUM has no object "
         "labels, so a missed or hallucinated object is not visible here.", "",
         "| Sequence | Frames | With >= 1 detection | Most frequent classes (frames containing) |", "|---|---|---|---|"]
    for seq, v in d["sequences"].items():
        top = ", ".join(f"{k} {c['frames']}" for k, c in list(v["per_class"].items())[:6])
        L.append(f"| {seq} | {v['n_frames']:,} | {v['frames_with_detection']:,} ({pct(v['share_with_detection'])}) | {top} |")
    return L + [""]


def main() -> int:
    data = {}
    for name in ("tum", "sunrgbd"):
        p = DQ / f"{name}.json"
        if not p.exists():
            raise SystemExit(f"missing {p}; run scripts/inspect_dataset.py --dataset {name}")
        data[name] = json.loads(p.read_text(encoding="utf-8"))

    L = ["# Data Quality Report (spec 01.4)", "",
         "_Generated by `scripts/make_quality_report.py` from `benchmarks/data_quality/{tum,sunrgbd}.json`. Do not edit by hand; "
         "re-run the scripts instead._", ""]
    for name, d in data.items():
        L += [f"## {'TUM RGB-D' if name == 'tum' else 'SUN RGB-D'}", "",
              f"- Frames: **{d['frames']:,}** ({'all' if d['n_requested'] == 0 else f'sample n={d['n_requested']}'}, seed {d['seed']}); "
              f"grouped by {d['group_by']}.",
              f"- Valid depth = finite and within {d['depth_range_m'][0]}–{d['depth_range_m'][1]} m. "
              f"Git commit: `{d['git_commit'] or 'none (no commit yet)'}`; config hash `{d['config_sha256_16']}`; generated {d['generated_utc']}.",
              f"- Plot: [{d['plots'][0]}](../{d['plots'][0]})", ""]
        L += group_table(d) + [""]
        c = d.get("classes") or {}
        if c:
            L += ["### COCO-overlap classes (2D boxes, mapping in `configs/class_map.yaml`)", "",
                  "| Split | 2D boxes | Mapped to COCO | Share |", "|---|---|---|---|"]
            for s in ("train", "val", "test"):
                L.append(f"| {s} | {c[s]['boxes2d']:,} | {c[s]['boxes2d_mapped_to_coco']:,} | {pct(c[s]['mapped_share'])} |")
            L += ["", "Test-split counts per COCO class:", "",
                  "| " + " | ".join(c["test"]["coco_counts"]) + " |",
                  "|" + "---|" * len(c["test"]["coco_counts"]),
                  "| " + " | ".join(f"{v:,}" for v in c["test"]["coco_counts"].values()) + " |", "",
                  f"Images without 3D boxes: {c['images_without_3d_boxes']}; without 2D boxes: {c['images_without_2d_boxes']}.", ""]

    L += ["## Anomalies and handling (AC3)", "",
          "Detected automatically by rules in `scripts/inspect_dataset.py`. The handling text is written in `make_quality_report.py`.", "",
          "| Dataset | Group | Anomaly | Value | Handling |", "|---|---|---|---|---|"]
    for name, d in data.items():
        for a in d["anomalies"]:
            L.append(f"| {name} | `{a['group']}` | {a['id']} | {a['value']} | {HANDLING.get(a['id'], '**unhandled; decide before Phase 02**')} |")
    L += ["", "## Visual observations (not automated)", ""] + [f"- {o}" for o in OBSERVATIONS] + [""]
    L += presence_section()
    L += ["## Benchmark frame lists (R4)", "",
          "Made with `scripts/make_benchmark_frames.py` (`perception.data.sampling.sample_ids`, seed 0) and stored in `configs/benchmark_frames/`:", ""]
    for p in sorted((REPO / "configs" / "benchmark_frames").glob("*.txt")):
        n = sum(1 for s in p.read_text(encoding="utf-8").splitlines() if s.strip() and not s.startswith("#"))
        L.append(f"- `{p.name}`: {n} frames")
    L.append("")
    (REPO / "docs" / "data_quality_report.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote docs/data_quality_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
