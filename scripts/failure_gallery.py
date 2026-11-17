"""Spec 05.4 R1-R2: render the rule-selected failure gallery (SUN test, detected mode) and count review categories.

Step 1 (no review yet):   python -I scripts/failure_gallery.py
  -> benchmarks/results/failure_gallery.json (selected items) and failure_gallery_<group>.png
Step 2 (after the by-eye review in docs/failure_review.yaml):   python -I scripts/failure_gallery.py --counts
  -> adds the category counts and the background-bleed answer to failure_gallery.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perception.config import data_root, load_config  # noqa: E402
from perception.data.sunrgbd_annotations import load_annotations  # noqa: E402
from perception.depth.decode import decode_sun  # noqa: E402
from perception.depth.estimate import region_of  # noqa: E402
from perception.eval.failures import GROUPS, category_counts, select  # noqa: E402
from perception.eval.results_index import git_state, sha256_file  # noqa: E402

OUT = REPO / "benchmarks" / "results"
NPZ = OUT / "localization_sun_test_objects.npz"
REVIEW = REPO / "docs" / "failure_review.yaml"
TITLES = {"large_error": "Largest 3D errors (valid)", "no_depth": "no_depth matched objects",
          "missed": "Missed GT objects (top-4 classes)", "false_positive": "High-score false positives"}


def render(items: list[dict], path: Path, shrink: float, cols: int = 4) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    dcfg = load_config("dataset")
    root = data_root(dcfg) / "sunrgbd"
    anns = {a.image_id: a for a in load_annotations(root)}
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, 2 * cols, figsize=(4.2 * cols, 2.5 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for n, it in enumerate(items):
        ann = anns[it["image_id"]]
        rgb = cv2.cvtColor(cv2.imread(str(root / ann.rgb_rel), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        d, v = decode_sun(cv2.imread(str(root / ann.depth_rel), cv2.IMREAD_UNCHANGED),
                          dcfg["depth"]["min_m"], dcfg["depth"]["max_m"])
        x1, y1, x2, y2 = it["box"]
        c0, r0, c1, r1 = region_of(it["box"], d.shape, shrink)
        r, c = divmod(n, cols)
        for ax, img, kw in ((axes[r, 2 * c], rgb, {}), (axes[r, 2 * c + 1], np.where(v, d, np.nan), {"cmap": "turbo"})):
            ax.imshow(img, **kw)
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=1.5))
            if it["group"] != "missed":
                ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec="yellow", lw=1, ls="--"))
            pad = 0.6 * max(x2 - x1, y2 - y1)
            ax.set_xlim(max(0, x1 - pad), min(img.shape[1], x2 + pad))
            ax.set_ylim(min(img.shape[0], y2 + pad), max(0, y1 - pad))
        sub = (f"err {it['err3d_m']:.2f} m, GT Z {it['gt_centre'][2]:.1f}" if it.get("err3d_m") is not None
               else f"score {it['score']}" if it.get("score") is not None else f"GT Z {it['gt_centre'][2]:.1f}")
        axes[r, 2 * c].set_title(f"#{n + 1} {it['image_id']} {it['raw_class']} ({it['sensor']})", fontsize=7)
        axes[r, 2 * c + 1].set_title(f"{sub}, frac {it.get('valid_frac', float('nan')):.2f}", fontsize=7)
    fig.suptitle(TITLES[items[0]["group"]] + "  (green: box; yellow dashed: depth sampling region)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=85)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", action="store_true", help="read docs/failure_review.yaml and add the counts")
    a = ap.parse_args(argv)
    z = dict(np.load(NPZ, allow_pickle=False))
    items = select(z)
    path = OUT / "failure_gallery.json"
    if not a.counts:
        shrink = float(load_config("depth")["association"]["shrink"]) if "association" in load_config("depth") \
            else float(load_config("depth")["shrink"])
        figs = {}
        for g, v in items.items():
            p = OUT / f"failure_gallery_{g}.png"
            render(v, p, shrink)
            figs[g] = p.relative_to(REPO).as_posix()
        doc = {"spec": "05.4 R1", "label": "Measured: rule-selected SUN test failure cases (detected mode)",
               "rules": {g: n for g, n in GROUPS.items()}, "items": items, "figures": figs,
               "header": {**git_state(REPO), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "input_sha256": {"objects_npz": sha256_file(NPZ)}}}
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)} and {len(figs)} figures; review them into {REVIEW.relative_to(REPO)}")
        return 0
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc["items"] != json.loads(json.dumps(items)):
        raise SystemExit("gallery selection changed since rendering; re-render first")
    review = yaml.safe_load(REVIEW.read_text(encoding="utf-8"))
    counts = category_counts(review, items)
    le = counts["per_group"]["large_error"]
    bleed = le.get("background bleed", 0)
    n_le = sum(le.values())
    doc["counts"] = counts
    doc["background_bleed"] = {
        "large_error_items": n_le, "background_bleed": bleed, "share": round(bleed / n_le, 3) if n_le else None,
        "dominates": bleed > n_le / 2,
        "next_most_common": max(((c, k) for c, k in le.items() if c != "background bleed"), key=lambda x: x[1],
                                default=(None, 0))[0]}
    doc["link_sensitivity"] = ("weak GT 2D-3D links (IoU < 0.3) are quantified in 05.2: "
                               "localization_sun_test.json -> <mode>.link_iou_lt_0.3_fraction and "
                               "sensitivity_link_iou_ge_0.3_valid")
    doc["review_sha256"] = sha256_file(REVIEW)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(json.dumps({"total": counts["total"], "background_bleed": doc["background_bleed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
