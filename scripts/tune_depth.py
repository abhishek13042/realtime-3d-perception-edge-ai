"""Spec 03.3: choose the depth-association parameters on SUN RGB-D val (never test).

  python -I scripts/tune_depth.py --split sunrgbd_val                  # compute (~15 min on CPU) + analyse
  python -I scripts/tune_depth.py --split sunrgbd_val --stage analyse  # re-analyse the cached grid, rewrite outputs
  python -I scripts/tune_depth.py --split sunrgbd_val --limit 20       # smoke run, writes *_smoke files only

Compute stage: one pass over val depth maps (no RGB, no detector). Every GT object linked to a 3D box (R1) is
evaluated with its GT 2D box (oracle mode) and with the stored sun_val_full detection matched to it at
conf_operational (detected mode, R2), for every grid configuration (R5). Per object and configuration it stores
n_valid, valid_frac, spread and the 3D centre, so status thresholds can be swept afterwards without re-estimating.

Analyse stage, with rules fixed before the run:
  R5  among configurations with oracle valid fraction >= 0.6 (status rule with the placeholder thresholds
      n_min 20, tau_cov 0.3, tau_spread 0.25), minimise the oracle median 3D centre error over `valid` objects.
      Within 5 mm of the best: prefer estimator A, then larger shrink, then the lower error.
  R6  on the chosen configuration (oracle): tau_cov x tau_spread grid; maximise median error(low_coverage) /
      median error(valid), subject to valid fraction >= 0.6 and >= 30 low_coverage objects. n_min stays 20 unless
      objects with 20-50 valid pixels have > 2x the median error of objects with >= 50 (then n_min = 50).
  R7  benchmarks/geometry/tune_val.json, configs/depth.yaml, docs/depth_tuning.md (generated), worst-12 gallery.
      Gallery categories come from docs/depth_tuning_review.yaml (filled in by eye after looking at the gallery).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perception.config import data_root, load_config  # noqa: E402
from perception.data.splits import read_split  # noqa: E402
from perception.data.sunrgbd_annotations import load_annotations  # noqa: E402
from perception.depth.decode import decode_sun  # noqa: E402
from perception.depth.estimate import AssocParams, estimate_box  # noqa: E402
from perception.detection.classes import COCO_NAMES  # noqa: E402
from perception.detection.runio import read_run  # noqa: E402
from perception.eval.localization import (BUCKETS, breakdown, error_stats, greedy_match, gt_objects, iou,  # noqa: E402
                                          link_2d_3d, match_detections, project_box, require_tuning_split, status_of,
                                          surface_point)
from perception.geometry.object3d import object_centre  # noqa: E402

OUT = REPO / "benchmarks" / "geometry"
RUN = REPO / "benchmarks" / "detection" / "sun_val_full.jsonl"
REVIEW = REPO / "docs" / "depth_tuning_review.yaml"

SHRINK = (0.3, 0.4, 0.5, 0.6, 0.7)
KS = (1.5, 2.0, 3.0)
BINS = (0.025, 0.05, 0.1)
MFRACS = (0.1, 0.15, 0.25)
N_MIN = 20
R5_STATUS = {"n_min": 20, "tau_cov": 0.3, "tau_spread_m": 0.25}  # placeholder thresholds used while choosing R5
TAU_COV = (0.2, 0.3, 0.4, 0.5)
TAU_SPREAD = (0.1, 0.15, 0.2, 0.25, 0.35, 0.5)
MIN_VALID, TIE_M, MIN_LOW = 0.6, 0.005, 30
SIGMA_FLOOR = 0.02  # not tuned (configs/depth.yaml centre.sigma_floor_m)


def est_keys() -> list[tuple]:
    keys = [("A", s, None, None) for s in SHRINK]
    keys += [("B", s, b, m) for s in SHRINK for b in BINS for m in MFRACS]
    return keys


def grid() -> list[dict]:
    return [{"estimator": e, "shrink": s, "bin_m": b, "m_frac": m, "k": k, "est_index": i}
            for i, (e, s, b, m) in enumerate(est_keys()) for k in KS]


# ----------------------------------------------------------------------------------------------- compute stage
def compute(split: str, limit: int | None, cache: Path) -> None:
    dcfg = load_config("dataset")
    root = data_root(dcfg) / "sunrgbd"
    sun_to_coco = yaml.safe_load((REPO / "configs" / "class_map.yaml").read_text())["sun_to_coco"]
    conf_op = float(load_config("detector")["conf_operational"])
    ids = read_split(REPO / "configs" / "splits" / f"{split}.txt")[:limit]
    anns = {a.image_id: a for a in load_annotations(root)}
    _, recs = read_run(RUN)
    dets = {r["frame_id"]: r for r in recs}
    keys, cfgs = est_keys(), grid()

    static = {k: [] for k in ("mode", "image_id", "sensor", "cls", "raw_cls", "gt", "half_depth", "link_iou",
                              "score", "box")}
    nv, fr, sp, cen = [], [], [], []  # per object: (n_est,), (n_est,), (n_est,), (n_cfg, 3)
    link = {"images": 0, "boxes3d": 0, "linked": 0, "linked_mapped": 0, "link_iou_ge_0.5": 0,
            "has3d_count_equal": 0, "has3d_pairs": 0, "has3d_pairs_iou_ge_0.5": 0, "has3d_agrees_with_projection": 0}
    det_stats = {"detections_at_conf": 0, "matched_any_gt": 0, "matched_linked": 0, "mapped_gt_2d": 0}
    t0 = time.perf_counter()
    for n_img, iid in enumerate(ids):
        ann = anns[iid]
        raw = cv2.imread(str(root / ann.depth_rel), cv2.IMREAD_UNCHANGED)
        depth, valid = decode_sun(raw, dcfg["depth"]["min_m"], dcfg["depth"]["max_m"])
        hw = depth.shape
        K = ann.K_np
        link["images"] += 1
        link["boxes3d"] += len(ann.boxes3d)

        # R1 verification on every val image: does the has3d order pair 2D and 3D boxes?
        pairs = link_2d_3d(ann, hw)
        link["linked"] += len(pairs)
        link["link_iou_ge_0.5"] += sum(v >= 0.5 for _, _, v in pairs)
        h3 = [i for i, b in enumerate(ann.boxes2d) if b.has3d]
        if len(h3) == len(ann.boxes3d):
            link["has3d_count_equal"] += 1
            chosen = {(i, j) for i, j, _ in pairs}
            for j, i in enumerate(h3):
                pb = project_box(ann.boxes3d[j].corners_cam(ann.Rtilt_np), K, hw)
                link["has3d_pairs"] += 1
                link["has3d_pairs_iou_ge_0.5"] += int(pb is not None and iou(ann.boxes2d[i].xyxy, pb) >= 0.5)
                link["has3d_agrees_with_projection"] += int((i, j) in chosen)

        gobjs, all2d = gt_objects(ann, hw, sun_to_coco)
        link["linked_mapped"] += len(gobjs)
        det_stats["mapped_gt_2d"] += len(all2d)
        items = [("oracle", np.asarray(o.box2d, np.float64), o, math.nan) for o in gobjs]
        r = dets.get(iid)
        if r is not None:
            b = np.asarray(r["boxes"], np.float64).reshape(-1, 4)
            s, c = np.asarray(r["scores"], np.float64), np.asarray(r["class_ids"], int)
            keep = s >= conf_op
            b, s, c = b[keep], s[keep], c[keep]
            det_stats["detections_at_conf"] += len(b)
            names = [COCO_NAMES[i] for i in c]
            matches = match_detections(b, s, names, gobjs, all2d)
            det_stats["matched_linked"] += len(matches)
            det_stats["matched_any_gt"] += len(greedy_match(b, s, names, all2d))
            items += [("detected", b[d], gobjs[k], float(s[d])) for d, k in matches]

        for mode, box, o, score in items:
            for key, val in (("mode", mode), ("image_id", iid), ("sensor", ann.sensor), ("cls", o.cls),
                             ("raw_cls", o.raw_cls), ("gt", o.centre), ("half_depth", o.half_depth),
                             ("link_iou", o.link_iou), ("score", score), ("box", box)):
                static[key].append(val)
            ests = []
            for e, s_, b_, m_ in keys:
                p = AssocParams(shrink=s_, estimator=e, bin_m=b_ or 0.05, m_frac=m_ or 0.15, n_min=N_MIN,
                                tau_cov=0.0, tau_spread_m=math.inf)
                ests.append(estimate_box(depth, valid, box, p))
            nv.append([x.n_valid for x in ests])
            fr.append([x.valid_frac for x in ests])
            sp.append([x.spread for x in ests])
            cen.append([object_centre(depth, valid, K, ests[c_["est_index"]], c_["k"], SIGMA_FLOOR).centre
                        for c_ in cfgs])
        if (n_img + 1) % 25 == 0:
            el = time.perf_counter() - t0
            print(f"{n_img + 1}/{len(ids)} images, {len(nv)} objects, {el:.0f}s "
                  f"(~{el / (n_img + 1) * (len(ids) - n_img - 1):.0f}s left)", flush=True)

    meta = {"split": split, "n_images": len(ids), "run": str(RUN.relative_to(REPO)).replace("\\", "/"),
            "conf_operational": conf_op, "linkage": link, "detections": det_stats, "configs": cfgs,
            "seconds": round(time.perf_counter() - t0, 1)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, meta=json.dumps(meta),
        mode=np.array(static["mode"]), image_id=np.array(static["image_id"]), sensor=np.array(static["sensor"]),
        cls=np.array(static["cls"]), raw_cls=np.array(static["raw_cls"]),
        gt=np.array(static["gt"], np.float64).reshape(-1, 3), half_depth=np.array(static["half_depth"]),
        link_iou=np.array(static["link_iou"]), score=np.array(static["score"]),
        box=np.array(static["box"], np.float64).reshape(-1, 4),
        n_valid=np.array(nv, np.int32).T, valid_frac=np.array(fr, np.float32).T, spread=np.array(sp, np.float32).T,
        centre=np.array(cen, np.float32).transpose(1, 0, 2))
    print(f"cached {len(nv)} objects x {len(cfgs)} configs -> {cache} ({meta['seconds']}s)")


# ----------------------------------------------------------------------------------------------- analyse stage
class Grid:
    def __init__(self, cache: Path):
        d = np.load(cache, allow_pickle=False)
        self.meta = json.loads(str(d["meta"]))
        self.cfgs = self.meta["configs"]
        for k in ("mode", "image_id", "sensor", "cls", "raw_cls", "gt", "half_depth", "link_iou", "score", "box",
                  "n_valid", "valid_frac", "spread", "centre"):
            setattr(self, k, d[k])
        self.centre = self.centre.astype(np.float64)
        self.gt_z = self.gt[:, 2]
        self.surf = np.array([surface_point(c, h) for c, h in zip(self.gt, self.half_depth)]).reshape(-1, 3)

    def errors(self, ci: int, surface: bool = False):
        ref = self.surf if surface else self.gt
        c = self.centre[ci]
        return np.linalg.norm(c - ref, axis=1), c[:, 2] - ref[:, 2]

    def status(self, ci: int, n_min: int, tau_cov: float, tau_spread: float) -> np.ndarray:
        e = self.cfgs[ci]["est_index"]
        return status_of(self.n_valid[e], self.valid_frac[e], np.nan_to_num(self.spread[e], nan=np.inf),
                         n_min, tau_cov, tau_spread)


def fractions(st: np.ndarray) -> dict:
    n = max(1, st.size)
    return {s: round(float((st == s).sum()) / n, 4) for s in ("valid", "low_coverage", "no_depth")} | {"n": int(st.size)}


def med(x) -> float:
    return float(np.median(x)) if np.size(x) else math.nan


def summarise_config(g: Grid, ci: int, th: dict) -> dict:
    err, dz = g.errors(ci)
    st = g.status(ci, th["n_min"], th["tau_cov"], th["tau_spread_m"])
    out = {k: v for k, v in g.cfgs[ci].items() if k != "est_index"}
    for mode in ("oracle", "detected"):
        m = g.mode == mode
        v = m & (st == "valid")
        out[mode] = {**fractions(st[m]), "median_err_valid_m": round(med(err[v]), 4),
                     "p90_err_valid_m": round(float(np.percentile(err[v], 90)), 4) if v.any() else None,
                     "z_signed_mean_valid_m": round(float(dz[v].mean()), 4) if v.any() else None}
    return out


def select_r5(rows: list[dict]) -> tuple[int, dict]:
    feas = [(i, r) for i, r in enumerate(rows) if r["oracle"]["valid"] >= MIN_VALID]
    if not feas:
        raise SystemExit("R5: no configuration reaches the 0.6 valid fraction; report this, do not relax silently")
    best = min(r["oracle"]["median_err_valid_m"] for _, r in feas)
    near = [(i, r) for i, r in feas if r["oracle"]["median_err_valid_m"] <= best + TIE_M]
    i, r = min(near, key=lambda t: (t[1]["estimator"] != "A", -t[1]["shrink"], t[1]["oracle"]["median_err_valid_m"]))
    return i, {"best_objective_m": best, "n_feasible": len(feas), "n_within_5mm": len(near),
               "chosen_objective_m": r["oracle"]["median_err_valid_m"], "chosen_valid_fraction": r["oracle"]["valid"],
               "best_overall_config": min(feas, key=lambda t: t[1]["oracle"]["median_err_valid_m"])[1]}


def sweep_r6(g: Grid, ci: int, min_low: int = MIN_LOW) -> tuple[dict, list[dict], dict]:
    err, _ = g.errors(ci)
    O = g.mode == "oracle"
    e = g.cfgs[ci]["est_index"]
    nvo = g.n_valid[e][O]
    has = nvo >= N_MIN
    small, big = has & (nvo < 50), nvo >= 50
    nmin_check = {"n_20_50": int(small.sum()), "median_err_20_50_m": round(med(err[O][small]), 4),
                  "n_ge_50": int(big.sum()), "median_err_ge_50_m": round(med(err[O][big]), 4)}
    ratio = nmin_check["median_err_20_50_m"] / nmin_check["median_err_ge_50_m"] if big.any() and small.any() else 0
    nmin_check["ratio"] = round(ratio, 3)
    n_min = 50 if ratio > 2 else N_MIN
    nmin_check["n_min"] = n_min
    rows = []
    for tc in TAU_COV:
        for ts in TAU_SPREAD:
            st = g.status(ci, n_min, tc, ts)[O]
            ev, el = err[O][st == "valid"], err[O][st == "low_coverage"]
            r = {"tau_cov": tc, "tau_spread_m": ts, **fractions(st), "n_low": int(el.size),
                 "median_err_valid_m": round(med(ev), 4), "median_err_low_m": round(med(el), 4)}
            r["ratio_low_over_valid"] = round(r["median_err_low_m"] / r["median_err_valid_m"], 3) \
                if ev.size and el.size else None
            r["feasible"] = r["valid"] >= MIN_VALID and el.size >= min_low
            rows.append(r)
    feas = [r for r in rows if r["feasible"]]
    if not feas:
        raise SystemExit("R6: no threshold pair keeps >= 60% valid with >= 30 low_coverage objects; report this")
    best = max(feas, key=lambda r: (r["ratio_low_over_valid"], r["valid"]))
    return {"n_min": n_min, "tau_cov": best["tau_cov"], "tau_spread_m": best["tau_spread_m"]}, rows, nmin_check


def final_report(g: Grid, ci: int, th: dict) -> dict:
    err, dz = g.errors(ci)
    err_s, dz_s = g.errors(ci, surface=True)
    st = g.status(ci, th["n_min"], th["tau_cov"], th["tau_spread_m"])
    out = {}
    for mode in ("oracle", "detected"):
        m = g.mode == mode
        v, lo = m & (st == "valid"), m & (st == "low_coverage")
        ev, el = med(err[v]), med(err[lo])
        out[mode] = {
            "status_fractions": fractions(st[m]),
            "valid": breakdown(err, dz, g.gt_z, g.sensor, v),
            "low_coverage": error_stats(err[lo], dz[lo]),
            "ratio_low_over_valid_median": round(el / ev, 3) if v.any() and lo.any() else None,
            "status_fractions_by_distance": {b: fractions(st[m & (np.array([_b(z) for z in g.gt_z]) == b)])
                                             for b, _, _ in BUCKETS},
            "analysis_only_surface_corrected_valid": breakdown(err_s, dz_s, g.gt_z, g.sensor, v),
            # sensitivity: weak projection links (low IoU) may pair the wrong 3D box; not a headline either
            "sensitivity_link_iou_ge_0.3_valid": error_stats(err[v & (g.link_iou >= 0.3)], dz[v & (g.link_iou >= 0.3)]),
            "link_iou_lt_0.3_fraction": round(float((g.link_iou[m] < 0.3).mean()), 4) if m.any() else None,
        }
    return out


def _b(z: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= z < hi:
            return name
    return "?"


def worst(g: Grid, ci: int, th: dict, n: int = 12) -> list[dict]:
    err, dz = g.errors(ci)
    st = g.status(ci, th["n_min"], th["tau_cov"], th["tau_spread_m"])
    idx = np.flatnonzero((g.mode == "oracle") & (st == "valid"))
    idx = idx[np.argsort(-err[idx])][:n]
    e = g.cfgs[ci]["est_index"]
    out = []
    for j in idx:
        x1, y1, x2, y2 = g.box[j]
        out.append({"key": f"{g.image_id[j]}:{g.raw_cls[j]}:{int(x1)}_{int(y1)}", "image_id": str(g.image_id[j]),
                    "sensor": str(g.sensor[j]), "class": str(g.cls[j]), "raw_class": str(g.raw_cls[j]),
                    "box": [round(float(v), 1) for v in g.box[j]], "gt_centre": [round(float(v), 3) for v in g.gt[j]],
                    "est_centre": [round(float(v), 3) for v in g.centre[ci][j]], "err3d_m": round(float(err[j]), 3),
                    "dz_m": round(float(dz[j]), 3), "half_depth_m": round(float(g.half_depth[j]), 3),
                    "n_valid": int(g.n_valid[e][j]), "valid_frac": round(float(g.valid_frac[e][j]), 3),
                    "spread_m": round(float(g.spread[e][j]), 3), "link_iou": round(float(g.link_iou[j]), 3)})
    return out


def render_gallery(items: list[dict], cfg: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from perception.depth.estimate import region_of
    dcfg = load_config("dataset")
    root = data_root(dcfg) / "sunrgbd"
    anns = {a.image_id: a for a in load_annotations(root)}
    fig, axes = plt.subplots(len(items), 2, figsize=(9, 3.1 * len(items)))
    axes = np.atleast_2d(axes)
    for row, it in zip(axes, items):
        ann = anns[it["image_id"]]
        rgb = cv2.cvtColor(cv2.imread(str(root / ann.rgb_rel), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        d, v = decode_sun(cv2.imread(str(root / ann.depth_rel), cv2.IMREAD_UNCHANGED),
                          dcfg["depth"]["min_m"], dcfg["depth"]["max_m"])
        x1, y1, x2, y2 = it["box"]
        c0, r0, c1, r1 = region_of(it["box"], d.shape, cfg["shrink"])
        dv = np.where(v, d, np.nan)
        for ax, img, kw in ((row[0], rgb, {}), (row[1], dv, {"cmap": "turbo"})):
            ax.imshow(img, **kw)
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=1.5))
            ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec="yellow", lw=1, ls="--"))
            pad = 0.6 * max(x2 - x1, y2 - y1)
            ax.set_xlim(max(0, x1 - pad), min(img.shape[1], x2 + pad))
            ax.set_ylim(min(img.shape[0], y2 + pad), max(0, y1 - pad))
            ax.axis("off")
        row[0].set_title(f"{it['image_id']} {it['raw_class']} ({it['sensor']})", fontsize=8)
        row[1].set_title(f"err {it['err3d_m']:.2f} m, dz {it['dz_m']:+.2f}, GT Z {it['gt_centre'][2]:.2f}, "
                         f"n {it['n_valid']}, frac {it['valid_frac']:.2f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def write_depth_yaml(cfg: dict, th: dict, sel: dict) -> None:
    src = (f"spec 03.3 on SUN RGB-D val (scripts/tune_depth.py; benchmarks/geometry/tune_val.json). "
           f"Oracle median 3D centre error {sel['chosen_objective_m'] * 100:.1f} cm at valid fraction "
           f"{sel['chosen_valid_fraction']:.2f} (R5 iteration 2, at the iteration-1 R6 thresholds).")
    text = f"""# Depth association and object geometry (specs 03.1, 03.2). Values chosen by {src}
# Frozen for Phase 05; do not tune on test.

association:
  shrink: {cfg['shrink']}            # box shrink factor about its centre (width and height both x shrink)
  estimator: {cfg['estimator']}           # A = median of valid depths; B = nearest significant histogram mode
  bin_m: {cfg['bin_m'] if cfg['bin_m'] is not None else 0.05}            # estimator B histogram bin width (m){'' if cfg['estimator'] == 'B' else '; unused by A'}
  m_frac: {cfg['m_frac'] if cfg['m_frac'] is not None else 0.15}           # estimator B: significant-bin fraction{'' if cfg['estimator'] == 'B' else '; unused by A'}

status:
  n_min: {th['n_min']}              # fewer valid pixels -> no_depth
  tau_cov: {th['tau_cov']}            # valid fraction below this -> low_coverage
  tau_spread_m: {th['tau_spread_m']}       # spread (1.4826 * MAD) above this -> low_coverage

centre:
  k: {cfg['k']}                 # inlier band: |z - estimate| <= k * max(spread, sigma_floor)
  sigma_floor_m: {SIGMA_FLOOR}    # floor on the band width (m); fixed, not tuned
"""
    (REPO / "configs" / "depth.yaml").write_text(text, encoding="utf-8")


def fmt_stats(s: dict) -> str:
    if not s.get("n"):
        return "| 0 | – | – | – | – | – | – |"
    return (f"| {s['n']} | {s['centre_median_m'] * 100:.1f} | {s['centre_mean_m'] * 100:.1f} | "
            f"{s['centre_p90_m'] * 100:.1f} | {s['z_mae_m'] * 100:.1f} | {s['z_rmse_m'] * 100:.1f} | "
            f"{s['z_signed_mean_m'] * 100:+.1f} |")


def write_doc(rep: dict, gallery_rel: str) -> None:
    review = yaml.safe_load(REVIEW.read_text(encoding="utf-8")) if REVIEW.exists() else {}
    review = review or {}
    c, th, sel, meta = rep["chosen"], rep["thresholds"]["chosen"], rep["selection"], rep["meta"]
    L = ["<!-- GENERATED by scripts/tune_depth.py from benchmarks/geometry/tune_val.json. Do not edit by hand; "
         "gallery categories live in docs/depth_tuning_review.yaml. -->", "",
         "# Depth parameter tuning on SUN RGB-D val (spec 03.3)", "",
         f"**Measured** on `{meta['split']}` ({meta['n_images']} images). Test split not used (R8 guard). "
         "All errors in cm. Status rule and estimator: spec 03.1; centres: spec 03.2.", "",
         "## Linkage (R1)", ""]
    lk = meta["linkage"]
    L += [f"- `has3d` order check over all val images: counts equal in {lk['has3d_count_equal']}/{lk['images']} "
          f"images; projected-3D vs 2D IoU ≥ 0.5 for {lk['has3d_pairs_iou_ge_0.5']}/{lk['has3d_pairs']} pairs "
          f"(fixed check fails), and the has3d pair is the best projection match for "
          f"{lk['has3d_agrees_with_projection']}/{lk['has3d_pairs']}.",
          "- 2D GT boxes are visible extents, projected 3D boxes are amodal, so IoU is often < 0.5 for a correct "
          "pair. Per R1 the **projection linkage** is used: same class, Hungarian on IoU, IoU > 0.",
          f"- Linked: {lk['linked']} of {lk['boxes3d']} 3D boxes; {lk['linked_mapped']} in mapped (COCO) classes "
          f"→ oracle objects. Detected mode: {meta['detections']['matched_linked']} detections at conf "
          f"{meta['conf_operational']} matched to a linked object (of {meta['detections']['detections_at_conf']}).",
          "", "## Grid choice (R5)", "",
          "R5 ran twice (review decision 2026-11-11). Iteration 1 used placeholder thresholds (tau_cov 0.3, "
          "tau_spread 0.25 m); its R6 then chose looser ones, under which configurations excluded in iteration 1 "
          "became feasible. Iteration 2 re-ran R5 at the iteration-1 R6 thresholds, then R6 again. One extra pass "
          "only, all on val.", "",
          f"- Iteration 1: estimator {rep['iteration1']['chosen']['estimator']}, shrink "
          f"{rep['iteration1']['chosen']['shrink']}, k {rep['iteration1']['chosen']['k']} → "
          f"{rep['iteration1']['chosen_objective_m'] * 100:.1f} cm at valid "
          f"{rep['iteration1']['chosen_valid_fraction']:.2f}; R6 → tau_cov "
          f"{rep['iteration1']['r6_thresholds']['tau_cov']}, tau_spread "
          f"{rep['iteration1']['r6_thresholds']['tau_spread_m']} m.",
          f"- R6 thresholds after iteration 2 "
          f"{'unchanged' if rep['iteration1']['r6_stable_in_iteration2'] else '**changed** (reported, not iterated)'}.",
          "", "Iteration 2:", "",
          f"{len(rep['grid'])} configurations. Objective: oracle median 3D error over `valid` objects, valid "
          f"fraction ≥ 0.6, ties within 5 mm → A, then larger shrink.", "",
          f"- **Chosen:** estimator {c['estimator']}, shrink {c['shrink']}, k {c['k']}"
          + (f", bin {c['bin_m']} m, m_frac {c['m_frac']}" if c["estimator"] == "B" else "")
          + f" → {sel['chosen_objective_m'] * 100:.1f} cm at valid fraction {sel['chosen_valid_fraction']:.2f}.",
          f"- Best feasible objective {sel['best_objective_m'] * 100:.1f} cm; {sel['n_within_5mm']} of "
          f"{sel['n_feasible']} feasible configurations within 5 mm.", "",
          "Best configuration per estimator × shrink (oracle, iteration-2 thresholds):", "",
          "| est | shrink | k | bin | m_frac | valid | median err | p90 err | signed dz | det. median err |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for est in ("A", "B"):
        for s in SHRINK:
            rows = [r for r in rep["grid"] if r["estimator"] == est and r["shrink"] == s]
            r = min(rows, key=lambda r: r["oracle"]["median_err_valid_m"])
            o, d = r["oracle"], r["detected"]
            L.append(f"| {est} | {s} | {r['k']} | {r['bin_m'] or '–'} | {r['m_frac'] or '–'} | {o['valid']:.2f} | "
                     f"{o['median_err_valid_m'] * 100:.1f} | {o['p90_err_valid_m'] * 100:.1f} | "
                     f"{o['z_signed_mean_valid_m'] * 100:+.1f} | {d['median_err_valid_m'] * 100:.1f} |")
    nm = rep["thresholds"]["n_min_check"]
    L += ["", "## Status thresholds (R6)", "",
          f"- n_min check: objects with 20–50 valid pixels: median {nm['median_err_20_50_m'] * 100:.1f} cm "
          f"(n={nm['n_20_50']}) vs ≥ 50: {nm['median_err_ge_50_m'] * 100:.1f} cm (n={nm['n_ge_50']}); ratio "
          f"{nm['ratio']} → n_min = {nm['n_min']}.",
          f"- **Chosen:** tau_cov {th['tau_cov']}, tau_spread {th['tau_spread_m']} m (max low/valid median-error "
          f"ratio with valid ≥ 0.6 and ≥ 30 low_coverage).", "",
          "| tau_cov | tau_spread | valid | low | no_depth | n_low | med err valid | med err low | ratio |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rep["thresholds"]["sweep"]:
        mark = " **←**" if (r["tau_cov"], r["tau_spread_m"]) == (th["tau_cov"], th["tau_spread_m"]) else ""
        L.append(f"| {r['tau_cov']} | {r['tau_spread_m']} | {r['valid']:.2f} | {r['low_coverage']:.2f} | "
                 f"{r['no_depth']:.2f} | {r['n_low']} | {r['median_err_valid_m'] * 100:.1f} | "
                 f"{r['median_err_low_m'] * 100:.1f} | {r['ratio_low_over_valid']}{mark} |")
    L += ["", "## Final numbers on val (chosen configuration and thresholds)", ""]
    hdr = ["| subset | n | median | mean | p90 | Z MAE | Z RMSE | signed Z |", "|---|---|---|---|---|---|---|---|"]
    for mode in ("oracle", "detected"):
        f = rep["final"][mode]
        sf = f["status_fractions"]
        L += [f"### {mode.capitalize()} boxes", "",
              f"Status: valid {sf['valid']:.2f}, low_coverage {sf['low_coverage']:.2f}, no_depth "
              f"{sf['no_depth']:.2f} (n={sf['n']}). **low_coverage / valid median error = "
              f"{f['ratio_low_over_valid_median']}** (AC3).", ""] + hdr
        v = f["valid"]
        L.append("| valid, all " + fmt_stats(v["all"]))
        for b, _, _ in BUCKETS:
            L.append(f"| valid, Z {b} m " + fmt_stats(v["by_distance"][b]))
        for s_, st in v["by_sensor"].items():
            L.append(f"| valid, {s_} " + fmt_stats(st))
        L.append("| low_coverage " + fmt_stats(f["low_coverage"]))
        L.append("")
        sc = f["analysis_only_surface_corrected_valid"]["all"]
        L += [f"*Analysis only (R4), not a headline:* against the GT centre moved half the box depth toward the "
              f"camera, valid median error {sc['centre_median_m'] * 100:.1f} cm, signed Z "
              f"{sc['z_signed_mean_m'] * 100:+.1f} cm (vs {v['all']['z_signed_mean_m'] * 100:+.1f} cm against the "
              f"volume centre).", ""]
    L += ["## Bias (R4, AC4)", "",
          "Signed Z error is estimate − GT. The estimate is a visible-surface point and the GT a volume centre, so "
          "a negative sign (estimate nearer than the centre) is expected; its size is reported above per mode.", "",
          "## Worst 12 val objects (oracle, valid)", "", f"![worst 12]({gallery_rel})", "",
          "| # | image | class | sensor | err | dz | GT Z | n | frac | spread | category | note |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, it in enumerate(rep["worst12"], 1):
        rv = review.get(it["key"], {}) or {}
        L.append(f"| {i} | {it['image_id']} | {it['raw_class']} | {it['sensor']} | {it['err3d_m'] * 100:.0f} | "
                 f"{it['dz_m'] * 100:+.0f} | {it['gt_centre'][2]:.2f} | {it['n_valid']} | {it['valid_frac']:.2f} | "
                 f"{it['spread_m'] * 100:.0f} | {rv.get('category', '*not yet reviewed*')} | {rv.get('note', '')} |")
    L.append("")
    (REPO / "docs" / "depth_tuning.md").write_text("\n".join(L), encoding="utf-8")


def analyse(cache: Path, smoke: bool) -> None:
    g = Grid(cache)
    min_low = 3 if smoke else MIN_LOW  # smoke: too few objects for 30
    # Iteration 1: R5 under the placeholder thresholds, then R6.
    rows1 = [summarise_config(g, i, R5_STATUS) for i in range(len(g.cfgs))]
    ci1, sel1 = select_r5(rows1)
    th1, _, _ = sweep_r6(g, ci1, min_low)
    # Iteration 2 (review decision 2026-11-11): R5 again under the R6 thresholds, then R6 again. Exactly one extra
    # pass, no loop to convergence; if R6 moves again that is reported, not iterated.
    rows = [summarise_config(g, i, th1) for i in range(len(g.cfgs))]
    ci, sel = select_r5(rows)
    th, sweep, nmin_check = sweep_r6(g, ci, min_low)
    iteration1 = {"chosen": {k: v for k, v in g.cfgs[ci1].items() if k != "est_index"},
                  "chosen_objective_m": sel1["chosen_objective_m"],
                  "chosen_valid_fraction": sel1["chosen_valid_fraction"], "r5_thresholds": R5_STATUS,
                  "r6_thresholds": th1, "r6_stable_in_iteration2": th == th1}
    chosen = {k: v for k, v in g.cfgs[ci].items() if k != "est_index"}
    rep = {"spec": "03.3", "label": "Measured on SUN RGB-D val; parameters chosen here are frozen for Phase 05",
           "meta": {k: v for k, v in g.meta.items() if k != "configs"},
           "rules": {"r5": "min oracle median 3D error over valid, valid fraction >= 0.6; ties within 5 mm -> A, then "
                           "larger shrink. Iteration 1 at n_min 20 / tau_cov 0.3 / tau_spread 0.25, iteration 2 at "
                           "the iteration-1 R6 thresholds",
                     "r6": "max median err(low_coverage)/median err(valid), valid >= 0.6, n_low >= 30; n_min 20 "
                           "unless 20-50 px objects have > 2x the median error of >= 50 px objects"},
           "iteration1": iteration1, "grid": rows, "selection": sel, "chosen": chosen,
           "thresholds": {"chosen": th, "sweep": sweep, "n_min_check": nmin_check},
           "final": final_report(g, ci, th), "worst12": worst(g, ci, th)}
    suffix = "_smoke" if smoke else ""
    (OUT / f"tune_val{suffix}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    gallery = OUT / f"depth_tuning_worst12{suffix}.png"
    render_gallery(rep["worst12"], chosen, gallery)
    print(json.dumps({"chosen": chosen, "selection": {k: v for k, v in sel.items() if k != "best_overall_config"},
                      "thresholds": th, "n_min_check": nmin_check,
                      "oracle": {k: rep["final"]["oracle"][k] for k in ("status_fractions",
                                                                         "ratio_low_over_valid_median")},
                      "oracle_valid_all": rep["final"]["oracle"]["valid"]["all"],
                      "detected_valid_all": rep["final"]["detected"]["valid"]["all"]}, indent=1))
    if smoke:
        print("smoke run: configs/depth.yaml and docs/depth_tuning.md NOT written")
        return
    write_depth_yaml(chosen, th, sel)
    write_doc(rep, "../" + str(gallery.relative_to(REPO)).replace("\\", "/"))
    print("wrote benchmarks/geometry/tune_val.json, configs/depth.yaml, docs/depth_tuning.md, " + gallery.name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--stage", choices=("all", "compute", "analyse"), default="all")
    ap.add_argument("--limit", type=int, default=None, help="first N images only (smoke run; writes *_smoke files)")
    a = ap.parse_args(argv)
    try:
        require_tuning_split(a.split)
    except ValueError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    smoke = a.limit is not None
    cache = OUT / f"tune_val_objects{'_smoke' if smoke else ''}.npz"
    if a.stage in ("all", "compute"):
        compute(a.split, a.limit, cache)
    if a.stage in ("all", "analyse"):
        analyse(cache, smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
