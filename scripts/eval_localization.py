"""Spec 05.2: 3D localization on SUN RGB-D test (G3, G4), once, with the frozen Phase 03 settings.

  python -I scripts/eval_localization.py --split sunrgbd_val                 # check: must reproduce 03.3 val
  python -I scripts/eval_localization.py --split sunrgbd_test --limit 20     # smoke, writes *_smoke files only
  python -I scripts/eval_localization.py --split sunrgbd_test                # THE one-shot headline run
  python -I scripts/eval_localization.py --split sunrgbd_test --doc-only     # regenerate the doc from the JSON
  python -I scripts/eval_localization.py --split sunrgbd_test --rerun-reason "..."   # logged re-run

Nothing is chosen here. Refuses to run if a frozen input (protocol section 9) or the protocol itself changed.
Modes (protocol section 3): oracle = GT 2D boxes drive depth association; detected = stored YOLO26-n detections at
conf_operational, matched greedy-by-score to same-class GT 2D boxes at IoU >= 0.5 (03.3 code, unchanged).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
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
from perception.eval.localization import (BUCKETS, breakdown, bucket_of, error_stats, greedy_match,  # noqa: E402
                                          gt_objects, match_detections, surface_point)
from perception.eval.results_index import (PROTOCOL, check_frozen, check_protocol, git_state,  # noqa: E402
                                           sha256_file)
from perception.eval.sun_test import (STATUSES, bootstrap_ci, bootstrap_ratio, check_status_complete,  # noqa: E402
                                      log_rerun, one_shot_guard)
from perception.geometry.object3d import object_centre  # noqa: E402

OUT = REPO / "benchmarks" / "results"
INDEX = OUT / "index.json"
VAL_REF = REPO / "benchmarks" / "geometry" / "tune_val.json"
DOC = REPO / "docs" / "localization_results.md"
SPLITS = {  # split -> (output stem, stored detection run, headline?)
    "sunrgbd_test": ("localization_sun_test", REPO / "benchmarks" / "detection" / "sun_test_full.jsonl", True),
    "sunrgbd_val": ("localization_sun_val_check", REPO / "benchmarks" / "detection" / "sun_val_full.jsonl", False),
}
FROZEN = ["configs/depth.yaml", "configs/detector.yaml", "configs/class_map.yaml"]
N_BOOT, SEED = 1000, 0
ORACLE_C, DETECTED_C = "#2a78d6", "#eb6834"  # dataviz categorical slots 1, 2 (light surface)


def rel(p: Path) -> str:
    return p.relative_to(REPO).as_posix()


# ------------------------------------------------------------------------------------------------------ compute
def compute(split: str, limit: int | None, run: Path) -> tuple[dict, dict, dict]:
    dcfg = load_config("dataset")
    root = data_root(dcfg) / "sunrgbd"
    sun_to_coco = yaml.safe_load((REPO / "configs" / "class_map.yaml").read_text())["sun_to_coco"]
    dep = load_config("depth")
    p = AssocParams.from_config(dep)
    k, floor = float(dep["centre"]["k"]), float(dep["centre"]["sigma_floor_m"])
    conf_op = float(load_config("detector")["conf_operational"])
    ids = read_split(REPO / "configs" / "splits" / f"{split}.txt")[:limit]
    anns = {a.image_id: a for a in load_annotations(root)}
    run_header, recs = read_run(run)
    dets = {r["frame_id"]: r for r in recs}

    O = {k_: [] for k_ in ("mode", "image_id", "sensor", "cls", "raw_cls", "gt", "half_depth", "link_iou", "score",
                           "box", "z", "n_valid", "valid_frac", "spread", "status", "centre")}
    B = {k_: [] for k_ in ("image_id", "sensor", "cls", "score", "box", "status", "matched_linked", "matched_any")}
    counts = {"images": 0, "images_without_run_record": 0, "boxes3d": 0, "linked_mapped": 0, "mapped_gt_2d": 0,
              "detections_at_conf": 0, "matched_any_gt": 0, "matched_linked": 0}
    t0 = time.perf_counter()
    for n_img, iid in enumerate(ids):
        ann = anns[iid]
        depth, valid = decode_sun(cv2.imread(str(root / ann.depth_rel), cv2.IMREAD_UNCHANGED),
                                  dcfg["depth"]["min_m"], dcfg["depth"]["max_m"])
        K = ann.K_np
        counts["images"] += 1
        counts["boxes3d"] += len(ann.boxes3d)
        gobjs, all2d = gt_objects(ann, depth.shape, sun_to_coco)
        counts["linked_mapped"] += len(gobjs)
        counts["mapped_gt_2d"] += len(all2d)
        items = [("oracle", np.asarray(o.box2d, np.float64), o, math.nan, None) for o in gobjs]

        r = dets.get(iid)
        if r is None:
            counts["images_without_run_record"] += 1
        else:
            b = np.asarray(r["boxes"], np.float64).reshape(-1, 4)
            s, c = np.asarray(r["scores"], np.float64), np.asarray(r["class_ids"], int)
            keep = s >= conf_op
            b, s, c = b[keep], s[keep], c[keep]
            names = [COCO_NAMES[i] for i in c]
            matches = match_detections(b, s, names, gobjs, all2d)
            any_m = {d for d, _ in greedy_match(b, s, names, all2d)}
            linked_m = {d: kk for d, kk in matches}
            counts["detections_at_conf"] += len(b)
            counts["matched_any_gt"] += len(any_m)
            counts["matched_linked"] += len(matches)
            # G4: every output box at conf_operational gets a depth status, matched or not.
            for d in range(len(b)):
                est = estimate_box(depth, valid, b[d], p)
                for key, val in (("image_id", iid), ("sensor", ann.sensor), ("cls", names[d]),
                                 ("score", float(s[d])), ("box", b[d]), ("status", est.status),
                                 ("matched_linked", d in linked_m), ("matched_any", d in any_m)):
                    B[key].append(val)
                if d in linked_m:
                    items.append(("detected", b[d], gobjs[linked_m[d]], float(s[d]), est))

        for mode, box, o, score, est in items:
            est = est or estimate_box(depth, valid, box, p)
            obj = object_centre(depth, valid, K, est, k, floor)
            for key, val in (("mode", mode), ("image_id", iid), ("sensor", ann.sensor), ("cls", o.cls),
                             ("raw_cls", o.raw_cls), ("gt", o.centre), ("half_depth", o.half_depth),
                             ("link_iou", o.link_iou), ("score", score), ("box", box), ("z", est.z),
                             ("n_valid", est.n_valid), ("valid_frac", est.valid_frac), ("spread", est.spread),
                             ("status", est.status), ("centre", obj.centre)):
                O[key].append(val)
        if (n_img + 1) % 250 == 0:
            el = time.perf_counter() - t0
            print(f"{n_img + 1}/{len(ids)} images, {len(O['mode'])} objects, {el:.0f}s "
                  f"(~{el / (n_img + 1) * (len(ids) - n_img - 1):.0f}s left)", flush=True)

    objs = {"mode": np.array(O["mode"]), "image_id": np.array(O["image_id"]), "sensor": np.array(O["sensor"]),
            "cls": np.array(O["cls"]), "raw_cls": np.array(O["raw_cls"]),
            "gt": np.array(O["gt"], np.float64).reshape(-1, 3), "half_depth": np.array(O["half_depth"], np.float64),
            "link_iou": np.array(O["link_iou"], np.float64), "score": np.array(O["score"], np.float64),
            "box": np.array(O["box"], np.float64).reshape(-1, 4), "z": np.array(O["z"], np.float64),
            "n_valid": np.array(O["n_valid"], np.int32), "valid_frac": np.array(O["valid_frac"], np.float64),
            "spread": np.array(O["spread"], np.float64), "status": np.array(O["status"]),
            "centre": np.array(O["centre"], np.float64).reshape(-1, 3)}
    boxes = {"image_id": np.array(B["image_id"]), "sensor": np.array(B["sensor"]), "cls": np.array(B["cls"]),
             "score": np.array(B["score"], np.float64), "box": np.array(B["box"], np.float64).reshape(-1, 4),
             "status": np.array(B["status"]), "matched_linked": np.array(B["matched_linked"], bool),
             "matched_any": np.array(B["matched_any"], bool)}
    meta = {"split": split, "n_images": len(ids), "run": rel(run), "run_header": run_header,
            "conf_operational": conf_op, "assoc_params": p.__dict__, "centre": {"k": k, "sigma_floor_m": floor},
            "counts": counts, "seconds": round(time.perf_counter() - t0, 1)}
    return objs, boxes, meta


# ------------------------------------------------------------------------------------------------------- report
def fractions(st: np.ndarray) -> dict:
    n = max(1, st.size)
    return {s: round(float((st == s).sum()) / n, 4) for s in STATUSES} | {"n": int(st.size)}


def med(x) -> float:
    return float(np.median(x)) if np.size(x) else math.nan


def mode_report(o: dict, mode: str) -> dict:
    err = np.linalg.norm(o["centre"] - o["gt"], axis=1)
    dz = o["centre"][:, 2] - o["gt"][:, 2]
    surf = np.array([surface_point(c, h) for c, h in zip(o["gt"], o["half_depth"])]).reshape(-1, 3)
    err_s = np.linalg.norm(o["centre"] - surf, axis=1)
    dz_s = o["centre"][:, 2] - surf[:, 2]
    gz, st, sensor, img = o["gt"][:, 2], o["status"], o["sensor"], o["image_id"]
    m = o["mode"] == mode
    v, lo = m & (st == "valid"), m & (st == "low_coverage")
    bk = np.array([bucket_of(z) for z in gz]) if gz.size else np.zeros(0, str)
    ev, el = med(err[v]), med(err[lo])
    return {
        "status_fractions": fractions(st[m]),
        "status_fractions_by_distance": {b: fractions(st[m & (bk == b)]) for b, _, _ in BUCKETS},
        "status_fractions_by_sensor": {s: fractions(st[m & (sensor == s)]) for s in sorted(set(sensor[m].tolist()))},
        "valid": breakdown(err, dz, gz, sensor, v),
        "low_coverage": error_stats(err[lo], dz[lo]),
        "low_coverage_by_distance": {b: error_stats(err[lo & (bk == b)], dz[lo & (bk == b)]) for b, _, _ in BUCKETS},
        "ratio_low_over_valid_median": round(el / ev, 3) if v.any() and lo.any() else None,
        "bootstrap_valid": {"n_boot": N_BOOT, "seed": SEED, "unit": "image",
                            **bootstrap_ci(err[v], img[v], n_boot=N_BOOT, seed=SEED)},
        "bootstrap_ratio_low_over_valid": bootstrap_ratio(err[m], lo[m], v[m], img[m], n_boot=N_BOOT, seed=SEED),
        "analysis_only_surface_corrected_valid": breakdown(err_s, dz_s, gz, sensor, v),
        "sensitivity_link_iou_ge_0.3_valid": error_stats(err[v & (o["link_iou"] >= 0.3)],
                                                         dz[v & (o["link_iou"] >= 0.3)]),
        "link_iou_lt_0.3_fraction": round(float((o["link_iou"][m] < 0.3).mean()), 4) if m.any() else None,
    }


def val_vs_test(rep: dict) -> dict | None:
    if not VAL_REF.exists():
        return None
    val = json.loads(VAL_REF.read_text(encoding="utf-8"))["final"]
    out = {}
    for mode in ("oracle", "detected"):
        a, b = val[mode], rep[mode]
        row = {}
        for name, x in (("val", a), ("test", b)):
            va = x["valid"]["all"]
            row[name] = {"n_objects": x["status_fractions"]["n"], "valid_fraction": x["status_fractions"]["valid"],
                         "low_coverage_fraction": x["status_fractions"]["low_coverage"],
                         "no_depth_fraction": x["status_fractions"]["no_depth"],
                         "median_m": va.get("centre_median_m"), "p90_m": va.get("centre_p90_m"),
                         "z_signed_mean_m": va.get("z_signed_mean_m"),
                         "ratio_low_over_valid": x["ratio_low_over_valid_median"],
                         "valid_share_by_sensor": {s: round(st["n"] / max(1, va["n"]), 3)
                                                   for s, st in x["valid"]["by_sensor"].items()},
                         "median_m_by_distance": {d: st.get("centre_median_m")
                                                  for d, st in x["valid"]["by_distance"].items()}}
        row["test_minus_val_median_m"] = round(row["test"]["median_m"] - row["val"]["median_m"], 4)
        out[mode] = row
    return out


def build_report(objs: dict, boxes: dict, meta: dict, split: str, headline: bool, smoke: bool) -> dict:
    g4 = check_status_complete(boxes["status"].tolist() + objs["status"].tolist())  # raises on any gap
    rep = {"spec": "05.2",
           "label": ("Measured on SUN RGB-D test (headline, G3/G4)" if headline and not smoke else
                     "Measured; smoke run, not a result" if smoke else
                     "Measured on SUN RGB-D val; code check against 03.3, not a headline"),
           "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           **git_state(REPO),
           "protocol_sha256": sha256_file(REPO / PROTOCOL),
           "config_sha256": {Path(f).stem: sha256_file(REPO / f) for f in FROZEN},
           "split_sha256": sha256_file(REPO / "configs" / "splits" / f"{split}.txt"),
           "meta": meta,
           "g4_status_coverage": {**g4, "scope": "every output box at conf_operational plus every oracle box"},
           "output_boxes_status_fractions": fractions(boxes["status"]),
           "detected_recall_of_linked_objects": round(int((objs["mode"] == "detected").sum())
                                                      / max(1, int((objs["mode"] == "oracle").sum())), 4),
           "oracle": mode_report(objs, "oracle"), "detected": mode_report(objs, "detected")}
    rep["val_vs_test"] = val_vs_test(rep) if split == "sunrgbd_test" else None
    if split == "sunrgbd_val" and VAL_REF.exists():
        ref = json.loads(VAL_REF.read_text(encoding="utf-8"))["final"]
        rep["reproduces_03_3"] = {
            mode: {"this": rep[mode]["valid"]["all"], "tune_val": ref[mode]["valid"]["all"],
                   "status_this": rep[mode]["status_fractions"], "status_tune_val": ref[mode]["status_fractions"],
                   "equal": rep[mode]["valid"]["all"] == ref[mode]["valid"]["all"]
                   and rep[mode]["status_fractions"] == ref[mode]["status_fractions"]}
            for mode in ("oracle", "detected")}
    return rep


# ---------------------------------------------------------------------------------------------------- figures/doc
def figures(objs: dict, rep: dict, stem: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    paths = {}
    # Error vs distance: median (filled) and p90 (hollow) per bucket, both modes, n under each bucket.
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    names = [b for b, _, _ in BUCKETS]
    x = np.arange(len(names))
    for off, mode, col in ((-0.12, "oracle", ORACLE_C), (0.12, "detected", DETECTED_C)):
        bd = rep[mode]["valid"]["by_distance"]
        mdn = [bd[b].get("centre_median_m", np.nan) * 100 for b in names]
        p90 = [bd[b].get("centre_p90_m", np.nan) * 100 for b in names]
        ax.plot(x + off, mdn, "o-", color=col, lw=2, ms=8, mec="#fcfcfb", mew=2, label=f"{mode} median")
        ax.plot(x + off, p90, "o--", color=col, lw=1.2, ms=8, mfc="#fcfcfb", mew=2, label=f"{mode} p90")
        for xi, (b, yv) in enumerate(zip(names, mdn)):
            ax.annotate(f"{yv:.0f}", (xi + off, yv), textcoords="offset points", xytext=(0, -14), ha="center",
                        fontsize=8, color="#0b0b0b")
    ax.set_xticks(x, [f"{b} m\n(n={rep['oracle']['valid']['by_distance'][b]['n']} / "
                      f"{rep['detected']['valid']['by_distance'][b]['n']})" for b in names])
    ax.set_ylabel("3D centre error (cm), valid objects")
    ax.set_xlabel("GT distance bucket (n oracle / detected)")
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    paths["distance"] = OUT / f"{stem}_error_vs_distance.png"
    fig.savefig(paths["distance"], dpi=110)
    plt.close(fig)

    # CDF of the 3D centre error, valid objects (solid) and low_coverage objects (dashed), per mode.
    err = np.linalg.norm(objs["centre"] - objs["gt"], axis=1)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for mode, col in (("oracle", ORACLE_C), ("detected", DETECTED_C)):
        for st, ls, lw in (("valid", "-", 2), ("low_coverage", "--", 1.2)):
            e = np.sort(err[(objs["mode"] == mode) & (objs["status"] == st)]) * 100
            if e.size:
                ax.step(e, np.arange(1, e.size + 1) / e.size, where="post", color=col, ls=ls, lw=lw,
                        label=f"{mode}, {st} (n={e.size})")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 1)
    ax.set_xlabel("3D centre error (cm), clipped at 200")
    ax.set_ylabel("fraction of objects")
    ax.grid(color="#e4e3df", lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    paths["cdf"] = OUT / f"{stem}_error_cdf.png"
    fig.savefig(paths["cdf"], dpi=110)
    plt.close(fig)
    return {k: rel(v) for k, v in paths.items()}


def cm(x, sign=False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x * 100:+.1f}" if sign else f"{x * 100:.1f}"


def fmt_stats(s: dict) -> str:
    if not s.get("n"):
        return "| 0 | – | – | – | – | – | – |"
    return (f"| {s['n']} | {cm(s['centre_median_m'])} | {cm(s['centre_mean_m'])} | {cm(s['centre_p90_m'])} | "
            f"{cm(s['z_mae_m'])} | {cm(s['z_rmse_m'])} | {cm(s['z_signed_mean_m'], True)} |")


def write_doc(rep: dict, figs: dict, json_rel: str) -> None:
    m = rep["meta"]
    c = m["counts"]
    L = [f"<!-- GENERATED by scripts/eval_localization.py from {json_rel}. Do not edit by hand. -->", "",
         "# 3D localization on SUN RGB-D test (spec 05.2; goals G3, G4)", "",
         f"**Measured**, one-shot, on `{m['split']}` ({m['n_images']} images), {rep['created_utc']}. All errors in "
         "cm. Nothing was tuned here: depth settings frozen on val (03.3), detections from the stored 02.2 run.", "",
         "## Provenance", "",
         f"- Protocol `docs/evaluation_protocol.md` sha256 `{rep['protocol_sha256'][:12]}…` (frozen, checked).",
         "- Frozen inputs checked before the run: " + ", ".join(f"`{k}.yaml` `{v[:12]}…`"
                                                                 for k, v in rep["config_sha256"].items())
         + f", split `{rep['split_sha256'][:12]}…`.",
         f"- Detections: `{m['run']}` at conf_operational {m['conf_operational']}; run header weights sha "
         f"`{m['run_header']['weights_sha256'][:12]}…`, head {m['run_header']['head']}, imgsz "
         f"{m['run_header']['imgsz']}.",
         f"- Depth: estimator {m['assoc_params']['estimator']}, shrink {m['assoc_params']['shrink']}, n_min "
         f"{m['assoc_params']['n_min']}, tau_cov {m['assoc_params']['tau_cov']}, tau_spread "
         f"{m['assoc_params']['tau_spread_m']} m, centre k {m['centre']['k']}.",
         f"- git commit: {rep['git_commit'] or 'none (repository has no commits yet)'}; dirty {rep['git_dirty']}.",
         "", "## Coverage", "",
         f"- 3D boxes {c['boxes3d']}; linked GT objects in mapped classes (oracle set) {c['linked_mapped']}.",
         f"- Detections at conf_operational {c['detections_at_conf']}; matched to any mapped GT 2D box "
         f"{c['matched_any_gt']}; matched to a linked GT object (detected set) {c['matched_linked']} → "
         f"{rep['detected_recall_of_linked_objects']:.2f} of the oracle set.",
         f"- **G4:** {rep['g4_status_coverage']['with_status']} of {rep['g4_status_coverage']['n_boxes']} boxes "
         f"carry a depth status (coverage {rep['g4_status_coverage']['coverage']:.0%}; asserted, the run fails "
         "otherwise). All output boxes at conf_operational: valid "
         f"{rep['output_boxes_status_fractions']['valid']:.2f}, low_coverage "
         f"{rep['output_boxes_status_fractions']['low_coverage']:.2f}, no_depth "
         f"{rep['output_boxes_status_fractions']['no_depth']:.2f}.", "",
         "## Headline (G3)", "",
         "| mode | valid fraction | median | 95% CI | p90 | 95% CI | low/valid median ratio | 95% CI |",
         "|---|---|---|---|---|---|---|---|"]
    for mode in ("oracle", "detected"):
        r = rep[mode]
        bm, bp, br = r["bootstrap_valid"]["median"], r["bootstrap_valid"]["p90"], r["bootstrap_ratio_low_over_valid"]
        L.append(f"| {mode} | {r['status_fractions']['valid']:.2f} | **{cm(bm['point'])}** | "
                 f"{cm(bm['lo'])}–{cm(bm['hi'])} | **{cm(bp['point'])}** | {cm(bp['lo'])}–{cm(bp['hi'])} | "
                 f"{r['ratio_low_over_valid_median']} | {br['lo']}–{br['hi']} |")
    L += ["", f"95% CIs: percentile bootstrap, {N_BOOT} resamples of images (objects in one image share a depth "
          f"map), seed {SEED}.", "",
          f"![error vs distance](../{figs['distance']})", "", f"![error CDF](../{figs['cdf']})", ""]
    hdr = ["| subset | n | median | mean | p90 | Z MAE | Z RMSE | signed Z |", "|---|---|---|---|---|---|---|---|"]
    for mode in ("oracle", "detected"):
        r = rep[mode]
        sf = r["status_fractions"]
        L += [f"## {mode.capitalize()} boxes", "",
              f"Status: valid {sf['valid']:.2f}, low_coverage {sf['low_coverage']:.2f}, no_depth "
              f"{sf['no_depth']:.2f} (n={sf['n']}).", ""] + hdr
        v = r["valid"]
        L.append("| valid, all " + fmt_stats(v["all"]))
        for b, _, _ in BUCKETS:
            L.append(f"| valid, Z {b} m " + fmt_stats(v["by_distance"][b]))
        for s_, st in v["by_sensor"].items():
            L.append(f"| valid, {s_} " + fmt_stats(st))
        L.append("| low_coverage, all " + fmt_stats(r["low_coverage"]))
        L += ["", "Status fractions by distance and sensor:", "", "| subset | n | valid | low_coverage | no_depth |",
              "|---|---|---|---|---|"]
        for name, d in list(r["status_fractions_by_distance"].items()) + list(r["status_fractions_by_sensor"].items()):
            lab = f"Z {name} m" if name in {b for b, _, _ in BUCKETS} else name
            L.append(f"| {lab} | {d['n']} | {d['valid']:.2f} | {d['low_coverage']:.2f} | {d['no_depth']:.2f} |")
        sc = r["analysis_only_surface_corrected_valid"]["all"]
        sens = r["sensitivity_link_iou_ge_0.3_valid"]
        L += ["", f"*Analysis only, not a headline:* against the GT centre moved half the box depth toward the camera "
              f"(visible-surface proxy), valid median {cm(sc.get('centre_median_m'))} cm, signed Z "
              f"{cm(sc.get('z_signed_mean_m'), True)} cm (volume centre: {cm(v['all'].get('z_signed_mean_m'), True)}"
              f" cm). Links with projection IoU ≥ 0.3 only: median {cm(sens.get('centre_median_m'))} cm "
              f"(n={sens.get('n', 0)}); {r['link_iou_lt_0.3_fraction']:.2f} of links are below 0.3.", ""]
    vt = rep.get("val_vs_test")
    if vt:
        L += ["## Val vs test (R5)", "",
              "Val numbers are the 03.3 tuning results (`benchmarks/geometry/tune_val.json`, same code and "
              "settings). A gap is reported as a finding, not fixed.", "",
              "| mode | split | n | valid | low | no_depth | median | p90 | signed Z | low/valid | 0-2 m | 2-4 m | >4 m |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for mode in ("oracle", "detected"):
            for sp in ("val", "test"):
                x = vt[mode][sp]
                d = x["median_m_by_distance"]
                L.append(f"| {mode} | {sp} | {x['n_objects']} | {x['valid_fraction']:.2f} | "
                         f"{x['low_coverage_fraction']:.2f} | {x['no_depth_fraction']:.2f} | {cm(x['median_m'])} | "
                         f"{cm(x['p90_m'])} | {cm(x['z_signed_mean_m'], True)} | {x['ratio_low_over_valid']} | "
                         f"{cm(d.get('0-2'))} | {cm(d.get('2-4'))} | {cm(d.get('>4'))} |")
        L += ["", "Share of valid objects by sensor:", "", "| mode | split | " + " | ".join(
            sorted(vt["oracle"]["test"]["valid_share_by_sensor"])) + " |",
            "|---|---|" + "---|" * len(vt["oracle"]["test"]["valid_share_by_sensor"])]
        for mode in ("oracle", "detected"):
            for sp in ("val", "test"):
                sh = vt[mode][sp]["valid_share_by_sensor"]
                L.append(f"| {mode} | {sp} | " + " | ".join(f"{sh.get(s, 0):.2f}" for s in
                                                             sorted(vt["oracle"]["test"]["valid_share_by_sensor"]))
                         + " |")
        L.append("")
    L += ["## Limitations", "",
          "- The estimate is a visible-surface point; GT is the 3D box volume centre. A negative signed Z is "
          "expected and is part of the error (see the surface-corrected line, analysis only).",
          "- GT 2D↔3D linkage is by projection + Hungarian IoU (protocol D2). Weak links may pair the wrong 3D box; "
          "the IoU ≥ 0.3 sensitivity line bounds this.",
          "- Detected mode scores only detections matched to a linked GT object; misses and false positives are "
          "counted above, not in the error. Failure cases: spec 05.4.",
          "- SUN RGB-D raw depth (not hole-filled); classes limited to the approved COCO mapping.", ""]
    DOC.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--limit", type=int, default=None, help="first N images; writes *_smoke files, no guard")
    ap.add_argument("--rerun-reason", default=None)
    ap.add_argument("--doc-only", action="store_true", help="regenerate figures and doc from the stored JSON")
    a = ap.parse_args(argv)

    stem, run, headline = SPLITS[a.split]
    smoke = a.limit is not None
    stem += "_smoke" if smoke else ""
    out_json, out_npz = OUT / f"{stem}.json", OUT / f"{stem}_objects.npz"

    if a.doc_only:
        rep = json.loads(out_json.read_text(encoding="utf-8"))
        d = np.load(out_npz, allow_pickle=False)
        objs = {k[2:]: d[k] for k in d.files if k.startswith("o_")}
        if headline and not smoke:
            write_doc(rep, figures(objs, rep, stem), rel(out_json))
        print(f"regenerated figures/doc from {rel(out_json)}")
        return 0

    # Freeze checks (05.1): frozen inputs and the protocol text.
    check_frozen(REPO, FROZEN + [f"configs/splits/{a.split}.txt"])
    check_protocol(REPO, json.loads(INDEX.read_text(encoding="utf-8")))
    rerun = None
    if headline and not smoke:
        rerun = one_shot_guard(out_json, a.rerun_reason)  # raises on a second run without a reason

    objs, boxes, meta = compute(a.split, a.limit, run)
    rep = build_report(objs, boxes, meta, a.split, headline, smoke)
    if rerun:
        rep["reruns"] = rerun.pop("previous_reruns") + [rerun]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **{f"o_{k}": v for k, v in objs.items()}, **{f"b_{k}": v for k, v in boxes.items()})
    out_json.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    if rerun:
        log_rerun(REPO / "PROGRESS.md", rel(out_json), rerun)
    if headline and not smoke:
        write_doc(rep, figures(objs, rep, stem), rel(out_json))
    print(json.dumps({"out": rel(out_json), "seconds": meta["seconds"], "counts": meta["counts"],
                      "g4": rep["g4_status_coverage"],
                      **{mode: {"status": rep[mode]["status_fractions"], "valid_all": rep[mode]["valid"]["all"],
                                "ratio": rep[mode]["ratio_low_over_valid_median"]} for mode in ("oracle", "detected")},
                      "reproduces_03_3": {k: v["equal"] for k, v in (rep.get("reproduces_03_3") or {}).items()}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
