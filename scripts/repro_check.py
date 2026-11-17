"""Spec 05.4 R3: re-run the headline results from scratch and compare with the reported values.

Tolerances are fixed in perception.eval.repro.TOLERANCES (set before any re-run):
  sun_test_200     05.2 on the first 200 SUN test images: per-object 3D error, abs 1e-9 m, statuses identical
  fr3_s1_metrics   04.2 fr3 stride-1 tracking metrics, recomputed from the stored detections: exact
  detection_map    02.3 SUN test mAP with the detector re-run on all 5,050 test images (fresh run file): +-0.001
  pipeline_median  05.3 pipeline median per list, one fresh profiling process vs the reported pooled median: +-15%

  python -I scripts/repro_check.py                          # all items
  python -I scripts/repro_check.py --items sun_test_200 fr3_s1_metrics

Items are merged into benchmarks/results/repro_check.json, so a slow item can be run on its own later.
Nothing here writes to the reported result files.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from perception.config import CONFIG_DIR, data_root, load_config  # noqa: E402
from perception.eval.repro import TOLERANCES, compare_arrays, compare_exact, compare_scalar  # noqa: E402
from perception.eval.results_index import git_state, sha256_file  # noqa: E402

OUT = REPO / "benchmarks" / "results" / "repro_check.json"
SUN_N = 200
REPRO_DET_RUN = "repro_sun_test_full"


def rel(p: Path) -> str:
    return p.relative_to(REPO).as_posix()


def item_sun_test_200() -> dict:
    loc = importlib.import_module("eval_localization")
    stem, run, _ = loc.SPLITS["sunrgbd_test"]
    ref_npz = REPO / "benchmarks" / "results" / f"{stem}_objects.npz"
    ref = np.load(ref_npz, allow_pickle=False)
    objs, boxes, meta = loc.compute("sunrgbd_test", SUN_N, run)
    ids = set(objs["image_id"].tolist()) | set(boxes["image_id"].tolist())
    ids_split = set(loc.read_split(REPO / "configs" / "splits" / "sunrgbd_test.txt")[:SUN_N])
    m = np.isin(ref["o_image_id"], list(ids_split))
    err_ref = np.linalg.norm(ref["o_centre"][m] - ref["o_gt"][m], axis=1)
    err_new = np.linalg.norm(objs["centre"] - objs["gt"], axis=1)
    res = compare_arrays("sun_test_200", err_ref, err_new)
    same_rows = (m.sum() == len(objs["mode"]) and np.array_equal(ref["o_mode"][m], objs["mode"])
                 and np.array_equal(ref["o_image_id"][m], objs["image_id"]))
    same_status = same_rows and np.array_equal(ref["o_status"][m], objs["status"])
    bm = np.isin(ref["b_image_id"], list(ids_split))
    same_boxes = bool(bm.sum() == len(boxes["status"]) and np.array_equal(ref["b_status"][bm], boxes["status"])
                      and np.array_equal(ref["b_matched_any"][bm], boxes["matched_any"]))
    res.update({"n_images": SUN_N, "images_with_objects": len(ids), "rows_identical": bool(same_rows),
                "statuses_identical": bool(same_status), "output_boxes_identical": same_boxes,
                "seconds": meta["seconds"], "reported_file": rel(ref_npz), "reported_sha256": sha256_file(ref_npz)})
    res["pass"] = bool(res["pass"] and same_rows and same_status and same_boxes)
    return res


def item_fr3_s1_metrics() -> dict:
    et = importlib.import_module("eval_tracking")
    from perception.data.tum import TUMSequence
    from perception.detection.runio import read_run
    from perception.tracking.bytetrack import TrackerParams
    from perception.tracking.runner import Centres3D, align_records, track_sequence

    seq_name, det_path = et.SEQUENCES["fr3"]
    ref_path = REPO / "benchmarks" / "tracking" / "eval_fr3_report.json"
    reported = json.loads(ref_path.read_text(encoding="utf-8"))["strides"]["1"]
    dcfg, gcfg, tcfg = load_config("dataset"), load_config("depth"), load_config("tracker")
    params = TrackerParams.from_config(tcfg, stride=1)
    if params.__dict__ != reported["params"]:
        return {"item": "fr3_s1_metrics", "pass": False, "reason": "tracker params differ from the reported run"}
    seq = TUMSequence.from_config(seq_name, dcfg)
    t0 = time.perf_counter()
    records = align_records(seq, read_run(det_path)[1])
    lines, _ = track_sequence(records, params, 1, Centres3D(seq, gcfg))
    lines = json.loads(json.dumps(lines))                     # same float handling as the stored frame lines
    metrics = et.metrics_for(lines, params, tcfg, seq_name)
    res = compare_exact("fr3_s1_metrics", reported["metrics"], metrics)
    res.update({"n_frames": len(lines), "seconds": round(time.perf_counter() - t0, 1),
                "reported_file": rel(ref_path), "reported_sha256": sha256_file(ref_path),
                "detections": rel(det_path)})
    return res


def item_detection_map() -> dict:
    ed = importlib.import_module("eval_detection")
    from perception.data.sunrgbd_annotations import load_annotations
    from perception.detection.classes import COCO_ID

    ref_path = REPO / "benchmarks" / "detection" / "eval_baseline.json"
    reported = json.loads(ref_path.read_text(encoding="utf-8"))
    run_file = REPO / "benchmarks" / "detection" / f"{REPRO_DET_RUN}.jsonl"
    t0 = time.perf_counter()
    if run_file.exists():
        run_file.unlink()                                     # from scratch: never resume a previous repro run
    subprocess.run([sys.executable, "-I", str(REPO / "scripts" / "run_detector.py"), "--sun", "sunrgbd_test",
                    "--name", REPRO_DET_RUN], check=True)
    det_s = time.perf_counter() - t0
    th, trecs, thw = ed.load_run(REPRO_DET_RUN)
    ed.check_complete(REPRO_DET_RUN, trecs, "sunrgbd_test")
    rh = reported["run_header"]
    same_cfg = {k: th.get(k) == rh.get(k) for k in ("weights_sha256", "head", "imgsz", "conf")}
    sun_to_coco = yaml.safe_load((REPO / "configs" / "class_map.yaml").read_text())["sun_to_coco"]
    targets = {COCO_ID[c] for c in sun_to_coco.values()}
    anns = load_annotations(data_root(load_config("dataset")) / "sunrgbd")
    tgt, _ = ed.build_coco_gt(anns, thw, sun_to_coco)
    summ = ed.coco_evaluate(tgt, ed.build_coco_dets(trecs, targets))["summary"]
    res = compare_scalar("detection_map", reported["test"]["coco_summary"]["mAP"], summ["mAP"])
    res.update({"AP50": {"reported": reported["test"]["coco_summary"]["AP50"], "rerun": summ["AP50"]},
                "run_settings_match": same_cfg, "n_images": len(trecs), "detector_seconds": round(det_s, 1),
                "rerun_file": rel(run_file), "reported_file": rel(ref_path), "reported_sha256": sha256_file(ref_path)})
    res["pass"] = bool(res["pass"] and all(same_cfg.values()))
    return res


def item_pipeline_median() -> dict:
    prof = importlib.import_module("profile_pipeline")
    ref_path = REPO / "benchmarks" / "results" / "pipeline_profile.json"
    if not ref_path.exists():
        return {"item": "pipeline_median", "pass": None, "status": "pending",
                "reason": "benchmarks/results/pipeline_profile.json not produced yet (05.3 timed run)"}
    reported = json.loads(ref_path.read_text(encoding="utf-8"))
    load = prof.machine_load()
    if not load["idle"]:                                  # same gate as profile_pipeline.py: no timing under load
        return {"item": "pipeline_median", "pass": None, "status": "not_idle", "machine_state": load,
                "reason": "machine not idle; timing not run (retry when idle)"}
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "repro.json"
        subprocess.run([sys.executable, "-I", str(REPO / "scripts" / "profile_pipeline.py"), "--child",
                        "--run-id", "99", "--child-out", str(out)], check=True)
        run = json.loads(out.read_text(encoding="utf-8"))
    per_list = {}
    for name, v in run["lists"].items():
        new = float(np.median([x["pipeline"] for x in v["rows"]]))
        ref = reported["lists"][name]["stages_ms"]["pipeline"]["pooled"]["median"]
        per_list[name] = compare_scalar("pipeline_median", ref, new)
    return {"item": "pipeline_median", "what": TOLERANCES["pipeline_median"]["what"],
            "tolerance": {k: TOLERANCES["pipeline_median"][k] for k in ("kind", "tol", "unit")},
            "per_list": per_list, "machine_state": load, "reported_under_load": reported.get("under_load"),
            "reported_file": rel(ref_path), "reported_sha256": sha256_file(ref_path),
            "pass": all(r["pass"] for r in per_list.values())}


ITEMS = {"sun_test_200": item_sun_test_200, "fr3_s1_metrics": item_fr3_s1_metrics,
         "detection_map": item_detection_map, "pipeline_median": item_pipeline_median}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", nargs="+", choices=sorted(ITEMS), default=list(ITEMS))
    a = ap.parse_args(argv)
    doc = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {
        "spec": "05.4 R3", "label": "Measured: from-scratch re-runs vs the reported values",
        "tolerances": TOLERANCES, "items": {}}
    for name in a.items:
        print(f"== {name}", flush=True)
        t0 = time.perf_counter()
        r = ITEMS[name]()
        r["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        r["wall_s"] = round(time.perf_counter() - t0, 1)
        if r.get("status") == "not_idle":                 # a refused timing run is not a result; keep the old one
            print(json.dumps(r), flush=True)
            continue
        if name in doc["items"]:                          # never overwrite silently: earlier attempts stay on record
            doc.setdefault("history", {}).setdefault(name, []).append(doc["items"][name])
        doc["items"][name] = r
        print(json.dumps({k: r.get(k) for k in ("item", "pass", "max_abs_diff", "diff", "differences", "status")}),
              flush=True)
    done = [v["pass"] for v in doc["items"].values()]
    doc["all_pass"] = (all(done) if len(done) == len(ITEMS) and None not in done else None)
    doc["header"] = {**git_state(REPO), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "config_sha256": {k: sha256_file(CONFIG_DIR / f"{k}.yaml")
                                       for k in ("depth", "tracker", "detector", "class_map")}}
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {rel(OUT)}; all_pass = {doc['all_pass']}")
    return 0 if all(v["pass"] is not False for v in doc["items"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
