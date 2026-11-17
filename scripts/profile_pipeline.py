"""Spec 05.3 R3-R6: per-stage CPU latency of the end-to-end pipeline (protocol §5) and reliability (§6).

Procedure (protocol §5):
- Frames: the fixed lists in configs/benchmark_frames/: fr1_desk 200 and fr3_long_office 200 with tracking, and the
  first 200 of sunrgbd_test_500 in single-image mode (no tracking).
- Each run is a fresh process (this script re-launches itself with --child). Per list: 20 warm-up iterations on the
  list's first frames (discarded; the tracker is reset afterwards), then every frame is timed once.
- Boundaries: `detector` = model forward only; `detect_total` = original image -> boxes in original pixels;
  `pipeline` = Frame in memory -> FrameResult (detect + track + depth + geometry + assembly). `io` (image + depth
  decode into a Frame) is timed separately and never enters the pipeline FPS. FPS = 1000 / mean pipeline ms.
- Machine state: the script refuses to start while another heavy process from this project is running (e.g. the
  Kaggle upload) or the CPU is busy, unless --allow-load is given; the result is then labelled "under load".

Tracking lists are random samples of the sequence (sorted), not consecutive frames: the fr1 list has a median gap of
~2 frames, fr3 ~10 frames. The tracker uses the tuned stride-1 parameters on fr1 and stride-6 on fr3 (the closest
tuned stride). Recovery on these lists is therefore not comparable with the 04.2 numbers; it is a liveness check.

Usage:  python -I scripts/profile_pipeline.py [--runs 3] [--allow-load] [--limit N]
Writes benchmarks/results/pipeline_profile.json (or pipeline_profile_smoke.json with --limit).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perception.config import CONFIG_DIR, data_root, load_config  # noqa: E402
from perception.detection.runio import latency_stats  # noqa: E402
from perception.eval.results_index import git_state, sha256_file  # noqa: E402

BF = REPO / "configs" / "benchmark_frames"
LISTS = {  # name -> (frame list, kind, sequence, tracker stride)
    "tum_fr1_desk_200": (BF / "tum_rgbd_dataset_freiburg1_desk_200.txt", "tum", "rgbd_dataset_freiburg1_desk", 1),
    "tum_fr3_long_office_200": (BF / "tum_rgbd_dataset_freiburg3_long_office_household_200.txt", "tum",
                                "rgbd_dataset_freiburg3_long_office_household", 6),
    "sunrgbd_test_200": (BF / "sunrgbd_test_500.txt", "sun", None, None),
}
N_FRAMES, WARMUP = 200, 20
STAGES = ("io", "preprocess", "detector", "detect", "detect_total", "track", "depth", "geometry", "assemble",
          "pipeline")
SHARE_STAGES = ("preprocess", "detect", "track", "depth", "geometry", "assemble")
TARGET_FPS = 10.0
OUT = REPO / "benchmarks" / "results"


def read_ids(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def power_state() -> dict:
    out: dict = {}
    try:
        b = psutil.sensors_battery()
        out["plugged_in"] = None if b is None else bool(b.power_plugged)
        out["battery_percent"] = None if b is None else b.percent
    except (AttributeError, RuntimeError):
        out["plugged_in"] = None
    try:
        r = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10)
        out["power_scheme"] = r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        out["power_scheme"] = None
    return out


def machine_load() -> dict:
    """Heavy processes from this project (Kaggle upload, other project scripts) and the CPU load over 3 s."""
    me = {os.getpid()} | {a.pid for a in psutil.Process().parents()}   # this script and its launching shells
    heavy = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] in me:
                continue
            cmd = " ".join(p.info["cmdline"] or [])
            name = (p.info["name"] or "").lower()
            if "kaggle" in name or ("python" in name and ("kaggle" in cmd.lower() or "Project 5 CV" in cmd)):
                heavy.append({"pid": p.info["pid"], "name": p.info["name"], "cmd": cmd[:160]})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    cpu = psutil.cpu_percent(interval=3.0)
    return {"heavy_processes": heavy, "cpu_percent_3s": cpu, "idle": not heavy and cpu < 25.0}


# ------------------------------------------------------------------------------------------------------ child
def child(run_id: int, limit: int | None, out_path: Path) -> int:
    import torch
    from threadpoolctl import threadpool_info

    from perception.data.sunrgbd import SunRGBD
    from perception.data.tum import TUMSequence
    from perception.detection.torch_backend import build_detector
    from perception.eval.tracking import interruptions, static_tracks
    from perception.detection.classes import COCO_ID
    from perception.pipeline import Pipeline, validate_result
    from perception.tracking.track_history import TrackHistory

    dcfg, gcfg, tcfg, detcfg = (load_config(n) for n in ("dataset", "depth", "tracker", "detector"))
    det = build_detector(detcfg)
    conf_op = float(detcfg["conf_operational"])
    excluded = {COCO_ID[c] for c in tcfg["excluded_static_classes"]}
    proc = psutil.Process()
    res: dict = {"run_id": run_id, "pid": os.getpid(), "lists": {}, "power_before": power_state(),
                 "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                 "blas": [{k: d.get(k) for k in ("internal_api", "num_threads", "version")} for d in threadpool_info()]}
    sun = None
    for name, (lst, kind, seq_name, stride) in LISTS.items():
        ids = read_ids(lst)[:N_FRAMES][:limit]
        if kind == "tum":
            src = TUMSequence.from_config(seq_name, dcfg)
            pos = {f"{t:.6f}": i for i, (t, _, _) in enumerate(src.items)}
            idx = [pos[i] for i in ids]
            pipe = Pipeline(det, gcfg, tcfg, stride, conf_operational=conf_op, track=True)
        else:
            sun = sun or SunRGBD(data_root(dcfg) / "sunrgbd")
            src, idx = sun, [int(i) - 1 for i in ids]       # official 1-based image ids
            pipe = Pipeline(det, gcfg, tcfg, 1, conf_operational=conf_op, track=False)

        for j in idx[:WARMUP]:                              # warm-up, discarded
            pipe.process(src[j])
        pipe.reset()

        rows, failed, invalid, obs, schema_err = [], [], [], [], 0
        for n, j in enumerate(idx):
            t0 = time.perf_counter()
            try:
                frame = src[j]
            except Exception as e:                         # noqa: BLE001 (counted as a failed frame, §6)
                failed.append({"index": n, "stage": "io", "error": repr(e)[:200]})
                continue
            io_ms = (time.perf_counter() - t0) * 1e3
            try:
                r = pipe.process(frame, seq_index=j)
            except Exception as e:                         # noqa: BLE001
                failed.append({"index": n, "stage": "pipeline", "error": repr(e)[:200]})
                continue
            tm = r.timing_ms
            rows.append({"io": io_ms, "preprocess": tm["preprocess"], "detector": r.detector_ms,
                         "detect": tm["detect"], "detect_total": r.extra["detect_total_ms"], "track": tm["track"],
                         "depth": tm["depth"], "geometry": tm["geometry"], "assemble": r.extra["assemble_ms"],
                         "pipeline": tm["total"], "n_objects": len(r.objects),
                         "n_tracked": sum(o.raw_track_id is not None for o in r.objects)})
            invalid.append(1.0 - float(frame.valid.mean()))         # outside the timed region
            if kind == "tum":
                obs += r.observations()
            if n < 5:                                               # schema spot check, outside the timing
                try:
                    validate_result(json.loads(json.dumps(r.to_dict(), allow_nan=False)))
                except ValueError:
                    schema_err += 1
        rel = {"n_frames": len(idx), "failed_frames": len(failed), "failures": failed[:10],
               "zero_object_frames": sum(x["n_objects"] == 0 for x in rows),
               "mean_invalid_depth_rate": round(float(np.mean(invalid)), 4) if invalid else None,
               "schema_errors_first5": schema_err}
        if kind == "tum":
            hist = TrackHistory()
            hist.add(obs)
            static = static_tracks(hist.tracks, int(tcfg["min_track_len"]), excluded)
            rec = loss = 0
            for v in static.values():
                a, b, _ = interruptions(v, len(rows), pipe.params.track_buffer)
                rec, loss = rec + a, loss + b
            rel["recovery"] = {"recovered_gaps": rec, "losses": loss,
                               "rate": round(rec / (rec + loss), 3) if rec + loss else None,
                               "n_tracks": len(hist.tracks), "n_static_confirmed": len(static),
                               "tracker_stride_params": stride}
        res["lists"][name] = {"rows": rows, "reliability": rel}
        p = [x["pipeline"] for x in rows]
        print(f"[run {run_id}] {name}: pipeline median {np.median(p):.1f} ms, mean {np.mean(p):.1f} ms, "
              f"failed {len(failed)}", flush=True)
    mem = proc.memory_info()
    res["peak_rss_mb"] = round(getattr(mem, "peak_wset", mem.rss) / 2**20, 1)
    res["power_after"] = power_state()
    out_path.write_text(json.dumps(res), encoding="utf-8")
    return 0


# ------------------------------------------------------------------------------------------------------ parent
def summarise(runs: list[dict]) -> dict:
    out = {}
    for name in LISTS:
        per_run = [r["lists"][name] for r in runs]
        stages = {}
        for s in STAGES:
            each = [latency_stats([x[s] for x in pr["rows"]]) for pr in per_run]
            pooled = latency_stats([x[s] for pr in per_run for x in pr["rows"]])
            stages[s] = {"runs": each, "pooled": pooled,
                         "run_spread_median_ms": [min(e["median"] for e in each), max(e["median"] for e in each)]}
        fps = [round(1000 / e["mean"], 2) for e in stages["pipeline"]["runs"]]
        pmed = stages["pipeline"]["pooled"]["median"]
        pmean = stages["pipeline"]["pooled"]["mean"]
        out[name] = {
            "stages_ms": stages,
            "pipeline_fps_per_run": fps, "pipeline_fps_pooled": round(1000 / pmean, 2),
            "pipeline_fps_from_median": round(1000 / pmed, 2),
            "share_of_pipeline_median": {s: round(stages[s]["pooled"]["median"] / pmed, 3) for s in SHARE_STAGES},
            "share_of_pipeline_mean": {s: round(stages[s]["pooled"]["mean"] / pmean, 3) for s in SHARE_STAGES},
            "reliability_per_run": [pr["reliability"] for pr in per_run],
            "mean_objects_per_frame": round(float(np.mean([x["n_objects"] for pr in per_run for x in pr["rows"]])), 2),
        }
    return out


def bottleneck(summary: dict) -> dict:
    rows = {}
    for name, s in summary.items():
        share = s["share_of_pipeline_mean"]
        dom = max(share, key=share.get)
        rows[name] = {"fps": s["pipeline_fps_pooled"], "meets_10fps": s["pipeline_fps_pooled"] >= TARGET_FPS,
                      "dominant_stage": dom, "dominant_share_of_mean": share[dom]}
    worst = min(rows.values(), key=lambda r: r["fps"])
    detect_share = max(s["share_of_pipeline_mean"]["preprocess"] + s["share_of_pipeline_mean"]["detect"]
                       for s in summary.values())
    text = (f"Lowest pipeline FPS {worst['fps']:.2f} (target {TARGET_FPS:.0f}, soft). "
            + ("All lists meet the target. " if all(r["meets_10fps"] for r in rows.values()) else
               "The target is missed. ")
            + f"Detection (preprocess + model + postprocess) is up to {100 * detect_share:.0f}% of the mean "
              "pipeline time; depth, geometry and tracking together are the rest. Phase 06 levers, in the order "
              "the protocol lists them: ONNX Runtime FP32, ONNX Runtime INT8 (static quantisation), the input-size "
              "sweep (imgsz 640 -> 512 / 416, with the mAP change reported), and optionally OpenVINO. "
              "No measurement boundary was moved.")
    return {"per_list": rows, "target_fps": TARGET_FPS, "statement": text}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, help="frames per list (smoke test; writes a _smoke file)")
    ap.add_argument("--allow-load", action="store_true", help="run even if the machine is busy; labelled 'under load'")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--run-id", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--child-out", type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        return child(a.run_id, a.limit, a.child_out)

    load_before = machine_load()
    if not load_before["idle"] and not a.allow_load:
        print("machine is not idle; refusing to profile (use --allow-load to run anyway, labelled 'under load'):\n"
              + json.dumps(load_before, indent=2), file=sys.stderr)
        return 2
    under_load = not load_before["idle"]
    runs = []
    with tempfile.TemporaryDirectory() as td:
        for k in range(a.runs):
            out = Path(td) / f"run{k}.json"
            cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "--child", "--run-id", str(k),
                   "--child-out", str(out)] + (["--limit", str(a.limit)] if a.limit else [])
            t0 = time.perf_counter()
            subprocess.run(cmd, check=True)
            r = json.loads(out.read_text(encoding="utf-8"))
            r["wall_s"] = round(time.perf_counter() - t0, 1)
            runs.append(r)
    load_after = machine_load()
    summary = summarise(runs)
    detcfg = load_config("detector")
    smoke = a.limit is not None
    label = ("Measured: PyTorch CPU pipeline, local laptop" + (", UNDER LOAD" if under_load else "")
             + (", SMOKE (not a result)" if smoke else ""))
    report = {
        "spec": "05.3 R3-R6", "label": label, "under_load": under_load, "backend": "pytorch-cpu",
        "procedure": {"runs": a.runs, "fresh_process_per_run": True, "warmup_iterations": WARMUP,
                      "frames_per_list": a.limit or N_FRAMES, "fps": "1000 / mean pipeline ms",
                      "boundaries": {"detector": "model forward only", "detect_total": "image -> boxes (pre+infer+post)",
                                     "pipeline": "Frame -> FrameResult (detect+track+depth+geometry+assembly)",
                                     "io": "Frame decode; never in pipeline FPS"}},
        "machine": {"cpu": "Intel Core i3-1115G4 (2C/4T)", "platform": platform.platform(),
                    "processor": platform.processor(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
                    "python": platform.python_version()},
        "machine_state": {"before": load_before, "after": load_after},
        "power": [{"run": r["run_id"], "before": r["power_before"], "after": r["power_after"]} for r in runs],
        "threads": {"torch": runs[0]["torch_threads"], "torch_interop": runs[0]["torch_interop_threads"],
                    "blas": runs[0]["blas"]},
        "versions": _versions(),
        "peak_rss_mb_per_run": [r["peak_rss_mb"] for r in runs],
        "wall_s_per_run": [r["wall_s"] for r in runs],
        "lists": summary,
        "bottleneck": bottleneck(summary),
        "header": {**git_state(REPO), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "config_sha256": {k: sha256_file(CONFIG_DIR / f"{k}.yaml") for k in ("depth", "tracker", "detector")},
                   "input_sha256": {Path(v[0]).name: sha256_file(v[0]) for v in LISTS.values()},
                   "detector_weights_sha256": detcfg["weights_sha256"], "imgsz": detcfg["imgsz"],
                   "conf": detcfg["conf"], "conf_operational": detcfg["conf_operational"]},
    }
    path = OUT / ("pipeline_profile_smoke.json" if smoke else "pipeline_profile.json")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, s in summary.items():
        st = s["stages_ms"]
        print(f"{name}: pipeline median {st['pipeline']['pooled']['median']:.1f} ms, p95 "
              f"{st['pipeline']['pooled']['p95']:.1f}, FPS {s['pipeline_fps_pooled']} (runs {s['pipeline_fps_per_run']}); "
              f"detector median {st['detector']['pooled']['median']:.1f} ms")
    print(report["bottleneck"]["statement"])
    print(f"wrote {path.relative_to(REPO)} ({label})")
    return 0


def _versions() -> dict:
    import importlib.metadata as md
    out = {}
    for p in ("torch", "torchvision", "ultralytics", "numpy", "opencv-python", "opencv-python-headless", "lap",
              "psutil"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            continue
    return out


if __name__ == "__main__":
    raise SystemExit(main())
