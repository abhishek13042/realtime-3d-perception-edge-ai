"""Label-free tracking evaluation on TUM (spec 04.2 R2-R5, R7).

    python scripts/eval_tracking.py --stage tune                     # fr1_desk only: grid, choose, freeze per_stride
    python scripts/eval_tracking.py --stage report --sequence fr3    # once, frozen config, reads run_tracking output
    python scripts/eval_tracking.py --stage report --sequence fr1    # tuning-set row (labelled as such)

Tuning (R3) runs the tracker in-process over the grid and writes `per_stride` into configs/tracker.yaml.
Reporting never tunes: it reads benchmarks/tracking/tum_<seq>_s<stride>.jsonl from scripts/run_tracking.py and refuses
if those files were not produced with the current (frozen) tracker and depth configs. The fr3 report is one-shot
(R7): a second run needs --rerun-reason, which is stored in the output and must also be logged in PROGRESS.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from perception.config import CONFIG_DIR, load_config  # noqa: E402
from perception.data.tum import TUMSequence  # noqa: E402
from perception.detection.classes import COCO_ID, COCO_NAMES  # noqa: E402
from perception.detection.runio import read_run  # noqa: E402
from perception.eval.tracking import (D_JUMP, camera_speeds, evaluate, largest_jumps, motion_trigger,  # noqa: E402
                                      split_guard)
from perception.tracking.bytetrack import TrackerParams  # noqa: E402
from perception.tracking.runner import Centres3D, align_records, track_sequence  # noqa: E402
from perception.tracking.track_history import TrackHistory, read_tracks, run_header, sha256_file  # noqa: E402

SEQUENCES = {
    "fr1": ("rgbd_dataset_freiburg1_desk", REPO / "benchmarks" / "detection" / "tum_fr1_desk_full.jsonl"),
    "fr3": ("rgbd_dataset_freiburg3_long_office_household",
            REPO / "benchmarks" / "detection" / "tum_fr3_long_office_full.jsonl"),
}
STRIDES = (1, 6)
GRID = {"track_buffer": (15, 30, 60), "match_thresh": (0.7, 0.8, 0.9), "class_aware": (True, False)}
TIE = 0.05
OUT = REPO / "benchmarks" / "tracking"
INVENTORY = REPO / "docs" / "tum_object_inventory.yaml"
TRACKER_YAML = CONFIG_DIR / "tracker.yaml"


def inventory_for(seq_name: str) -> dict:
    inv = yaml.safe_load(INVENTORY.read_text(encoding="utf-8"))
    return inv[seq_name]["counts"]


def excluded_ids(tcfg: dict) -> set[int]:
    return {COCO_ID[c] for c in tcfg["excluded_static_classes"]}


def metrics_for(lines: list[dict], params: TrackerParams, tcfg: dict, seq_name: str) -> dict:
    hist = TrackHistory.from_frames(lines)
    return evaluate(hist.tracks, len(lines), sum(1 for ln in lines if not ln["tracks"]),
                    min_len=int(tcfg["min_track_len"]), buffer=params.track_buffer, excluded_ids=excluded_ids(tcfg),
                    inventory=inventory_for(seq_name), names=list(COCO_NAMES))


# ---- tuning -------------------------------------------------------------------------------------------------------

def choose(rows: list[dict], ref: dict) -> tuple[dict, str]:
    """R3: minimise fragmentation subject to jump rate <= the reference's; ties within TIE go to the reference."""
    jr = lambda r: r["metrics"]["id_switch_proxy"][f"{D_JUMP:.2f}"]["per_100"]  # noqa: E731
    fr = lambda r: r["metrics"]["fragmentation"]["overall"]  # noqa: E731
    ok = [r for r in rows if jr(r) <= jr(ref) + 1e-12]
    best = min(ok, key=lambda r: (fr(r), jr(r), r["params"]["track_buffer"] != 30, r["params"]["match_thresh"] != 0.8,
                                  r["params"]["class_aware"] is not True))
    if fr(ref) - fr(best) < TIE:
        return ref, (f"reference kept: best eligible fragmentation {fr(best):.3f} is within {TIE} of the reference "
                     f"{fr(ref):.3f}")
    return best, (f"fragmentation {fr(ref):.3f} -> {fr(best):.3f} (>= {TIE} better) with jump rate "
                  f"{jr(best):.3f} <= reference {jr(ref):.3f} per 100 valid track-frames")


def write_per_stride(chosen: dict[int, dict]) -> None:
    txt = TRACKER_YAML.read_text(encoding="utf-8")
    body = "per_stride:\n" + "".join(
        f"  {s}: {{track_buffer: {p['track_buffer']}, match_thresh: {p['match_thresh']}, "
        f"class_aware: {str(p['class_aware']).lower()}}}\n" for s, p in sorted(chosen.items()))
    new, n = re.subn(r"^per_stride:.*?(?=\n\S|\n\n|\Z)\n?", body, txt, count=1, flags=re.S | re.M)
    if n != 1:
        raise RuntimeError("per_stride block not found in tracker.yaml")
    TRACKER_YAML.write_text(new, encoding="utf-8")
    back = yaml.safe_load(new)["per_stride"]
    assert {int(k): v for k, v in back.items()} == {s: {k: p[k] for k in GRID} for s, p in chosen.items()}


def tune(a) -> int:
    seq_name, det_path = SEQUENCES[a.sequence]
    split_guard("tune", seq_name)
    dcfg, gcfg, tcfg = load_config("dataset"), load_config("depth"), load_config("tracker")
    if tcfg.get("per_stride") and not a.force:
        raise SystemExit("per_stride is already frozen in configs/tracker.yaml; use --force to re-tune (log it)")
    seq = TUMSequence.from_config(seq_name, dcfg)
    records = align_records(seq, read_run(det_path)[1])
    centres = Centres3D(seq, gcfg)
    base = TrackerParams.from_config(tcfg)  # reference = 04.1 frozen settings (buffer 30, match 0.8, class-aware)
    out = {"stage": "tune", "sequence": seq_name, "grid": GRID, "objective": choose.__doc__, "tie": TIE,
           "d_jump": D_JUMP, "reference_params": base.__dict__, "strides": {}}
    chosen = {}
    for stride in STRIDES:
        rows = []
        for tb, mt, ca in itertools.product(*GRID.values()):
            p = base.with_(track_buffer=tb, match_thresh=mt, class_aware=ca)
            lines, _ = track_sequence(records, p, stride, centres)
            m = metrics_for(lines, p, tcfg, seq_name)
            m.pop("stability_values_m")
            m.pop("loss_seq_indices")
            rows.append({"params": {"track_buffer": tb, "match_thresh": mt, "class_aware": ca}, "metrics": m})
            print(f"s{stride} buf={tb:2d} match={mt} ca={ca!s:5}  frag={m['fragmentation']['overall']:.3f}  "
                  f"jumps/100={m['id_switch_proxy'][f'{D_JUMP:.2f}']['per_100']:.3f}  "
                  f"stab_med={m['stability_m']['median']}  rec={m['recovery']['rate']}", flush=True)
        ref = next(r for r in rows if r["params"] == {"track_buffer": base.track_buffer,
                                                       "match_thresh": base.match_thresh,
                                                       "class_aware": base.class_aware})
        pick, why = choose(rows, ref)
        chosen[stride] = pick["params"]
        out["strides"][str(stride)] = {"rows": rows, "chosen": pick["params"], "reason": why}
        print(f"stride {stride}: chosen {pick['params']} — {why}")
    write_per_stride(chosen)
    out["header"] = run_header(REPO, {"tracker_after_freeze": TRACKER_YAML, "depth": CONFIG_DIR / "depth.yaml",
                                      "dataset": CONFIG_DIR / "dataset.yaml"},
                               {"detections": det_path, "inventory": INVENTORY})
    (OUT / "eval_fr1_tune.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", OUT / "eval_fr1_tune.json", "and per_stride in", TRACKER_YAML)
    return 0


# ---- reporting ----------------------------------------------------------------------------------------------------

def sequence_speeds(seq: TUMSequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.array([it[0] for it in seq.items])
    T = np.stack([seq.poses(x) for x in t])
    lin, ang = camera_speeds(T, t)
    return t, lin, ang


def report(a) -> int:
    seq_name, det_path = SEQUENCES[a.sequence]
    split_guard("report", seq_name)
    dcfg, tcfg = load_config("dataset"), load_config("tracker")
    if not tcfg.get("per_stride") or any(s not in {int(k) for k in tcfg["per_stride"]} for s in STRIDES):
        raise SystemExit("tracker.yaml per_stride is not frozen for strides 1 and 6; run --stage tune first")
    dst = OUT / f"eval_{a.sequence}_report.json"
    if a.sequence == "fr3" and dst.exists() and not a.rerun_reason:
        raise SystemExit(f"{dst.name} exists. The fr3 report is one-shot (R7); pass --rerun-reason and log it in "
                         "PROGRESS.md")
    sha = {"tracker": sha256_file(TRACKER_YAML), "depth": sha256_file(CONFIG_DIR / "depth.yaml")}
    seq = TUMSequence.from_config(seq_name, dcfg)
    t_all, lin, ang = sequence_speeds(seq)
    out = {"stage": "report", "sequence": seq_name, "split": "test" if a.sequence == "fr3" else "tuning (fr1)",
           "rerun_reason": a.rerun_reason, "d_jump": D_JUMP, "strides": {}}
    inputs = {"inventory": INVENTORY}
    for stride in STRIDES:
        params = TrackerParams.from_config(tcfg, stride=stride)
        path = OUT / f"tum_{a.sequence}_s{stride}.jsonl"
        header, lines = read_tracks(path)
        stale = [k for k in sha if header["config_sha256"].get(k) != sha[k]]
        if stale or header["params"] != params.__dict__:
            raise SystemExit(f"{path.name} was not produced with the frozen {stale or 'params'}; re-run "
                             f"scripts/run_tracking.py --sequence {a.sequence} --stride {stride}")
        inputs[path.name] = path
        m = metrics_for(lines, params, tcfg, seq_name)
        hist = TrackHistory.from_frames(lines)
        proc = np.array([ln["seq_index"] for ln in lines])
        loss = np.array(m["loss_seq_indices"], int)
        motion = {"linear_m_s": motion_trigger(lin[loss], lin[proc]),
                  "angular_deg_s": motion_trigger(ang[loss], ang[proc])}
        motion["fires"] = motion["linear_m_s"]["fires"] or motion["angular_deg_s"]["fires"]
        big = largest_jumps(hist.tracks, int(tcfg["min_track_len"]), excluded_ids(tcfg), 0.0, 8)
        summ = json.loads((OUT / f"tum_{a.sequence}_s{stride}_summary.json").read_text(encoding="utf-8"))
        out["strides"][str(stride)] = {
            "params": params.__dict__, "tracks_file": path.name, "tracks_created_utc": header["created_utc"],
            "metrics": m, "camera_motion": motion,
            "largest_jumps": [{**j.__dict__, "cls_name": COCO_NAMES[j.cls], "distance_m": round(j.distance_m, 4),
                               "above_d_jump": j.distance_m > D_JUMP} for j in big],
            "tracker_update_ms": summ["tracker_update_ms"],
            "speed_processed": {"seq_index": proc.tolist(), "linear_m_s": np.round(lin[proc], 4).tolist(),
                                "angular_deg_s": np.round(ang[proc], 3).tolist()},
        }
        mm = m
        print(f"s{stride}: static={mm['n_static_confirmed']} stab_med={mm['stability_m']['median']} "
              f"p90={mm['stability_m']['p90']} frag={mm['fragmentation']['overall']} "
              f"jumps={ {k: v['per_100'] for k, v in mm['id_switch_proxy'].items()} } rec={mm['recovery']} "
              f"valid={mm['coverage']['valid_share']} motion_fires={motion['fires']}")
    out["header"] = run_header(REPO, {"tracker": TRACKER_YAML, "depth": CONFIG_DIR / "depth.yaml",
                                      "dataset": CONFIG_DIR / "dataset.yaml"}, inputs)
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", dst)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("tune", "report"), required=True)
    ap.add_argument("--sequence", choices=sorted(SEQUENCES), default="fr1")
    ap.add_argument("--force", action="store_true", help="tune: overwrite an existing per_stride")
    ap.add_argument("--rerun-reason", default=None, help="report: required to repeat the one-shot fr3 report")
    a = ap.parse_args()
    try:
        return tune(a) if a.stage == "tune" else report(a)
    except PermissionError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
