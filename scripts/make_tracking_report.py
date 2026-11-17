"""Spec 04.2 R6: generate docs/tracking_results.md from benchmarks/tracking/*.json. Do not edit the output by hand.

    python scripts/make_tracking_report.py --gallery   # also render the jump galleries (needs the TUM data)
    python scripts/make_tracking_report.py             # markdown + plots only

Inputs: eval_fr1_tune.json, eval_fr3_report.json (required); eval_fr1_report.json, tum_*_summary.json (optional);
docs/tracking_jump_review.yaml (the by-eye categories of the largest jumps, AC3; missing entries print as "not
reviewed").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

B = REPO / "benchmarks" / "tracking"
DOC = REPO / "docs" / "tracking_results.md"
REVIEW = REPO / "docs" / "tracking_jump_review.yaml"
# Validated default palette (dataviz skill references/palette.md), light mode: slots 1 and 2 in fixed order.
C1, C2, INK2, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
STRIDE_COL = {"1": C1, "6": C2}


def load(name):
    p = B / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def jump_key(stride, j):
    return f"s{stride}:{j['track_id']}:{j['frame_a']}"


# ---- figures ------------------------------------------------------------------------------------------------------

def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURF)


def stability_hist(rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor=SURF)
    vals = {s: np.array(d["metrics"]["stability_values_m"]) * 100 for s, d in rep["strides"].items()}
    hi = max(float(np.percentile(np.concatenate(list(vals.values())), 98)), 1.0)
    bins = np.linspace(0, hi, 31)
    for s, v in vals.items():
        ax.hist(np.clip(v, 0, hi), bins=bins, histtype="step", lw=2, color=STRIDE_COL[s],
                label=f"stride {s} (n = {len(v)} tracks, median {np.median(v):.1f} cm)")
    _style(ax)
    ax.set_xlabel("per-track 3D RMS of world centres about the median (cm; last bin includes larger)", color=INK2,
                  fontsize=9)
    ax.set_ylabel("tracks", color=INK2, fontsize=9)
    ax.set_title("fr3: label-free stability of static tracks", loc="left", fontsize=11, color="#0b0b0b")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURF)
    plt.close(fig)


def speed_plot(rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d1 = rep["strides"]["1"]["speed_processed"]
    t = np.array(d1["seq_index"])
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, facecolor=SURF)
    for ax, key, lab in zip(axes, ("linear_m_s", "angular_deg_s"), ("linear speed (m/s)", "angular speed (deg/s)")):
        ax.plot(t, d1[key], color=MUTED, lw=1.0, label="camera speed (GT poses)")
        for s, d in rep["strides"].items():
            loss = np.array(d["metrics"]["loss_seq_indices"], int)
            if len(loss):
                lookup = dict(zip(d1["seq_index"], d1[key]))
                y = [lookup.get(int(i), np.nan) for i in loss]
                ax.plot(loss, y, ls="none", marker="o" if s == "1" else "D", ms=5, mfc=STRIDE_COL[s],
                        mec=SURF, mew=1.2, label=f"track loss, stride {s} (n = {len(loss)})")
        _style(ax)
        ax.set_ylabel(lab, color=INK2, fontsize=9)
    axes[-1].set_xlabel("sequence frame index", color=INK2, fontsize=9)
    axes[0].set_title("fr3: camera speed with track-loss events", loc="left", fontsize=11, color="#0b0b0b")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURF)
    plt.close(fig)


def gallery(rep, seq_alias, path_for):
    """For each stride: the 8 largest consecutive-centre displacements, frame before | frame after, box drawn."""
    import cv2

    from perception.config import load_config
    from perception.data.tum import TUMSequence
    from perception.tracking.track_history import read_tracks
    seq = TUMSequence.from_config(rep["sequence"], load_config("dataset"))
    for s, d in rep["strides"].items():
        _, lines = read_tracks(B / d["tracks_file"])
        by_fi = {ln["frame_index"]: ln for ln in lines}
        rows = []
        for k, j in enumerate(d["largest_jumps"]):
            tiles = []
            for fi in (j["frame_a"], j["frame_b"]):
                ln = by_fi[fi]
                o = next(t for t in ln["tracks"] if t["track_id"] == j["track_id"])
                img = cv2.cvtColor(seq[ln["seq_index"]].rgb, cv2.COLOR_RGB2BGR)
                x1, y1, x2, y2 = map(int, o["box"])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 255), 3)
                for other in ln["tracks"]:
                    if other["track_id"] != j["track_id"]:
                        a = list(map(int, other["box"]))
                        cv2.rectangle(img, (a[0], a[1]), (a[2], a[3]), (150, 150, 150), 1)
                cw = o["centre_world"]
                txt = (f"#{ln['seq_index']} id {j['track_id']} {j['cls_name']} {o['status']} "
                       f"w=({cw[0]:.2f},{cw[1]:.2f},{cw[2]:.2f})")
                cv2.putText(img, txt, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
                cv2.putText(img, txt, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                tiles.append(cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA))
            bar = np.full((360, 120, 3), 30, np.uint8)
            cv2.putText(bar, f"{k + 1}", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(bar, f"{j['distance_m']:.2f} m", (8, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            rows.append(np.hstack([bar, tiles[0], tiles[1]]))
        if rows:
            out = path_for(s)
            cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
            print("wrote", out)


# ---- markdown -----------------------------------------------------------------------------------------------------

def metrics_table(rep, title):
    L = [f"| {title} | " + " | ".join(f"stride {s}" for s in rep["strides"]) + " |",
         "|---|" + "---|" * len(rep["strides"])]
    ms = [d["metrics"] for d in rep["strides"].values()]
    ps = [d["params"] for d in rep["strides"].values()]

    def row(name, fn):
        L.append(f"| {name} | " + " | ".join(fn(m, p) for m, p in zip(ms, ps)) + " |")

    row("tracker params (buffer / match / class-aware)",
        lambda m, p: f"{p['track_buffer']} / {p['match_thresh']} / {'yes' if p['class_aware'] else 'no'}")
    row("processed frames", lambda m, p: str(m["coverage"]["n_processed"]))
    row("tracks / confirmed / static confirmed",
        lambda m, p: f"{m['n_tracks']} / {m['n_confirmed']} / {m['n_static_confirmed']}")
    row("mean static track length (processed frames)", lambda m, p: f(m["mean_track_len"], 1))
    row("**stability**, median (cm)", lambda m, p: f(None if m["stability_m"]["median"] is None
                                                    else 100 * m["stability_m"]["median"], 1))
    row("stability, p90 (cm)", lambda m, p: f(None if m["stability_m"]["p90"] is None
                                              else 100 * m["stability_m"]["p90"], 1))
    row("tracks with >= 5 valid centres", lambda m, p: str(m["stability_m"]["n_tracks"]))
    row("**fragmentation proxy** (tracks / object) [range from inventory]",
        lambda m, p: f"{f(m['fragmentation']['overall'], 2)} [{f(m['fragmentation']['overall_range'][0], 2)}, "
                     f"{f(m['fragmentation']['overall_range'][1], 2)}]")
    for k in ("0.50", "0.75", "1.00"):
        name = f"**ID-switch proxy**, jumps > {k} m per 100 valid pairs" if k == "0.75" else \
            f"ID-switch proxy, d_jump = {k} m (sensitivity)"
        row(name, lambda m, p, k=k: f"{f(m['id_switch_proxy'][k]['per_100'], 2)} "
                                    f"({m['id_switch_proxy'][k]['n_jumps']} / {m['id_switch_proxy'][k]['valid_pairs']})")
    row("**recovery** (resumed gaps / (resumed + losses))",
        lambda m, p: f"{f(m['recovery']['rate'], 2)} ({m['recovery']['recovered_gaps']} / "
                     f"{m['recovery']['recovered_gaps'] + m['recovery']['losses']})")
    row("coverage: valid 3D share of tracked boxes", lambda m, p: f(m["coverage"]["valid_share"], 3))
    row("frames with zero tracks", lambda m, p: str(m["coverage"]["frames_without_tracks"]))
    return L


def frag_table(rep):
    strides = list(rep["strides"])
    classes = list(rep["strides"][strides[0]]["metrics"]["fragmentation"]["per_class"])
    L = ["| class | objects (manual) | " + " | ".join(f"tracks, stride {s}" for s in strides) + " |",
         "|---|---|" + "---|" * len(strides)]
    for c in classes:
        pc = [rep["strides"][s]["metrics"]["fragmentation"]["per_class"][c] for s in strides]
        L.append(f"| {c} | {pc[0]['objects']} | " + " | ".join(
            f"{r['tracks']} ({r['tracks_per_object']:.2f}/obj)" for r in pc) + " |")
    return L


def tune_section(tune):
    L = ["## Tuning on fr1_desk (R3)", "",
         f"Grid: track_buffer {{15, 30, 60}} (processed frames) x match_thresh {{0.7, 0.8, 0.9}} x class-aware / "
         f"class-agnostic, at strides 1 and 6. Confidence thresholds fixed at conf_operational. Reference = the 04.1 "
         f"settings (buffer {tune['reference_params']['track_buffer']}, match {tune['reference_params']['match_thresh']}, "
         f"class-aware). Objective: lowest fragmentation proxy with an ID-switch proxy (d_jump = {tune['d_jump']} m) "
         f"no higher than the reference's; a gain smaller than {tune['tie']} tracks per object keeps the reference. "
         "(The stock Ultralytics tracker is class-agnostic; that configuration is the `ca = no, 30, 0.8` row.)", ""]
    for s, d in tune["strides"].items():
        ref = tune["reference_params"]
        L += [f"**Stride {s}.** Chosen: buffer {d['chosen']['track_buffer']}, match {d['chosen']['match_thresh']}, "
              f"class-aware {'yes' if d['chosen']['class_aware'] else 'no'}. Reason: {d['reason']}.", "",
              "| ca | buffer | match | frag. (tracks/obj) | jumps / 100 | stability med (cm) | recovery | static tracks |",
              "|---|---|---|---|---|---|---|---|"]
        for r in sorted(d["rows"], key=lambda r: (not r["params"]["class_aware"], r["params"]["track_buffer"],
                                                  r["params"]["match_thresh"])):
            p, m = r["params"], r["metrics"]
            tag = []
            if p == d["chosen"]:
                tag.append("**chosen**")
            if (p["track_buffer"], p["match_thresh"], p["class_aware"]) == (ref["track_buffer"], ref["match_thresh"],
                                                                            ref["class_aware"]):
                tag.append("ref")
            sm = m["stability_m"]["median"]
            L.append(f"| {'yes' if p['class_aware'] else 'no'} | {p['track_buffer']} | {p['match_thresh']} | "
                     f"{f(m['fragmentation']['overall'], 2)} | "
                     f"{f(m['id_switch_proxy'][f'{tune['d_jump']:.2f}']['per_100'], 2)} | "
                     f"{f(None if sm is None else 100 * sm, 1)} | {f(m['recovery']['rate'], 2)} | "
                     f"{m['n_static_confirmed']} {' '.join(tag)} |")
        L.append("")
    return L


def motion_section(rep):
    L = ["## Camera-motion check (R5, architecture switching trigger)", "",
         "Camera linear and angular speed per frame from the GT poses (central differences). Speed at track-loss "
         "events (the last observation of a confirmed static track that never resumes, with at least buffer + 2 "
         "processed frames left) is compared with the speed over all processed frames. Rule: fires if the median at "
         "losses is > 1.5x the overall median, with >= 20 loss events and a one-sided Mann-Whitney p < 0.01, for "
         "either speed.", "",
         "| stride | speed | losses | median at losses | median overall | ratio | p | fires |",
         "|---|---|---|---|---|---|---|---|"]
    for s, d in rep["strides"].items():
        for k, lab in (("linear_m_s", "linear (m/s)"), ("angular_deg_s", "angular (deg/s)")):
            m = d["camera_motion"][k]
            p = m["mannwhitney_p"]
            L.append(f"| {s} | {lab} | {m['n_losses']} | {f(m['median_at_losses'])} | {f(m['median_all'])} | "
                     f"{f(m['ratio'], 2)} | {'n/a' if p != p else f'{p:.2g}'} | {'**yes**' if m['fires'] else 'no'} |")
    fires = any(d["camera_motion"]["fires"] for d in rep["strides"].values())
    L += ["", "**Outcome:** " + (
        "the trigger **fires**. Track losses concentrate at fast camera motion, which is the case the architecture "
        "names for switching to 3D association (a constant-velocity Kalman filter on world-frame centres, gating by "
        "3D distance). It is **not built** in this phase; it is a proposal for approval." if fires else
        "the trigger does **not** fire. Track losses are not concentrated at fast camera motion by this rule, so the "
        "2D ByteTrack association stays; 3D association (a world-frame Kalman filter) is not needed on this "
        "evidence."), "", "![camera speed](../benchmarks/tracking/camera_speed_fr3.png)", ""]
    return L


def jumps_section(rep, review):
    L = ["## Largest jumps, reviewed by eye (AC3)", "",
         "The 8 largest displacements between consecutive valid world centres of one static track, per stride, "
         "whether or not they exceed d_jump. Each was looked at in the gallery (frame before | frame after, the "
         "track's box in yellow) and put in one category: **switch** (the ID moved to another physical object), "
         "**depth outlier** (same object, 3D centre wrong in one frame: background bleed, occluder, missing depth), "
         "or **pose/time** (same object and plausible depth; the displacement comes from timing or GT pose error).", ""]
    counts = {}
    for s, d in rep["strides"].items():
        L += [f"**Stride {s}** — gallery `benchmarks/tracking/jump_gallery_fr3_s{s}.jpg`", "",
              "| # | track | class | seq frames | distance (m) | > d_jump | category | note |",
              "|---|---|---|---|---|---|---|---|"]
        for k, j in enumerate(d["largest_jumps"]):
            r = review.get(jump_key(s, j), {})
            cat = r.get("category", "not reviewed")
            counts[cat] = counts.get(cat, 0) + 1
            L.append(f"| {k + 1} | {j['track_id']} | {j['cls_name']} | {j['seq_a']} -> {j['seq_b']} | "
                     f"{j['distance_m']:.2f} | {'yes' if j['above_d_jump'] else 'no'} | {cat} | {r.get('note', '')} |")
        L.append("")
    L += ["Totals over both strides: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ".", ""]
    return L


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gallery", action="store_true")
    a = ap.parse_args()
    tune, rep = load("eval_fr1_tune.json"), load("eval_fr3_report.json")
    if tune is None or rep is None:
        raise SystemExit("needs benchmarks/tracking/eval_fr1_tune.json and eval_fr3_report.json")
    rep1 = load("eval_fr1_report.json")
    review = yaml.safe_load(REVIEW.read_text(encoding="utf-8")) if REVIEW.exists() else {}
    review = review or {}
    if a.gallery:
        gallery(rep, "fr3", lambda s: B / f"jump_gallery_fr3_s{s}.jpg")
    stability_hist(rep, B / "stability_hist_fr3.png")
    speed_plot(rep, B / "camera_speed_fr3.png")

    h = rep["header"]
    timing = {s: d["tracker_update_ms"] for s, d in rep["strides"].items()}
    L = ["# Tracking results (Phase 04)", "",
         "> Generated by `scripts/make_tracking_report.py` from `benchmarks/tracking/*.json`. Do not edit by hand.", "",
         f"All numbers are **measured** on TUM RGB-D with stored YOLO26-n detections (CPU), ByteTrack (Ultralytics "
         f"8.4.208) and the frozen Phase 03 depth settings. Report run: {h['created_utc']}, git commit "
         f"`{h['git_commit']}` (dirty: {h['git_dirty']}), tracker.yaml sha256 `{h['config_sha256']['tracker'][:12]}`. "
         "There are **no identity labels**: every metric is a label-free proxy (evaluation protocol §4), defined "
         "before any run in spec 04.2. Tuning used fr1_desk only; fr3_long_office_household was evaluated once with "
         "the frozen configuration (R7)."
         + (f" **Re-run reason:** {rep['rerun_reason']}." if rep.get("rerun_reason") else ""), "",
         "## fr3_long_office_household (test split)", ""]
    L += metrics_table(rep, "metric (fr3)") + [""]
    L += ["Definitions: static tracks are confirmed tracks (>= 3 matched frames) whose majority class is not "
          "movable (person, animals, balls are excluded). Stability, jumps and recovery use `valid` 3D centres of "
          "static tracks only. Jumps are counted between consecutive valid centres; recovery counts gaps after which "
          "the same ID resumed against tracks that ended for good (at least buffer + 2 processed frames before the "
          "end of the sequence). Buffer lengths are in processed frames, so at stride 6 a buffer covers 6x the time.",
          "", "![stability histogram](../benchmarks/tracking/stability_hist_fr3.png)", "",
          "### Fragmentation per class (fr3)", "",
          "Objects are a manual count from contact sheets (`docs/tum_object_inventory.yaml`) with their own "
          "uncertainty; the bracket in the table above is the proxy over the low/high inventory counts.", ""]
    L += frag_table(rep) + [""]
    L += ["### Tracker update time (R8: < 2 ms per frame median on fr1, stride 1)", "",
          "| sequence | stride | isolated median (ms) | isolated p95 (ms) | < 2 ms | in-run median (ms) |",
          "|---|---|---|---|---|---|"]
    for alias in ("fr1", "fr3"):
        for s in ("1", "6"):
            iso = load(f"tum_{alias}_s{s}_timing.json")
            summ = load(f"tum_{alias}_s{s}_summary.json")
            if iso is None:
                continue
            t = iso["tracker_update_ms"]
            inrun = summ["tracker_update_ms"]["median"] if summ else None
            tag = " (R8)" if (alias, s) == ("fr1", "1") else ""
            L.append(f"| {alias}{tag} | {s} | {t['median']:.2f} | {t['p95']:.2f} | "
                     f"{'yes' if t['meets_target'] else '**no**'} | {f(inrun, 2)} |")
    L += ["", "Tracker.update only (2D association, frozen per-stride settings), CPU, after a 10-frame warm-up. "
          "*Isolated* = `run_tracking.py --timing-only`: a fresh process, median of 5 tracker-only passes. *In-run* = "
          "the same measurement taken inside the full run, right after loading frames and computing 3D in the same "
          "process; it is consistently higher. Under concurrent load (other runs and an upload) the same code measured "
          "10-12 ms, so the figure is load-sensitive. R8 is met on fr1 in isolation; on fr3 (more classes and tracks, "
          "so more per-class trackers) the tracker takes 2.4-2.6 ms, above the 2 ms budget. The class-agnostic "
          "variant takes about 0.8 ms on fr1, but it had far worse jump and stability numbers in tuning.", ""]
    L += jumps_section(rep, review)
    L += motion_section(rep)
    if rep1:
        L += ["## fr1_desk with the frozen settings (tuning split, for reference only)", "",
              "These numbers are on the sequence the parameters were chosen on, so they are optimistic.", ""]
        L += metrics_table(rep1, "metric (fr1, tuning)") + [""]
    L += tune_section(tune)
    L += ["## Limitations", "",
          "- **No identity labels.** Fragmentation, jumps and recovery are proxies. A track that silently moves to "
          "a neighbouring object of the same class at similar depth is not caught by the jump proxy.",
          "- **Static assumption.** World-frame metrics assume the objects do not move. Movable classes are "
          "excluded, but a cup or chair moved by a person in the sequence would count as instability or a jump.",
          "- **Manual inventory.** Object counts are by eye from every 30th frame, with a stated range; small, "
          "partly visible objects (bottles, books) are the least certain, and the fragmentation proxy inherits it.",
          "- **Jumps mix causes.** A jump can be a real ID switch or a one-frame depth outlier (background bleed, "
          "occluder). The by-eye review above separates them for the largest ones only.",
          "- **Two sequences.** One tuning and one test sequence, both offices with a hand-held Kinect; no claim "
          "is made beyond this setting.",
          "- **Detector errors pass through.** Misclassifications (the duck as `apple`, the cabinet as "
          "`refrigerator`) create real, consistent tracks with wrong labels; class-aware tracking also splits a track "
          "when the detector flips the class of one object.", ""]
    DOC.write_text("\n".join(L), encoding="utf-8")
    print("wrote", DOC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
