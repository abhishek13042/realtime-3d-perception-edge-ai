"""Spec 05.4 R4: generate docs/results_summary.md from benchmarks/results/index.json. No hand-typed numbers.

    python -I scripts/results_index.py          # refresh the index first
    python -I scripts/make_results_summary.py

Every value goes through perception.eval.summary.Sources, which refuses files missing from the index or changed
since it was written. Each value is printed next to a link to its source file. Missing optional results (the 05.3
profile, a repro item) are shown as pending, never filled in.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perception.eval.results_index import check_protocol  # noqa: E402
from perception.eval.summary import Sources, fmt  # noqa: E402

INDEX = REPO / "benchmarks" / "results" / "index.json"
DOC = REPO / "docs" / "results_summary.md"
FIG = REPO / "benchmarks" / "results" / "pipeline_stage_profile.png"
DET = "benchmarks/detection/eval_baseline.json"
LOC = "benchmarks/results/localization_sun_test.json"
TRK = "benchmarks/tracking/eval_fr3_report.json"
PROF = "benchmarks/results/pipeline_profile.json"
EQV = "benchmarks/results/pipeline_equivalence_fr1.json"
GAL = "benchmarks/results/failure_gallery.json"
REP = "benchmarks/results/repro_check.json"
STACK = ("preprocess", "detect", "depth", "geometry", "track", "assemble")
# dataviz reference palette, categorical slots 1-6 in fixed order (light surface), one per stage
STAGE_C = {"preprocess": "#2a78d6", "detect": "#eb6834", "depth": "#1baf7a", "geometry": "#eda100",
           "track": "#e87ba4", "assemble": "#008300"}


def link(path: str) -> str:
    return f"[{Path(path).name}](../{path})"


def goals(s: Sources) -> list[str]:
    rows = ["| Goal | Metric | Measured | Source | Status |", "|---|---|---|---|---|"]
    rows.append("| G1 back-projection exact | round trip < 1e-6 m, synthetic plane | verified by unit tests "
                "(no stored result file) | [test_geometry.py](../tests/test_geometry.py) | met (tests) |")
    cs, op = s.get(DET, "test", "coco_summary"), s.get(DET, "test", "at_operational")
    rows.append(f"| G2 detection baseline | SUN test mAP@[.5:.95] / AP50; P / R at conf {fmt(op['threshold'], 4)} | "
                f"{fmt(cs['mAP'])} / {fmt(cs['AP50'])}; {fmt(op['precision'])} / {fmt(op['recall'])} | {link(DET)} | "
                "met (documented) |")
    va, vo = s.get(LOC, "detected", "valid", "all"), s.get(LOC, "oracle", "valid", "all")
    rows.append(f"| G3 3D centre error | median / p90, valid, detected (oracle) | {fmt(va['centre_median_m'])} / "
                f"{fmt(va['centre_p90_m'])} m ({fmt(vo['centre_median_m'])} / {fmt(vo['centre_p90_m'])} m), "
                f"n = {va['n']} | {link(LOC)} | met (measured) |")
    g4 = s.get(LOC, "g4_status_coverage")
    rows.append(f"| G4 invalid depth never silent | boxes with a depth status | {g4['with_status']} / {g4['n_boxes']} "
                f"({fmt(g4['coverage'], pct=True)}) | {link(LOC)} | {'met' if g4['coverage'] == 1.0 else 'not met'} |")
    m1, m6 = s.get(TRK, "strides", "1", "metrics"), s.get(TRK, "strides", "6", "metrics")
    rows.append(f"| G5 tracking stability (fr3) | static-object world std median / p90; fragmentation; jumps "
                f">0.5 m per 100 | stride 1: {fmt(m1['stability_m']['median'])} / {fmt(m1['stability_m']['p90'])} m, "
                f"{fmt(m1['fragmentation']['overall'], 2)}, {fmt(m1['id_switch_proxy']['0.50']['per_100'])}; "
                f"stride 6: {fmt(m6['stability_m']['median'])} / {fmt(m6['stability_m']['p90'])} m, "
                f"{fmt(m6['fragmentation']['overall'], 2)}, {fmt(m6['id_switch_proxy']['0.50']['per_100'])} | "
                f"{link(TRK)} | met (measured) |")
    rows.append("| G6 optimisation measured fairly | PyTorch vs ORT FP32 vs INT8 | not yet run | Phase 06 | pending |")
    rows.append("| G7 demo reproducible | fresh clone + README runs the demo | not yet run | Phase 07 | pending |")
    if s.has(PROF):
        b = s.get(PROF, "bottleneck")
        lo = min(v["fps"] for v in b["per_list"].values())
        under = s.get(PROF, "under_load")
        rows.append(f"| Speed (soft target 10 FPS) | lowest pipeline FPS over the 3 lists | {fmt(lo, 2)}"
                    f"{' (UNDER LOAD)' if under else ''} | {link(PROF)} | "
                    f"{'met' if lo >= b['target_fps'] else 'not met'} |")
    else:
        rows.append("| Speed (soft target 10 FPS) | pipeline FPS | 05.3 timed run not done | - | pending |")
    return rows


def stage_figure(s: Sources) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lists = s.get(PROF, "lists")
    names = list(lists)
    fig, ax = plt.subplots(figsize=(8.5, 0.9 + 0.75 * len(names)))
    for y, n in enumerate(names):
        left = 0.0
        for st in STACK:
            w = lists[n]["stages_ms"][st]["pooled"]["mean"]
            ax.barh(y, w, left=left, color=STAGE_C[st], height=0.55, edgecolor="white", linewidth=2,
                    label=st if y == 0 else None)
            if w >= 8:
                ax.text(left + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=8, color="white")
            left += w
        tot = lists[n]["stages_ms"]["pipeline"]["pooled"]["mean"]
        ax.text(tot + 1.5, y, f"{tot:.0f} ms total ({lists[n]['pipeline_fps_pooled']:.1f} FPS)", va="center",
                fontsize=8, color="#444")
    ax.axvline(100, color="#999", lw=1, ls="--")
    ax.text(101, -0.5, "10 FPS target", fontsize=7, color="#666", ha="left", va="top")
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean ms per frame (pipeline boundary: Frame -> FrameResult; io excluded)", fontsize=8)
    ax.set_xlim(0, max(lists[n]["stages_ms"]["pipeline"]["pooled"]["mean"] for n in names) * 1.35)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color="#eee")
    ax.set_axisbelow(True)
    ax.legend(ncol=6, fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(FIG, dpi=110)
    plt.close(fig)
    return FIG.relative_to(REPO).as_posix()


def profile_section(s: Sources) -> list[str]:
    L = ["## Stage profile (05.3)", ""]
    if s.has(EQV):
        e = s.doc(EQV)
        L += [f"**Equivalence (AC2).** Live pipeline vs offline Phase 04 on fr1 (stride 1): detections identical on "
              f"{e['detections_identical_frames']} / {e['n_frames']} frames, tracks identical on "
              f"{e['tracks_identical_frames']} / {e['n_frames']}, schema errors {e['schema_errors']}. "
              f"Source: {link(EQV)}.", ""]
    if not s.has(PROF):
        return L + ["Pending: the timed run (`scripts/profile_pipeline.py`, 3 fresh processes, idle machine) has not "
                    "been done yet.", ""]
    lists = s.get(PROF, "lists")
    lab = s.get(PROF, "label")
    L += [f"Label: **{lab}**. Source: {link(PROF)}. FPS = 1000 / mean pipeline ms (protocol section 5).", ""]
    L += ["| List | detector | detect_total | depth | geometry | track | pipeline median / p95 | FPS (per run) | io |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, v in lists.items():
        st = v["stages_ms"]
        med = lambda k: fmt(st[k]["pooled"]["median"], 1)  # noqa: E731
        L.append(f"| {n} | {med('detector')} | {med('detect_total')} | {med('depth')} | {med('geometry')} | "
                 f"{med('track')} | {med('pipeline')} / {fmt(st['pipeline']['pooled']['p95'], 1)} | "
                 f"{fmt(v['pipeline_fps_pooled'], 2)} ({', '.join(fmt(x, 2) for x in v['pipeline_fps_per_run'])}) | "
                 f"{med('io')} |")
    L += ["", "Stage columns are pooled medians in ms over the runs.", ""]
    fig = stage_figure(s)
    L += [f"![Mean ms per stage]({Path('..') / fig})".replace("\\", "/"), "",
          f"**Bottleneck.** {s.get(PROF, 'bottleneck', 'statement')}", ""]
    return L


def failure_section(s: Sources) -> list[str]:
    c = s.get(GAL, "counts")
    groups = list(c["per_group"])
    cats = list(c["total"])
    L = ["## Failure categories (05.4 R1-R2)", "",
         f"Rule-selected SUN test cases (detected mode), categorised by eye in "
         f"[failure_review.yaml](../docs/failure_review.yaml). Source: {link(GAL)}.", "",
         "| Category | " + " | ".join(groups) + " | total |", "|---|" + "---|" * (len(groups) + 1)]
    for cat in cats:
        L.append(f"| {cat} | " + " | ".join(str(c["per_group"][g].get(cat, 0)) for g in groups)
                 + f" | {c['total'][cat]} |")
    bb = s.get(GAL, "background_bleed")
    L += ["", f"**Does background bleed dominate the largest errors?** "
              f"{'Yes' if bb['dominates'] else 'No'}: {bb['background_bleed']} of {bb['large_error_items']} "
              f"({fmt(bb['share'], pct=True)}). The most common category there is **{bb['next_most_common']}**.", ""]
    for g in groups:
        fig = s.get(GAL, "figures", g)
        L.append(f"- {g}: [{Path(fig).name}](../{fig})")
    return L + [""]


def repro_section(s: Sources) -> list[str]:
    L = ["## Reproducibility (05.4 R3)", ""]
    if not s.has(REP):
        return L + ["Pending: `scripts/repro_check.py` has not been run.", ""]
    items = s.get(REP, "items")
    tol = s.get(REP, "tolerances")
    L += [f"Source: {link(REP)}. Tolerances were fixed before the re-runs.", "",
          "| Item | What | Tolerance | Result | Pass |", "|---|---|---|---|---|"]
    for name, t in tol.items():
        r = items.get(name)
        tl = ("exact" if t["kind"] == "exact" else f"+-{100 * t['tol']:g}%" if t["kind"] == "rel"
              else f"+-{t['tol']:g} {t['unit']}")
        if r is None or r.get("status") == "pending":
            L.append(f"| {name} | {t['what']} | {tl} | not run yet | pending |")
            continue
        if "max_abs_diff" in r:
            res = f"max abs diff {r['max_abs_diff']:.3g} m over {r['n_rerun']} objects"
        elif "differences" in r:
            res = f"{len(r['differences'])} differing values"
        elif "per_list" in r:
            res = "; ".join(f"{n}: {v['rerun']:.1f} vs {v['reported']:.1f} ms ({100 * v['rel_diff']:+.1f}%)"
                            for n, v in r["per_list"].items())
        else:
            res = f"{r['rerun']:.5f} vs {r['reported']:.5f} (diff {r['diff']:+.5f})"
        L.append(f"| {name} | {t['what']} | {tl} | {res} | {'yes' if r['pass'] else 'NO'} |")
    hist = s.get(REP, "history") if "history" in s.doc(REP) else {}
    L.append("")
    for name, runs in hist.items():
        L.append(f"- {name}: {len(runs)} earlier attempt(s) kept in the file; passes: "
                 + ", ".join(("yes" if h["pass"] else "NO") + (f" (start CPU {h['machine_state']['cpu_percent_3s']}%, "
                             f"idle {fmt(h['machine_state']['idle'])})" if "machine_state" in h else "")
                             for h in runs) + ".")
    if "pipeline_median" in items and items["pipeline_median"].get("pass") is False and s.has(PROF):
        sp = {n: v["stages_ms"]["pipeline"] for n, v in s.get(PROF, "lists").items()}
        L.append("- pipeline_median cause (measured): the reported profile's own per-run medians already spread "
                 + "; ".join(f"{n} {min(r['median'] for r in v['runs']):.1f}-{max(r['median'] for r in v['runs']):.1f} ms"
                             for n, v in sp.items())
                 + ". One fresh run against +-15% is tighter than this laptop's run-to-run spread. "
                 "The tolerance was fixed before the run and is not changed.")
    return L + [""]


def limitations(s: Sources) -> list[str]:
    L = ["## Top limitations", "",
         "The full list is in [risks_and_limitations.md](risks_and_limitations.md). Measured here:", ""]
    wo, wd = s.get(LOC, "oracle", "link_iou_lt_0.3_fraction"), s.get(LOC, "detected", "link_iou_lt_0.3_fraction")
    sens = s.get(LOC, "detected", "sensitivity_link_iou_ge_0.3_valid")
    L.append(f"1. **Ground-truth linkage limits the 3D error metric.** {fmt(wo, pct=True)} of oracle and "
             f"{fmt(wd, pct=True)} of detected objects link to a GT 3D box with 2D IoU < 0.3; without them the "
             f"detected median / p90 is {fmt(sens['centre_median_m'])} / {fmt(sens['centre_p90_m'])} m "
             f"(diagnostic only; the headline keeps the protocol linkage). Source: {link(LOC)}.")
    c = s.get(GAL, "counts", "per_group", "no_depth")
    L.append(f"2. **Screens and black surfaces give no depth.** All {sum(c.values())} reviewed no_depth cases are "
             f"'{', '.join(c)}' (monitors, tvs, a keyboard). Source: {link(GAL)}.")
    by = s.get(LOC, "detected", "valid", "by_distance")
    L.append(f"3. **Error grows with distance.** Detected-mode median {fmt(by['0-2']['centre_median_m'])} m at 0-2 m "
             f"vs {fmt(by['>4']['centre_median_m'])} m beyond 4 m (n = {by['>4']['n']}). Source: {link(LOC)}.")
    rec = s.get(LOC, "detected_recall_of_linked_objects")
    L.append(f"4. **Pretrained COCO detector.** Recall of linked GT objects at conf_operational is "
             f"{fmt(rec, pct=True)}; many top false positives are real objects without a SUN label. "
             f"Source: {link(LOC)}, {link(GAL)}.")
    if s.has(PROF):
        sh = max(v["share_of_pipeline_mean"]["preprocess"] + v["share_of_pipeline_mean"]["detect"]
                 for v in s.get(PROF, "lists").values())
        L.append(f"5. **Detection is the CPU bottleneck.** Preprocess + detect is up to {fmt(sh, pct=True)} of the "
                 f"mean pipeline time on the PyTorch CPU baseline; Phase 06 targets it. Source: {link(PROF)}.")
    return L + [""]


def main() -> int:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    proto = check_protocol(REPO, index)
    s = Sources(REPO, index)
    L = ["# Results summary (Phase 05)", "",
         "Generated by `scripts/make_results_summary.py` from `benchmarks/results/index.json`; do not edit by hand. "
         "Every value is Measured and links to its source file. Targets are named as targets.", "",
         f"- Index written: {index['created_utc']}; frozen protocol sha256 `{proto[:12]}` (matches the index)",
         f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "",
         "## Goals G1-G7", ""] + goals(s) + [""]
    L += profile_section(s) + failure_section(s) + repro_section(s) + limitations(s)
    L += ["## Sources used", "", "| File | sha256 (from the index) |", "|---|---|"]
    L += [f"| {link(p)} | `{h[:12]}` |" for p, h in sorted(s.used.items())]
    DOC.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {DOC.relative_to(REPO)} from {len(s.used)} indexed sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
