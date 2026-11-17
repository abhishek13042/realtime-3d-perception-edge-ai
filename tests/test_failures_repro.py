"""Spec 05.4: failure-gallery selection, repro comparators, and the summary's source guard."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from perception.eval.failures import GROUPS, category_counts, select
from perception.eval.repro import TOLERANCES, compare_arrays, compare_exact, compare_scalar, within
from perception.eval.results_index import sha256_file
from perception.eval.summary import SourceError, Sources, fmt


def synthetic(seed: int = 0, n_img: int = 30) -> dict:
    """Oracle + detected objects over n_img images, with ties in error and score to exercise the tie-break."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_img):
        iid = f"{i:06d}"
        for k in range(4):
            cls = ["chair", "table", "bed", "sofa", "lamp"][(i + k) % 5]
            gt = np.array([k * 0.5, 0.1 * i, 2.0 + k], float)
            box = np.array([10 * k, 5 * k, 10 * k + 40 + i, 5 * k + 30 + k], float)
            rows.append(("oracle", iid, cls, box, gt, gt + 0.01, np.nan, "valid"))
            if (i + k) % 3:                                    # some GT objects are missed by the detector
                status = ["valid", "no_depth", "valid", "low_coverage"][k]
                c = gt + (round(rng.uniform(0, 1), 1) if status == "valid" else 0.0)  # rounded -> ties
                if status == "no_depth":
                    c = np.full(3, np.nan)
                rows.append(("detected", iid, cls, box, gt, c, round(rng.uniform(0.3, 0.9), 1), status))
    z = {"o_mode": np.array([r[0] for r in rows]), "o_image_id": np.array([r[1] for r in rows]),
         "o_cls": np.array([r[2] for r in rows]), "o_raw_cls": np.array([r[2] for r in rows]),
         "o_box": np.stack([r[3] for r in rows]), "o_gt": np.stack([r[4] for r in rows]),
         "o_centre": np.stack([r[5] for r in rows]), "o_score": np.array([r[6] for r in rows], float),
         "o_status": np.array([r[7] for r in rows]), "o_sensor": np.array(["kv2"] * len(rows)),
         "o_valid_frac": np.full(len(rows), 0.5), "o_link_iou": np.full(len(rows), 0.6)}
    nb = 3 * n_img
    z.update({"b_image_id": np.array([f"{i % n_img:06d}" for i in range(nb)]),
              "b_cls": np.array(["chair"] * nb), "b_box": np.tile([1.0, 2.0, 30.0, 40.0], (nb, 1)),
              "b_score": np.round(rng.uniform(0.25, 0.95, nb), 1), "b_matched_any": np.arange(nb) % 4 == 0,
              "b_status": np.array(["valid"] * nb), "b_sensor": np.array(["kv2"] * nb)})
    return z


def test_selection_deterministic_and_order_invariant():
    z = synthetic()
    a = select(z)
    assert json.dumps(a) == json.dumps(select(synthetic()))
    perm = np.random.default_rng(1).permutation(len(z["o_mode"]))
    zp = {k: (v[perm] if k.startswith("o_") else v) for k, v in z.items()}
    assert json.dumps(a) == json.dumps(select(zp))             # full tie-break: row order does not matter


def test_selection_rules():
    items = select(synthetic())
    for g, n in GROUPS.items():
        assert len(items[g]) <= n
        ids = [it["image_id"] for it in items[g]]
        assert len(ids) == len(set(ids)), f"{g}: one item per image"
    errs = [it["err3d_m"] for it in items["large_error"]]
    assert errs == sorted(errs, reverse=True) and all(it["status"] == "valid" for it in items["large_error"])
    assert all(it["status"] == "no_depth" for it in items["no_depth"])
    fp = [it["score"] for it in items["false_positive"]]
    assert fp == sorted(fp, reverse=True)
    keys = [it["key"] for v in items.values() for it in v]
    assert len(keys) == len(set(keys))


def test_category_counts_rejects_unreviewed_and_unknown():
    items = select(synthetic())
    keys = [it["key"] for v in items.values() for it in v]
    review = {k: {"category": "occluder"} for k in keys}
    c = category_counts(review, items)
    assert c["total"] == {"occluder": len(keys)}
    with pytest.raises(ValueError, match="no review entry"):
        category_counts({k: review[k] for k in keys[1:]}, items)
    with pytest.raises(ValueError, match="unknown categories"):
        category_counts({**review, keys[0]: {"category": "bad luck"}}, items)


@pytest.mark.parametrize("name", ["sun_test_200", "detection_map", "pipeline_median"])
def test_scalar_tolerance_edges(name):
    t = TOLERANCES[name]
    ref = 0.5 if name == "detection_map" else 100.0
    step = t["tol"] * (abs(ref) if t["kind"] == "rel" else 1.0)
    assert within(ref, ref + 0.999 * step, t["kind"], t["tol"])
    assert within(ref, ref - 0.999 * step, t["kind"], t["tol"])
    assert not within(ref, ref + 1.001 * step, t["kind"], t["tol"])
    if name != "sun_test_200":
        assert compare_scalar(name, ref, ref + 0.5 * step)["pass"]
        assert not compare_scalar(name, ref, ref - 2 * step)["pass"]


def test_within_nan_and_unknown_kind():
    assert within(math.nan, math.nan, "abs", 0.0)
    assert not within(1.0, math.nan, "abs", 1.0)
    with pytest.raises(ValueError):
        within(1.0, 1.0, "fuzzy", 0.0)


def test_compare_arrays():
    ref = np.array([0.1, np.nan, 0.3])
    assert compare_arrays("sun_test_200", ref, ref.copy())["pass"]
    assert compare_arrays("sun_test_200", ref, ref + np.array([5e-10, 0, 0]))["pass"]
    assert not compare_arrays("sun_test_200", ref, ref + np.array([2e-9, 0, 0]))["pass"]
    r = compare_arrays("sun_test_200", ref, np.array([0.1, 0.2, 0.3]))
    assert not r["pass"] and r["nan_mismatch"] == 1
    assert not compare_arrays("sun_test_200", ref, ref[:2])["pass"]


def test_compare_exact():
    ref = {"a": 1.0, "b": {"c": [1, 2, {"d": 0.178}]}}
    assert compare_exact("fr3_s1_metrics", ref, json.loads(json.dumps(ref)))["pass"]
    bad = {"a": 1.0, "b": {"c": [1, 2, {"d": 0.1780001}]}}
    r = compare_exact("fr3_s1_metrics", ref, bad)
    assert not r["pass"] and r["differences"] == ["/b/c[2]/d: 0.178 != 0.1780001"]
    assert not compare_exact("fr3_s1_metrics", ref, {"a": 1.0})["pass"]


def test_sources_guard(tmp_path):
    good = tmp_path / "r" / "good.json"
    good.parent.mkdir()
    good.write_text(json.dumps({"x": {"y": 0.25}}), encoding="utf-8")
    other = tmp_path / "r" / "other.json"
    other.write_text("{}", encoding="utf-8")
    index = {"files": [{"path": "r/good.json", "file_sha256": sha256_file(good)},
                       {"path": "r/gone.json", "file_sha256": "0" * 64}]}
    s = Sources(tmp_path, index)
    assert s.get("r/good.json", "x", "y") == 0.25 and "r/good.json" in s.used
    with pytest.raises(SourceError, match="not in the results index"):
        s.get("r/other.json")
    with pytest.raises(SourceError, match="missing on disk"):
        s.get("r/gone.json")
    with pytest.raises(SourceError, match="no value"):
        s.get("r/good.json", "x", "z")
    good.write_text(json.dumps({"x": {"y": 0.3}}), encoding="utf-8")
    with pytest.raises(SourceError, match="changed since the index"):
        Sources(tmp_path, index).get("r/good.json")
    assert not s.has("r/other.json") and not s.has("r/gone.json")


def test_fmt():
    assert fmt(None) == "n/a" and fmt(True) == "yes" and fmt(0.1475, pct=True) == "14.8%"
    assert fmt(0.24271) == "0.243" and fmt(5, 2, " m") == "5.00 m"


def test_repro_main_keeps_history_and_skips_not_idle(tmp_path, monkeypatch):
    import importlib
    import sys
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    rc = importlib.import_module("repro_check")
    out = tmp_path / "repro_check.json"
    monkeypatch.setattr(rc, "OUT", out)
    monkeypatch.setattr(rc, "rel", lambda q: q.name)
    monkeypatch.setattr(rc, "git_state", lambda repo: {"git_commit": "test"})
    results = iter([{"item": "x", "pass": True}, {"item": "x", "pass": False},
                    {"item": "x", "pass": None, "status": "not_idle"}])
    monkeypatch.setattr(rc, "ITEMS", {"x": lambda: next(results)})
    assert rc.main(["--items", "x"]) == 0
    assert rc.main(["--items", "x"]) == 1                      # a failing item makes the exit code non-zero
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["items"]["x"]["pass"] is False and [h["pass"] for h in doc["history"]["x"]] == [True]
    rc.main(["--items", "x"])                                  # not_idle: printed, nothing stored
    doc2 = json.loads(out.read_text(encoding="utf-8"))
    assert doc2["items"]["x"]["pass"] is False and len(doc2["history"]["x"]) == 1
    assert doc2["all_pass"] is False
    sys.modules.pop("repro_check", None)
