"""Spec 05.2 tests: bootstrap, one-shot guard, freeze check, breakdown counts, G4 status check."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from perception.eval.localization import BUCKETS, breakdown
from perception.eval.results_index import check_frozen
from perception.eval.sun_test import (bootstrap_ci, bootstrap_ratio, check_status_complete, log_rerun,
                                      one_shot_guard)

REPO = Path(__file__).resolve().parents[1]


def test_bootstrap_ci_contains_true_median():
    rng = np.random.default_rng(1)
    true_median = 0.30
    x = rng.lognormal(np.log(true_median), 0.6, 900)       # median of a lognormal = exp(mu)
    groups = np.repeat(np.arange(300), 3)                   # 3 objects per "image"
    ci = bootstrap_ci(x, groups, n_boot=500, seed=0)["median"]
    assert ci["lo"] <= true_median <= ci["hi"]
    assert ci["lo"] < ci["point"] < ci["hi"]
    # deterministic for a fixed seed
    assert bootstrap_ci(x, groups, n_boot=500, seed=0)["median"] == ci


def test_bootstrap_ratio_known_ratio():
    rng = np.random.default_rng(2)
    a = rng.lognormal(np.log(0.6), 0.3, 400)
    b = rng.lognormal(np.log(0.3), 0.3, 400)
    v = np.r_[a, b]
    num = np.r_[np.ones(400, bool), np.zeros(400, bool)]
    r = bootstrap_ratio(v, num, ~num, np.arange(800) // 4, n_boot=300, seed=0)
    assert r["lo"] <= 2.0 <= r["hi"]


def test_one_shot_guard(tmp_path):
    out = tmp_path / "localization_sun_test.json"
    assert one_shot_guard(out, None) is None                # first run
    out.write_text(json.dumps({"created_utc": "t0"}))
    with pytest.raises(PermissionError, match="one-shot"):
        one_shot_guard(out, None)
    with pytest.raises(PermissionError):
        one_shot_guard(out, "   ")
    rec = one_shot_guard(out, "bug in X fixed")
    assert rec["reason"] == "bug in X fixed" and rec["previous_created_utc"] == "t0"
    prog = tmp_path / "PROGRESS.md"
    prog.write_text("# Log\n")
    log_rerun(prog, "benchmarks/results/localization_sun_test.json", rec)
    log_rerun(prog, "benchmarks/results/localization_sun_test.json", rec)
    t = prog.read_text()
    assert t.count("## Phase 05 re-runs") == 1 and t.count("bug in X fixed") == 2


def test_freeze_check_refuses_modified_depth_yaml(tmp_path):
    for f in ("docs/evaluation_protocol.md", "configs/depth.yaml", "configs/detector.yaml", "configs/class_map.yaml",
              "configs/tracker.yaml", "configs/splits/sunrgbd_test.txt", "configs/splits/sunrgbd_val.txt"):
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / f, tmp_path / f)
    names = ["configs/depth.yaml", "configs/detector.yaml", "configs/splits/sunrgbd_test.txt"]
    assert set(check_frozen(tmp_path, names)) == set(names)
    p = tmp_path / "configs" / "depth.yaml"
    p.write_text(p.read_text().replace("shrink: 0.7", "shrink: 0.6"))
    with pytest.raises(PermissionError, match="configs/depth.yaml"):
        check_frozen(tmp_path, names)


def test_breakdown_counts_sum_to_total():
    rng = np.random.default_rng(3)
    n = 500
    err, dz = rng.random(n), rng.normal(0, 0.1, n)
    gz = rng.uniform(0.3, 7.0, n)
    sensor = rng.choice(["kv1", "kv2", "realsense", "xtion"], n)
    mask = rng.random(n) < 0.8
    out = breakdown(err, dz, gz, sensor, mask)
    total = out["all"]["n"]
    assert total == mask.sum()
    assert sum(out["by_distance"][b]["n"] for b, _, _ in BUCKETS) == total
    assert sum(s["n"] for s in out["by_sensor"].values()) == total


def test_g4_status_check():
    assert check_status_complete(["valid", "low_coverage", "no_depth"])["coverage"] == 1.0
    with pytest.raises(AssertionError, match="G4"):
        check_status_complete(["valid", None, "no_depth"])
    with pytest.raises(AssertionError):
        check_status_complete(["valid", ""])
