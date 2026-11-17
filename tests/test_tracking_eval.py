"""Spec 04.2 tests: label-free tracking metrics, motion trigger, split guard."""

from __future__ import annotations

import math

import numpy as np
import pytest

from perception.eval.tracking import (REPORT_SEQUENCE, TUNE_SEQUENCE, camera_speeds, evaluate, interruptions,
                                      jumps_in_track, motion_trigger, split_guard, stability_rms)
from perception.tracking.track_history import Observation


def obs(fi, centre, tid=1, cls=0, status="valid", si=None):
    c = None if centre is None else tuple(float(x) for x in centre)
    return Observation(fi, fi if si is None else si, float(fi), tid, cls, 0, (0, 0, 1, 1), 0.9, status, 100, c, c)


def test_stability_of_static_track_is_sqrt3_sigma():
    rng = np.random.default_rng(0)
    sigma = 0.03
    pts = np.array([1.0, 2.0, 3.0]) + rng.normal(0, sigma, (4000, 3))
    assert stability_rms(pts) == pytest.approx(math.sqrt(3) * sigma, rel=0.03)


def test_jump_detector_fires_on_1m_not_on_noise():
    rng = np.random.default_rng(1)
    pts = rng.normal(0, 0.2 / math.sqrt(3), (50, 3))  # ~0.2 m RMS noise; consecutive differences stay < 0.75 m
    pts = np.clip(pts, -0.2, 0.2)
    track = [obs(i, p) for i, p in enumerate(pts)]
    assert jumps_in_track(1, track, 0.75)[0] == []
    pts[25:] += [1.0, 0.0, 0.0]
    j, pairs = jumps_in_track(1, [obs(i, p) for i, p in enumerate(pts)], 0.75)
    assert len(j) == 1 and (j[0].frame_a, j[0].frame_b) == (24, 25) and pairs == 49


def test_jumps_skip_non_valid_centres():
    track = [obs(0, [0, 0, 0]), obs(1, [5, 0, 0], status="low_coverage"), obs(2, [0.1, 0, 0])]
    j, pairs = jumps_in_track(1, track, 0.75)
    assert j == [] and pairs == 1


def test_recovery_counting_on_hand_made_gaps():
    # Processed frames 0..99, buffer 10. Resumed gaps: 5..7, 10..19 (10 missing = buffer) and 21: 3 recoveries.
    frames = [0, 1, 2, 3, 4, 8, 9, 20, 22, 23]
    t = [obs(f, [0, 0, 0]) for f in frames]
    rec, lost, at = interruptions(t, 100, 10)
    assert (rec, lost, at) == (3, 1, [23])        # ends at 23 with 76 frames left >= buffer + 2 -> a loss
    rec, lost, at = interruptions(t, 34, 10)
    assert (rec, lost) == (3, 0)                  # 10 frames left < 12: censored (could still resume)
    assert interruptions(t, 35, 10)[1] == 0       # 11 left: Ultralytics still keeps the track (buffer + 1)
    assert interruptions(t, 36, 10)[1] == 1       # 12 left: certainly lost


def test_evaluate_excludes_movable_classes_and_short_tracks():
    tracks = {1: [obs(i, [0, 0, 0], tid=1, cls=56) for i in range(6)],          # chair, static
              2: [obs(i, [i, 0, 0], tid=2, cls=0) for i in range(6)],           # person, excluded
              3: [obs(i, [0, 0, 0], tid=3, cls=56) for i in range(2)]}          # too short
    names = ["person"] + ["x"] * 55 + ["chair"]
    r = evaluate(tracks, 6, 0, min_len=3, buffer=30, excluded_ids={0}, names=names,
                 inventory={"chair": {"n": 2, "range": [1, 3]}})
    assert r["n_static_confirmed"] == 1 and r["id_switch_proxy"]["0.75"]["n_jumps"] == 0
    assert r["fragmentation"]["overall"] == 0.5 and r["fragmentation"]["overall_range"] == [round(1 / 3, 3), 1.0]


def test_camera_speeds_analytic():
    t = np.arange(10) * 0.1
    T = np.repeat(np.eye(4)[None], 10, 0)
    for i in range(10):
        a = math.radians(9 * i)                     # 90 deg/s about z
        T[i, :3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        T[i, :3, 3] = [0.5 * t[i], 0, 0]            # 0.5 m/s
    lin, ang = camera_speeds(T, t)
    np.testing.assert_allclose(lin, 0.5, atol=1e-9)
    np.testing.assert_allclose(ang, 90.0, atol=1e-6)


def test_motion_trigger_fires_and_does_not_fire():
    rng = np.random.default_rng(2)
    all_s = rng.gamma(2.0, 0.1, 2000)
    fast = rng.gamma(2.0, 0.1, 40) + 0.4
    assert motion_trigger(fast, all_s)["fires"] is True
    assert motion_trigger(rng.choice(all_s, 40), all_s)["fires"] is False
    assert motion_trigger(fast[:10], all_s)["fires"] is False   # fewer than 20 loss events


def test_split_guard():
    split_guard("tune", TUNE_SEQUENCE)
    with pytest.raises(PermissionError):
        split_guard("tune", REPORT_SEQUENCE)
    split_guard("report", REPORT_SEQUENCE)
    with pytest.raises(ValueError):
        split_guard("train", TUNE_SEQUENCE)


def test_eval_script_refuses_fr3_tuning():
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    p = subprocess.run([sys.executable, str(repo / "scripts" / "eval_tracking.py"), "--stage", "tune",
                        "--sequence", "fr3"], capture_output=True, text=True)
    assert p.returncode != 0 and "allowed on" in (p.stderr + p.stdout)
