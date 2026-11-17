"""Spec 04.1 tests: ByteTrack adapter, 3D attachment, track history."""

from __future__ import annotations

import math

import numpy as np
import pytest

from perception.data.frame import Frame
from perception.depth.estimate import AssocParams
from perception.geometry.camera import make_K
from perception.tracking.bytetrack import Tracker, TrackerParams
from perception.tracking.track_history import (TrackHistory, attach_3d, frame_line, observations_of, read_tracks,
                                               write_tracks)
from perception.tracking.runner import track_sequence

P = TrackerParams()  # Ultralytics defaults with high/new = 0.2411, buffer 30, class-aware


def box(x, y=100.0, w=60.0, h=80.0):
    return [x, y, x + w, y + h]


def ids(out):
    return [t.track_id for t in out]


def test_constant_velocity_keeps_one_id():
    t = Tracker(P)
    seen = set()
    for f in range(50):
        out = t.update([box(20 + 4 * f)], [0.9], [0])
        assert len(out) == 1
        seen.add(out[0].track_id)
    assert seen == {1}


# Measured Ultralytics 8.4.208 behaviour: a lost track survives track_buffer + 1 missed frames (31 at buffer 30)
# and is removed at track_buffer + 2. AC1 needs persistence up to track_buffer, which holds.
@pytest.mark.parametrize("gap,same", [(10, True), (30, True), (31, True), (32, False), (45, False)])
def test_occlusion_gap_vs_buffer(gap, same):
    t = Tracker(P)
    for _ in range(10):
        first = t.update([box(100)], [0.9], [0])[0].track_id
    for _ in range(gap):
        assert t.update(np.empty((0, 4)), [], []) == []  # coasting tracks are never output (R1)
    out = t.update([box(100)], [0.9], [0])
    if same:
        assert ids(out) == [first]
    else:
        # A new track is unconfirmed on its first frame (Ultralytics), so it shows from the next frame on, new ID.
        assert out == []
        out = t.update([box(100)], [0.9], [0])
        assert len(out) == 1 and out[0].track_id != first


def test_classes_do_not_swap_ids_when_overlapping():
    t = Tracker(P)
    a_ids, b_ids = set(), set()
    for f in range(40):
        # Two boxes of different classes cross each other.
        ba, bb = box(50 + 6 * f), box(290 - 6 * f)
        out = {o.cls: o.track_id for o in t.update([ba, bb], [0.9, 0.9], [3, 7])}
        a_ids.add(out[3])
        b_ids.add(out[7])
    assert len(a_ids) == 1 and len(b_ids) == 1 and a_ids != b_ids


def test_class_agnostic_switch_lets_class_change_continue_track():
    t = Tracker(P.with_(class_aware=False))
    t.update([box(100)], [0.9], [1])
    i1 = t.update([box(100)], [0.9], [1])[0].track_id
    out = t.update([box(101)], [0.9], [2])
    assert ids(out) == [i1] and out[0].cls == 2
    t2 = Tracker(P)
    t2.update([box(100)], [0.9], [1])
    i2 = t2.update([box(100)], [0.9], [1])[0].track_id
    assert all(o.track_id != i2 for o in t2.update([box(101)], [0.9], [2]))


def test_ids_unique_across_class_trackers():
    t = Tracker(P)
    all_ids = []
    for f in range(5):
        all_ids += ids(t.update([box(10), box(200), box(400)], [0.9, 0.9, 0.9], [0, 1, 2]))
    assert sorted(set(all_ids)) == [1, 2, 3]


def test_low_confidence_detection_is_ignored():
    t = Tracker(P)
    for _ in range(5):
        assert t.update([box(100)], [0.05], [0]) == []  # below track_low_thresh (0.1)


def test_low_score_detection_continues_but_never_starts_a_track():
    t = Tracker(P)
    for _ in range(3):
        assert t.update([box(100)], [0.15], [0]) == []  # 0.1 < s < 0.2411 -> second-stage only, cannot start
    t.update([box(100)], [0.9], [0])
    i = t.update([box(100)], [0.9], [0])[0].track_id
    out = t.update([box(101)], [0.15], [0])
    assert ids(out) == [i] and out[0].score == pytest.approx(0.15)


def test_output_refers_to_input_rows():
    t = Tracker(P)
    boxes = [box(10), box(200), box(10), box(400)]
    for _ in range(3):
        out = t.update(boxes, [0.9, 0.05, 0.8, 0.9], [0, 0, 1, 2])
    assert [o.det_index for o in out] == [0, 2, 3]
    for o in out:
        assert list(o.box_xyxy) == boxes[o.det_index]


def test_class_tracker_created_mid_sequence_needs_confirmation():
    t = Tracker(P)
    for _ in range(5):
        t.update([box(10)], [0.9], [0])
    assert [o.cls for o in t.update([box(10), box(300)], [0.9, 0.9], [0, 5])] == [0]
    assert sorted(o.cls for o in t.update([box(10), box(300)], [0.9, 0.9], [0, 5])) == [0, 5]


def test_determinism():
    rng = np.random.default_rng(0)
    recs = []
    for f in range(60):
        n = int(rng.integers(0, 6))
        x = rng.uniform(0, 500, n)
        recs.append({"frame_id": f"{f:.6f}", "boxes": np.stack([x, x * 0 + 50, x + 40, x * 0 + 120], 1).tolist(),
                     "scores": rng.uniform(0, 1, n).tolist(), "class_ids": rng.integers(0, 3, n).tolist()})
    noop = lambda fi, si, tracked: []  # noqa: E731
    a = [[(o.track_id, o.det_index) for o in Tracker(P).update(r["boxes"], r["scores"], r["class_ids"])]
         for r in recs]
    l1, _ = track_sequence(recs, P, 2, noop)
    l2, _ = track_sequence(recs, P, 2, noop)
    strip = lambda ls: [{k: v for k, v in d.items() if k != "timings_ms"} for d in ls]  # noqa: E731
    assert strip(l1) == strip(l2) and len(l1) == 30 and a is not None


def test_params_from_config_per_stride():
    cfg = {"bytetrack": {"track_high_thresh": 0.3, "track_low_thresh": 0.1, "new_track_thresh": 0.3,
                         "track_buffer": 30, "match_thresh": 0.8, "fuse_score": True},
           "class_aware": False, "per_stride": {6: {"track_buffer": 15}}}
    assert TrackerParams.from_config(cfg).track_buffer == 30
    p6 = TrackerParams.from_config(cfg, stride=6)
    assert p6.track_buffer == 15 and p6.class_aware is False
    with pytest.raises(ValueError):
        TrackerParams.from_config(cfg, bogus=1)


# ---- 3D attachment and history ------------------------------------------------------------------------------------

def plane_frame(z=2.0, T=None):
    h, w = 120, 160
    depth = np.full((h, w), z, np.float32)
    return Frame(rgb=np.zeros((h, w, 3), np.uint8), depth_m=depth, valid=np.ones((h, w), bool),
                 K=make_K(100.0, 100.0, 80.0, 60.0), frame_id="0.000000", dataset="synthetic", scene_id="s",
                 T_world_cam=T, timestamp=0.0)


def test_world_transform_matches_analytic():
    yaw = math.radians(90)
    T = np.eye(4)
    T[:3, :3] = [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
    T[:3, 3] = [1.0, 2.0, 3.0]
    fr = plane_frame(2.0, T)
    t = Tracker(P)
    out = t.update([[60, 40, 100, 80]], [0.9], [0])  # box centred on the principal point
    obs = attach_3d(fr, out, AssocParams(shrink=0.7, estimator="A"), 3.0, 0.02, 0, 0)[0]
    assert obs.status == "valid"
    x, y, z = obs.centre_cam
    assert z == pytest.approx(2.0) and abs(x) < 0.02 and abs(y) < 0.02
    expect = T[:3, :3] @ np.array(obs.centre_cam) + T[:3, 3]
    np.testing.assert_allclose(obs.centre_world, expect, atol=1e-12)
    np.testing.assert_allclose(obs.centre_world, [1.0 - y, 2.0 + x, 5.0], atol=1e-12)


def test_no_pose_gives_no_world_centre_and_no_depth_gives_no_centre():
    fr = plane_frame(2.0, None)
    t = Tracker(P)
    o = attach_3d(fr, t.update([[60, 40, 100, 80]], [0.9], [0]), AssocParams(), 3.0, 0.02, 0, 0)[0]
    assert o.centre_cam is not None and o.centre_world is None
    fr0 = plane_frame(2.0, np.eye(4))
    fr0.valid[:] = False
    fr0.depth_m[:] = 0.0
    o0 = attach_3d(fr0, t.update([[60, 40, 100, 80]], [0.9], [0]), AssocParams(), 3.0, 0.02, 1, 1)[0]
    assert o0.status == "no_depth" and o0.centre_cam is None and o0.centre_world is None


def test_jsonl_roundtrip_preserves_ids_and_centres(tmp_path):
    fr = plane_frame(2.0, np.eye(4))
    t = Tracker(P)
    lines = []
    for f in range(4):
        obs = attach_3d(fr, t.update([[60, 40, 100, 80], [10, 10, 40, 40]], [0.9, 0.8], [0, 1]),
                        AssocParams(), 3.0, 0.02, f, f)
        lines.append(frame_line(f, f, f"{f:.6f}", float(f), 2, obs, {"tracker": 0.1}))
    path = tmp_path / "run.jsonl"
    write_tracks(path, {"run": "t"}, lines)
    header, back = read_tracks(path)
    assert header == {"run": "t"} and len(back) == 4
    h1, h2 = TrackHistory.from_frames(lines), TrackHistory.from_frames(back)
    assert sorted(h1.tracks) == sorted(h2.tracks) == [1, 2]
    for i in h1.tracks:
        for a, b in zip(h1.tracks[i], h2.tracks[i]):
            assert a.track_id == b.track_id and a.status == b.status
            np.testing.assert_allclose(a.centre_cam, b.centre_cam, atol=1e-6)
            np.testing.assert_allclose(a.centre_world, b.centre_world, atol=1e-6)
    assert len(TrackHistory.from_frames(back).confirmed(4)) == 2
    assert observations_of(back[0])[0].frame_index == 0
