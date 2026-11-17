"""Spec 05.3 tests: FrameResult schema and null rules, stage timings, and equality with the offline 04.1 runner."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from perception.config import load_config
from perception.data.frame import Frame
from perception.detection.base import Detections
from perception.pipeline import TIMING_KEYS, Pipeline, validate_result

REPO = Path(__file__).resolve().parents[1]
DEPTH, TRACKER = load_config("depth"), load_config("tracker")
CONF_OP = float(load_config("detector")["conf_operational"])
FR1_RUN = REPO / "benchmarks" / "detection" / "tum_fr1_desk_full.jsonl"


class StubDetector:
    """Returns the queued Detections in order (no model)."""

    def __init__(self, dets):
        self.queue = list(dets)

    def detect(self, rgb):
        t0 = time.perf_counter()
        d = self.queue.pop(0)
        ms = (time.perf_counter() - t0) * 1e3 / 3                   # real elapsed time, split over the 3 stages
        d.timings_ms = {"preprocess": ms, "infer": ms, "postprocess": ms}
        return d


def _dets(boxes, scores, classes):
    return Detections(np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32),
                      np.asarray(classes, np.int64))


def _frame(depth_fill=2.0, hole=None, fid="f0", ts=1.0):
    h, w = 120, 160
    depth = np.full((h, w), depth_fill, np.float32)
    if hole is not None:
        x1, y1, x2, y2 = hole
        depth[y1:y2, x1:x2] = 0.0
    K = np.array([[100.0, 0, 80], [0, 100.0, 60], [0, 0, 1]])
    return Frame(np.zeros((h, w, 3), np.uint8), depth, depth > 0, K, fid, "synthetic", "s", np.eye(4), ts)


def _pipe(dets, track=True):
    return Pipeline(StubDetector(dets), DEPTH, TRACKER, 1, conf_operational=CONF_OP, track=track)


def test_frame_result_schema_and_json_round_trip():
    p = _pipe([_dets([[20, 20, 60, 70], [100, 30, 140, 90], [5, 5, 10, 10]], [0.9, 0.5, 0.05], [56, 41, 0])])
    r = p.process(_frame())
    d = r.to_dict()
    validate_result(d)
    back = json.loads(json.dumps(d, allow_nan=False))
    assert back == d
    assert len(d["objects"]) == 2                                   # the 0.05 box is below conf_operational
    o = d["objects"][0]
    assert o["class_name"] == "chair" and o["depth_status"] == "valid"
    assert o["depth_m"] == pytest.approx(2.0) and o["center_cam_m"][2] == pytest.approx(2.0)
    assert o["track_id"] is None                                     # first frame: not yet confirmed


def test_no_depth_box_has_null_depth_fields():
    p = _pipe([_dets([[20, 20, 60, 70]], [0.9], [56])])
    d = p.process(_frame(hole=(0, 0, 100, 100))).to_dict()
    validate_result(d)
    o = d["objects"][0]
    assert o["depth_status"] == "no_depth"
    assert o["depth_m"] is None and o["depth_spread_m"] is None and o["center_cam_m"] is None
    assert o["valid_depth_fraction"] == 0.0
    bad = json.loads(json.dumps(d))
    bad["objects"][0]["depth_m"] = 1.0
    with pytest.raises(ValueError, match="no_depth"):
        validate_result(bad)


def test_total_at_least_sum_of_stages_and_track_confirmation():
    box = [[20, 20, 60, 70]]
    p = _pipe([_dets(box, [0.9], [56]) for _ in range(4)])
    ids = []
    for i in range(4):
        r = p.process(_frame(fid=f"f{i}", ts=float(i)))
        stages = sum(r.timing_ms[k] for k in TIMING_KEYS if k != "total")
        assert r.timing_ms["total"] >= stages
        ids.append(r.objects[0].track_id)
    # min_track_len = 3: the ID appears from the third matched frame on, and it is stable.
    assert ids[:2] == [None, None] and ids[2] is not None and ids[2] == ids[3]


def test_single_image_mode_never_assigns_ids():
    p = _pipe([_dets([[20, 20, 60, 70]], [0.9], [56]) for _ in range(4)], track=False)
    for i in range(4):
        assert all(o.track_id is None and o.raw_track_id is None for o in p.process(_frame()).objects)


@pytest.mark.skipif(not FR1_RUN.exists(), reason="fr1 detection run not present")
def test_stub_pipeline_equals_offline_runner_on_20_fr1_frames():
    from perception.data.tum import TUMSequence
    from perception.detection.runio import read_run, record_to_detections
    from perception.tracking.bytetrack import TrackerParams
    from perception.tracking.runner import Centres3D, align_records, track_sequence
    from perception.tracking.track_history import frame_line

    try:
        seq = TUMSequence.from_config("rgbd_dataset_freiburg1_desk", load_config("dataset"))
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"fr1 data not present: {e}")
    records = align_records(seq, read_run(FR1_RUN)[1])[:20]
    params = TrackerParams.from_config(TRACKER, stride=1)
    offline, _ = track_sequence(records, params, 1, Centres3D(seq, DEPTH))

    p = _pipe([record_to_detections(r) for r in records])
    for i, ref in enumerate(offline):
        r = p.process(seq[i], seq_index=i)
        line = frame_line(r.frame_index, r.seq_index, r.frame_id, float(r.frame_id), r.n_dets, r.observations(), {})
        assert line["tracks"] == ref["tracks"], f"frame {i} differs"
        assert line["n_dets"] == ref["n_dets"]
        validate_result(r.to_dict())
