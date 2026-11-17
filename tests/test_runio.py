"""Spec 02.2 tests: JSONL round trip, resume, header guard, latency statistics."""

import numpy as np
import pytest

from perception.detection.base import Detections
from perception.detection.runio import (RunWriter, completed_ids, frame_record, latency_stats, read_run,
                                        record_to_detections)

HEADER = {"weights_sha256": "abc", "head": "one2one", "imgsz": 640, "conf": 0.001, "max_det": 300,
          "iou_nms": None, "config_sha256": "cfg"}


def _det(seed=0, n=3):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 300, (n, 2)).astype(np.float32)
    boxes = np.concatenate([xy, xy + rng.uniform(1, 100, (n, 2)).astype(np.float32)], 1)
    return Detections(boxes, np.sort(rng.random(n).astype(np.float32))[::-1].copy(),
                      rng.integers(0, 80, n).astype(np.int64))


def _rec(i, det=None):
    return frame_record(det if det is not None else _det(i), f"{i:06d}", "sunrgbd", "s", "kv1", (530, 730),
                        {"infer": 1.0})


def test_roundtrip_exact(tmp_path):
    d = _det(1, 5)
    with RunWriter(tmp_path / "r.jsonl", HEADER, resume=False) as w:
        w.write(_rec(1, d))
    header, recs = read_run(tmp_path / "r.jsonl")
    back = record_to_detections(recs[0])
    assert header == HEADER
    assert np.array_equal(back.boxes, d.boxes) and np.array_equal(back.scores, d.scores)
    assert np.array_equal(back.class_ids, d.class_ids) and back.boxes.dtype == np.float32


def test_empty_detections_roundtrip(tmp_path):
    with RunWriter(tmp_path / "r.jsonl", HEADER, resume=False) as w:
        w.write(_rec(1, Detections.empty()))
    back = record_to_detections(read_run(tmp_path / "r.jsonl")[1][0])
    assert back.boxes.shape == (0, 4) and len(back) == 0


def test_resume_skips_done_and_drops_truncated_tail(tmp_path):
    p = tmp_path / "r.jsonl"
    with RunWriter(p, HEADER, resume=False) as w:
        for i in (1, 2, 3):
            w.write(_rec(i))
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"frame_id": "000004", "boxes": [[1,')  # crash mid-line
    assert completed_ids(p) == {"000001", "000002", "000003"}
    todo = [f"{i:06d}" for i in range(1, 7) if f"{i:06d}" not in completed_ids(p)]
    assert todo[0] == "000004"
    with RunWriter(p, HEADER, resume=True) as w:
        w.write(_rec(4))
    assert [r["frame_id"] for r in read_run(p)[1]] == ["000001", "000002", "000003", "000004"]


def test_refuses_overwrite_and_mismatched_resume(tmp_path):
    p = tmp_path / "r.jsonl"
    RunWriter(p, HEADER, resume=False).close()
    with pytest.raises(FileExistsError):
        RunWriter(p, HEADER, resume=False)
    with pytest.raises(ValueError, match="conf"):
        RunWriter(p, {**HEADER, "conf": 0.25}, resume=True)


def test_latency_stats_known_values():
    s = latency_stats(list(range(1, 101)))  # 1..100
    assert s["n"] == 100 and s["median"] == 50.5 and s["p95"] == pytest.approx(95.05)
    assert s["min"] == 1 and s["max"] == 100 and latency_stats([]) == {"n": 0}
