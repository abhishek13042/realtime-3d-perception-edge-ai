"""Spec 02.1: Detections contract, postprocessing, and a detector smoke/determinism test (skipped without weights)."""

import numpy as np
import pytest

from perception.config import REPO_ROOT, load_config
from perception.detection.base import Detections, Detector
from perception.detection.classes import COCO_NAMES
from perception.detection.letterbox import letterbox_params


class _Fixed(Detector):
    def __init__(self, raw, **kw):
        super().__init__(**kw)
        self.raw = raw

    def _infer(self, x):
        return self.raw


def test_empty_detections_shape():
    d = _Fixed(np.zeros((0, 6), np.float32), conf=0.5).detect(np.zeros((480, 640, 3), np.uint8))
    assert len(d) == 0 and d.boxes.shape == (0, 4) and d.boxes.dtype == np.float32


def test_postprocess_threshold_sort_maxdet_classes():
    raw = np.array([[0, 80, 10, 90, 0.3, 56], [0, 80, 20, 100, 0.9, 0], [0, 80, 30, 110, 0.6, 56],
                    [0, 80, 40, 120, 0.1, 56]], np.float32)
    det = _Fixed(raw, conf=0.2, max_det=2).detect(np.zeros((480, 640, 3), np.uint8))
    assert det.scores.tolist() == pytest.approx([0.9, 0.6])
    assert det.class_names == ["person", "chair"]
    assert det.boxes[0].tolist() == [0, 0, 20, 20]  # pad_y 80 removed
    only_chairs = _Fixed(raw, conf=0.2, classes=[56]).detect(np.zeros((480, 640, 3), np.uint8))
    assert set(only_chairs.class_ids.tolist()) == {56} and len(only_chairs) == 2


def test_rejects_non_uint8_rgb():
    with pytest.raises(ValueError):
        _Fixed(np.zeros((0, 6))).detect(np.zeros((480, 640), np.uint8))


def test_postprocess_uses_info():
    info = letterbox_params((530, 730), 640)
    det = _Fixed(np.zeros((0, 6))).postprocess(np.array([[0, 0, 640, 640, 1.0, 0]]), info)
    assert det.boxes[0].tolist() == pytest.approx([0, 0, 730, 530])  # clipped to the image


# --- real model (skipped if the weights have not been downloaded) -------------------------------------------
CFG = load_config("detector")
needs_weights = pytest.mark.skipif(not (REPO_ROOT / CFG["weights"]).exists(), reason="weights not downloaded")


@pytest.fixture(scope="module")
def detector():
    pytest.importorskip("torch")
    from perception.detection.torch_backend import build_detector

    return build_detector(CFG, conf=0.05)


def _test_image():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    img[100:400, 150:500] = (200, 40, 40)
    return img


@needs_weights
def test_class_names_match_weights(detector):
    assert [detector.names[i] for i in range(80)] == list(COCO_NAMES)


@needs_weights
def test_smoke_types(detector):
    d = detector.detect(_test_image())
    assert d.boxes.dtype == np.float32 and d.boxes.shape == (len(d), 4) and len(d.class_ids) == len(d)
    assert ((d.class_ids >= 0) & (d.class_ids < 80)).all()
    assert set(d.timings_ms) == {"preprocess", "infer", "postprocess"}


@needs_weights
def test_deterministic(detector):
    a, b = detector.detect(_test_image()), detector.detect(_test_image())
    assert np.array_equal(a.boxes, b.boxes) and np.array_equal(a.scores, b.scores)
    assert np.array_equal(a.class_ids, b.class_ids)
