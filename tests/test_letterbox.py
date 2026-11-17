"""Spec 02.1 AC1: letterbox round trip <= 1 px, geometry, clipping."""

import numpy as np
import pytest

from perception.detection.letterbox import letterbox, letterbox_boxes, letterbox_params, unletterbox_boxes

# (H, W): the 4 SUN sensors, TUM, and awkward sizes
SIZES = [(427, 561), (530, 730), (531, 681), (441, 591), (480, 640), (1, 1), (37, 1001), (999, 13)]


def _random_boxes(h, w, n, rng):
    x = rng.uniform(0, w, (n, 2))
    y = rng.uniform(0, h, (n, 2))
    return np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], 1)


@pytest.mark.parametrize("hw", SIZES)
@pytest.mark.parametrize("size", [320, 480, 640])
def test_round_trip_within_1px(hw, size):
    rng = np.random.default_rng(0)
    boxes = _random_boxes(*hw, 200, rng)
    info = letterbox_params(hw, size)
    back = unletterbox_boxes(letterbox_boxes(boxes, info), info)
    assert np.abs(back - boxes).max() <= 1.0
    assert np.abs(back - boxes).max() < 1e-9  # the inverse is exact in float64


def test_round_trip_through_integer_pixels_within_1px():
    """A detector outputs positions on the letterbox grid; rounding there costs at most 0.5/gain px."""
    rng = np.random.default_rng(1)
    for hw in SIZES[:5]:
        info = letterbox_params(hw, 640)
        boxes = _random_boxes(*hw, 200, rng)
        back = unletterbox_boxes(np.round(letterbox_boxes(boxes, info)), info)
        assert np.abs(back - boxes).max() <= 1.0


def test_geometry_640x480():
    img = np.zeros((480, 640, 3), np.uint8)
    canvas, info = letterbox(img, 640)
    assert canvas.shape == (640, 640, 3)
    assert (info.gain_x, info.gain_y, info.pad_x, info.pad_y) == (1.0, 1.0, 0, 80)
    assert (canvas[:80] == 114).all() and (canvas[560:] == 114).all() and (canvas[80:560] == 0).all()


def test_image_content_placed_where_boxes_say():
    img = np.zeros((530, 730, 3), np.uint8)
    img[100:200, 300:400] = 255
    canvas, info = letterbox(img, 640)
    b = letterbox_boxes(np.array([[300, 100, 400, 200]]), info)[0]
    inner = canvas[int(b[1]) + 2:int(b[3]) - 2, int(b[0]) + 2:int(b[2]) - 2]
    assert inner.size and (inner == 255).all()


def test_unletterbox_clips():
    info = letterbox_params((480, 640), 640)
    out = unletterbox_boxes(np.array([[-10, 0, 700, 700]]), info)
    assert out.tolist() == [[0, 0, 640, 480]]


def test_empty_boxes_shape():
    info = letterbox_params((480, 640), 640)
    assert unletterbox_boxes(np.zeros((0, 4)), info).shape == (0, 4)


def test_matches_ultralytics_letterbox():
    pytest.importorskip("ultralytics")
    from ultralytics.data.augment import LetterBox

    rng = np.random.default_rng(2)
    for hw in SIZES[:5]:
        img = rng.integers(0, 255, (*hw, 3), dtype=np.uint8)
        ours, _ = letterbox(img, 640)
        theirs = LetterBox((640, 640), auto=False)(image=img)
        assert ours.shape == theirs.shape and np.array_equal(ours, theirs)
