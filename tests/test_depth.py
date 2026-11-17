import numpy as np

from perception.depth.decode import decode_sun, decode_tum, mask_range


def test_tum_decode_scale_and_zero():
    raw = np.array([[5000, 0, 10000]], dtype=np.uint16)
    d, v = decode_tum(raw)
    assert d.dtype == np.float32
    assert d[0, 0] == 1.0 and d[0, 2] == 2.0
    assert not v[0, 1] and d[0, 1] == 0.0


def _sun_encode(mm: int) -> int:
    # inverse of the decoder's 3-bit right rotation: rotate left by 3
    return ((mm << 3) | (mm >> 13)) & 0xFFFF


def test_sun_decode_bit_rotation():
    raw = np.array([[_sun_encode(1234), 0]], dtype=np.uint16)
    d, v = decode_sun(raw)
    assert np.isclose(d[0, 0], 1.234)
    assert v[0, 0] and not v[0, 1]


def test_out_of_range_invalid():
    raw = np.array([[100, 5000 * 20]], dtype=np.uint32).astype(np.uint16)  # 0.02 m, and wrapped garbage
    d, v = decode_tum(raw, min_m=0.1, max_m=10.0)
    assert not v[0, 0]
    assert np.all(d[~v] == 0.0)


def test_nan_and_inf_invalid():
    d, v = mask_range(np.array([[np.nan, np.inf, 1.5, -1.0]], dtype=np.float32), 0.1, 10.0)
    assert v.tolist() == [[False, False, True, False]]
    assert d[0, 2] == 1.5 and np.all(d[~v] == 0.0)
