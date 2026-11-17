import numpy as np

from perception.data.quality import depth_histogram, depth_percentiles, differs_fraction, valid_fraction
from perception.data.sampling import sample_ids


def test_valid_fraction_exact():
    v = np.zeros((10, 10), bool)
    v[:3, :] = True
    assert valid_fraction(v) == 0.3


def test_percentiles_ignore_invalid():
    d = np.full((4, 4), 100.0, np.float32)  # junk where invalid
    v = np.zeros((4, 4), bool)
    d[0, :] = [1.0, 2.0, 3.0, 4.0]
    v[0, :] = True
    p5, p50, p95 = depth_percentiles(d, v)
    assert p50 == 2.5 and p95 < 4.01 and p5 > 0.99
    assert depth_percentiles(d, np.zeros_like(v)) is None


def test_histogram_counts_valid_only():
    d = np.array([[0.55, 0.55, 3.0, 12.0]], np.float32)
    v = np.array([[True, True, True, False]])
    h = depth_histogram(d, v)
    assert h.sum() == 3 and h[5] == 2 and h[30] == 1


def test_differs_fraction():
    assert differs_fraction(np.array([1, 2, 3, 4]), np.array([1, 0, 3, 0])) == 0.5


def test_sampler_deterministic():
    ids = [f"{i:06d}" for i in range(1000)]
    a = sample_ids(ids, 50, seed=0)
    assert a == sample_ids(list(reversed(ids)), 50, seed=0)
    assert len(a) == 50 == len(set(a)) and a == sorted(a)
    assert a != sample_ids(ids, 50, seed=1)
    assert sample_ids(ids[:10], 50, seed=0) == ids[:10]
