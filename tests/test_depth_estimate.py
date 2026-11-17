"""Spec 03.1 tests: depth association."""

import math

import numpy as np
import pytest

from perception.depth.estimate import AssocParams, estimate_box, estimate_boxes, region_of

HW = (100, 120)
BOX = (20, 20, 80, 80)   # with shrink 0.5 -> region cols/rows 35..64 (30 x 30 = 900 px)


def plane(z=2.0):
    d = np.full(HW, z, np.float32)
    return d, np.ones(HW, bool)


def P(**kw):
    return AssocParams(**{"shrink": 0.5, "estimator": "B", "bin_m": 0.05, "m_frac": 0.15, "n_min": 20,
                          "tau_cov": 0.3, "tau_spread_m": 0.25, **kw})


def test_region_rule():
    assert region_of((10, 10, 20, 20), HW, 1.0) == (10, 10, 20, 20)       # cols/rows 10..19 inclusive
    assert region_of((10.2, 10.7, 19.5, 19.1), HW, 1.0) == (10, 10, 20, 20)
    assert region_of(BOX, HW, 0.5) == (35, 35, 65, 65)


@pytest.mark.parametrize("est", ["A", "B"])
def test_constant_plane(est):
    d, v = plane(2.0)
    r = estimate_box(d, v, BOX, P(estimator=est))
    assert r.z == pytest.approx(2.0) and r.spread == 0.0 and r.status == "valid" and r.n_valid == 900
    assert r.valid_frac == 1.0 and not r.fallback


def test_all_invalid_and_too_few():
    d, v = plane()
    v[:] = False
    r = estimate_box(np.where(v, d, 0).astype(np.float32), v, BOX, P())
    assert r.status == "no_depth" and math.isnan(r.z) and r.n_valid == 0
    v[35:37, 35:44] = True                                                  # 18 valid pixels < n_min 20
    r = estimate_box(d, v, BOX, P())
    assert r.status == "no_depth" and math.isnan(r.z) and r.n_valid == 18


def test_half_invalid_coverage():
    d, v = plane()
    v[:, ::2] = False                                                       # exactly 50% valid in the region
    assert estimate_box(d, v, BOX, P(tau_cov=0.3)).status == "valid"
    r = estimate_box(d, v, BOX, P(tau_cov=0.6))
    assert r.status == "low_coverage" and r.valid_frac == pytest.approx(0.5) and r.z == pytest.approx(2.0)


def _fg_bg(fg_frac, fg=1.0, bg=3.0):
    """Region with the top fg_frac of rows at fg depth (plus mild noise) and the rest at bg."""
    rng = np.random.default_rng(0)
    d = np.full(HW, bg, np.float32) + rng.normal(0, 0.005, HW).astype(np.float32)
    rows = int(round(30 * fg_frac))
    d[35:35 + rows, :] = fg + rng.normal(0, 0.005, (rows, HW[1])).astype(np.float32)
    return d, np.ones(HW, bool)


def test_ac2_foreground_majority():
    d, v = _fg_bg(0.6)
    b = estimate_box(d, v, BOX, P(estimator="B"))
    a = estimate_box(d, v, BOX, P(estimator="A"))
    assert abs(b.z - 1.0) <= 0.05
    assert abs(a.z - 1.0) <= 0.05            # majority foreground: the median also lands on the object
    # Documented limitation (spec 03.1 note): MAD has a 50% breakdown point, so a 60/40 mix has a small spread
    # and is NOT flagged by tau_spread. Only mixes with no majority depth inflate the spread.
    assert b.spread < 0.05 and b.status == "valid"
    d, v = _fg_bg(0.5)
    assert estimate_box(d, v, BOX, P()).status == "low_coverage"   # 50/50 mix: spread ~ 1 m, flagged


def test_background_majority_b_still_picks_near_mode():
    d, v = _fg_bg(0.3)                       # 30% object, 70% background
    b = estimate_box(d, v, BOX, P(estimator="B"))
    a = estimate_box(d, v, BOX, P(estimator="A"))
    assert abs(b.z - 1.0) <= 0.05
    assert abs(a.z - 3.0) <= 0.05            # documented failure of A: the median is the background
    assert b.status == "valid"               # spread describes the background majority (see spec 03.1 note)


def test_flying_pixels_ignored():
    d, v = plane(2.0)
    d[35:37, 35:45] = 0.4                    # 20 px = 2.2% of the region, below m_frac
    r = estimate_box(d, v, BOX, P(estimator="B"))
    assert r.z == pytest.approx(2.0) and not r.fallback


def test_no_significant_bin_falls_back():
    d, v = plane()
    d[35:65, 35:65] = np.linspace(1.0, 6.0, 900, dtype=np.float32).reshape(30, 30)  # ~1% per 5 cm bin
    r = estimate_box(d, v, BOX, P(estimator="B"))
    assert r.fallback and r.z == pytest.approx(float(np.median(d[35:65, 35:65])))


def test_clipping_and_outside():
    d, v = plane()
    r = estimate_box(d, v, (-50, -50, 40, 40), P(shrink=1.0))
    assert r.region == (0, 0, 40, 40) and r.n_valid == 1600 and r.status == "valid"
    r = estimate_box(d, v, (130, 10, 160, 40), P(shrink=1.0))
    assert r.status == "no_depth" and r.n_valid == 0 and r.valid_frac == 0.0


def test_estimate_boxes_and_bad_estimator():
    d, v = plane()
    out = estimate_boxes(d, v, np.array([BOX, (0, 0, 10, 10)]), P())
    assert [o.status for o in out] == ["valid", "valid"] and estimate_boxes(d, v, np.zeros((0, 4)), P()) == []
    with pytest.raises(ValueError):
        estimate_box(d, v, BOX, P(estimator="C"))


def test_from_config():
    from perception.config import load_config
    p = AssocParams.from_config(load_config("depth"), shrink=0.3)
    assert p.shrink == 0.3 and p.estimator in ("A", "B") and p.n_min >= 1
