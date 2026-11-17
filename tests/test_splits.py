import pytest

from perception.data.splits import carve_val, check_no_overlap


def _data():
    ids = [f"{i:06d}" for i in range(1, 101)]
    groups = {i: f"g{int(i) // 3}" for i in ids}  # groups of ~3 images
    return ids, groups


def test_carve_val_deterministic_and_disjoint():
    ids, groups = _data()
    a = carve_val(ids, groups, 0.2, seed=0)
    b = carve_val(ids, groups, 0.2, seed=0)
    assert a == b
    tr, va = a
    assert sorted(tr + va) == ids
    check_no_overlap({"train": tr, "val": va}, groups)
    assert carve_val(ids, groups, 0.2, seed=1) != a


def test_overlap_detected():
    ids, groups = _data()
    with pytest.raises(AssertionError):
        check_no_overlap({"train": ids[:10], "val": ids[9:20]}, groups)
    # different ids but same group
    with pytest.raises(AssertionError):
        check_no_overlap({"train": ["000003"], "val": ["000004"]}, groups)
