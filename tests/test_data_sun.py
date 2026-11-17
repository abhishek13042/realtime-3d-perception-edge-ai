from perception.data.sunrgbd import _rel


def test_remote_prefix_stripped():
    p = "/n/fs/sun3d/data/SUNRGBD/kv2/kinect2data/x/image/0000103.jpg"
    assert _rel(p) == "SUNRGBD/kv2/kinect2data/x/image/0000103.jpg"


def test_relative_path_kept():
    assert _rel("SUNRGBD/kv1/a/image/1.jpg") == "SUNRGBD/kv1/a/image/1.jpg"
