import io
import tarfile
import zipfile

import pytest

from perception.config import load_config
from perception.data.download import parse_manifest, safe_extract


def _tar_with(tmp_path, member_name):
    archive = tmp_path / "evil.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        data = b"x"
        info = tarfile.TarInfo(member_name)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return archive


def test_safe_extract_rejects_tar_traversal(tmp_path):
    archive = _tar_with(tmp_path, "../evil.txt")
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_rejects_zip_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../evil.txt", "x")
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "out")


def test_safe_extract_accepts_normal_tar(tmp_path):
    archive = _tar_with(tmp_path, "seq/rgb.txt")
    safe_extract(archive, tmp_path / "out")
    assert (tmp_path / "out" / "seq" / "rgb.txt").read_bytes() == b"x"


def test_manifest_keys_unique_and_complete():
    entries = parse_manifest(load_config("dataset"))
    keys = [e.key for e in entries]
    assert len(keys) == len(set(keys))
    assert all(e.url.startswith("https://") for e in entries)


def test_dry_run_makes_no_files(tmp_path, monkeypatch, capsys):
    import scripts_download  # noqa: F401  (imported via conftest path)

    cfg = load_config("dataset")
    cfg["data_root"] = str(tmp_path / "root")
    monkeypatch.setattr(scripts_download, "load_config", lambda name: cfg)
    assert scripts_download.main(["--dry-run"]) == 0
    assert not (tmp_path / "root").exists()
    assert "TOTAL" in capsys.readouterr().out
