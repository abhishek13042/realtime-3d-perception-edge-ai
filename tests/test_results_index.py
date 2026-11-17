"""Spec 05.1 tests: results index rows and the freeze checks."""

import json
from pathlib import Path

import pytest

from perception.eval.results_index import (check_frozen, check_protocol, frozen_inputs, index_row, protocol_sha,
                                           sha256_file)

PROTO = """# Protocol
## 8. Other
## 9. Frozen inputs

| File | sha256 |
|---|---|
| `configs/a.yaml` | `{sha}` |
"""


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "configs" / "a.yaml").write_text("x: 1\n")
    sha = sha256_file(tmp_path / "configs" / "a.yaml")
    (tmp_path / "docs" / "evaluation_protocol.md").write_text(PROTO.format(sha=sha))
    return tmp_path


def test_complete_row(tmp_path):
    repo = _repo(tmp_path)
    p = repo / "benchmarks" / "r.json"
    p.write_text(json.dumps({"spec": "x", "header": {"git_commit": "abc", "git_dirty": False,
                                                    "config_sha256": {"depth": "f" * 64},
                                                    "created_utc": "2026-11-11T00:00:00+00:00"}}))
    row = index_row(p, repo)
    assert row["missing"] == []
    assert row["git_commit"] == "abc"
    assert row["sha256"] == {"header.config_sha256.depth": "f" * 64}


def test_missing_commit_flagged(tmp_path):
    repo = _repo(tmp_path)
    p = repo / "benchmarks" / "r.json"
    p.write_text(json.dumps({"config_sha256": "f" * 64, "created_utc": "t"}))
    assert index_row(p, repo)["missing"] == ["git_commit"]


def test_null_commit_is_present_not_missing(tmp_path):
    repo = _repo(tmp_path)
    p = repo / "benchmarks" / "r.json"
    p.write_text(json.dumps({"git_commit": None, "config_sha256": "f" * 64, "created_utc": "t"}))
    assert index_row(p, repo)["missing"] == []


def test_frozen_check(tmp_path):
    repo = _repo(tmp_path)
    assert list(check_frozen(repo)) == ["configs/a.yaml"]
    (repo / "configs" / "a.yaml").write_text("x: 2\n")
    with pytest.raises(PermissionError, match="configs/a.yaml"):
        check_frozen(repo)


def test_protocol_sha_check_fails_on_change(tmp_path):
    repo = _repo(tmp_path)
    idx = {"protocol": {"sha256": protocol_sha(repo)}}
    check_protocol(repo, idx)
    p = repo / "docs" / "evaluation_protocol.md"
    p.write_text(p.read_text() + "\nedit\n")
    with pytest.raises(PermissionError, match="protocol changed"):
        check_protocol(repo, idx)


def test_real_protocol_has_frozen_table():
    repo = Path(__file__).resolve().parents[1]
    table = frozen_inputs((repo / "docs" / "evaluation_protocol.md").read_text(encoding="utf-8"))
    assert {"configs/depth.yaml", "configs/detector.yaml", "configs/splits/sunrgbd_test.txt"} <= set(table)
