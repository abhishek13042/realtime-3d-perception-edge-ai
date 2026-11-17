"""Results index and freeze checks (spec 05.1, evaluation protocol sections 7 and 9).

- `index_rows` walks `benchmarks/**/*.json` and records the provenance fields of each file (git commit, dirty flag,
  config / input sha256, timestamp). Files missing a protocol section 7 field are flagged, not silently passed.
- `frozen_inputs` reads the protocol's section 9 table (file -> sha256). `check_frozen` refuses to continue when a
  frozen file changed. `protocol_sha` / `check_protocol` tie a Phase 05 result to the frozen protocol text.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROTOCOL = Path("docs") / "evaluation_protocol.md"
TIME_KEYS = ("created_utc", "timestamp", "timestamp_utc", "created")
SPLIT_KEYS = ("split", "source", "sequence", "dataset")
# Files whose numbers are headline results (protocol: test split / reporting sequence only).
HEADLINE = {
    "benchmarks/detection/eval_baseline.json",
    "benchmarks/tracking/eval_fr3_report.json",
    "benchmarks/results/localization_sun_test.json",
    "benchmarks/results/pipeline_profile.json",
    "benchmarks/results/repro_check.json",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _walk(obj: Any, depth: int = 0, prefix: str = ""):
    """Yield (dotted_key, value) for dict entries up to depth 3."""
    if isinstance(obj, dict) and depth <= 3:
        for k, v in obj.items():
            key = f"{prefix}{k}"
            yield key, v
            yield from _walk(v, depth + 1, key + ".")


def provenance_fields(doc: Any) -> dict:
    found: dict[str, Any] = {"git_commit_present": False, "git_commit": None, "git_dirty": None,
                             "sha256": {}, "timestamp": None, "split": None, "spec": None, "label": None}
    for key, v in _walk(doc):
        leaf = key.rsplit(".", 1)[-1]
        if leaf == "git_commit" and not found["git_commit_present"]:
            found["git_commit_present"], found["git_commit"] = True, v
        elif leaf == "git_dirty" and found["git_dirty"] is None:
            found["git_dirty"] = v
        elif "sha256" in leaf:
            if isinstance(v, str):
                found["sha256"][key] = v
            elif isinstance(v, dict):
                found["sha256"].update({f"{key}.{k}": x for k, x in v.items() if isinstance(x, str)})
        elif leaf in TIME_KEYS and found["timestamp"] is None and isinstance(v, str):
            found["timestamp"] = v
        elif leaf in SPLIT_KEYS and found["split"] is None and isinstance(v, str):
            found["split"] = v
        elif leaf == "spec" and found["spec"] is None:
            found["spec"] = v
        elif leaf == "label" and found["label"] is None and isinstance(v, str):
            found["label"] = v
    return found


def index_row(path: Path, repo: Path) -> dict:
    rel = path.relative_to(repo).as_posix()
    row: dict[str, Any] = {"path": rel, "headline": rel in HEADLINE,
                           "file_sha256": sha256_file(path),
                           "file_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                           .isoformat(timespec="seconds")}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return {**row, "error": f"unreadable: {e}", "missing": ["all"]}
    f = provenance_fields(doc)
    missing = []
    if not f["git_commit_present"]:
        missing.append("git_commit")
    if not f["sha256"]:
        missing.append("config_or_input_sha256")
    if f["timestamp"] is None:
        missing.append("timestamp")
    row.update({k: f[k] for k in ("spec", "label", "git_commit", "git_dirty", "split", "timestamp")})
    row["sha256"] = f["sha256"]
    row["missing"] = missing
    return row


def index_rows(repo: Path) -> list[dict]:
    files = sorted(p for p in (repo / "benchmarks").rglob("*.json") if p.name != "index.json")
    return [index_row(p, repo) for p in files]


# ---- freeze checks ------------------------------------------------------------------------------------------------

def frozen_inputs(protocol_text: str) -> dict[str, str]:
    """Section 9 table of the protocol: `| `path` | `sha256` |` rows."""
    sec = protocol_text.split("## 9.", 1)
    if len(sec) < 2:
        raise ValueError("protocol has no section 9 (frozen inputs)")
    return dict(re.findall(r"\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|", sec[1]))


def check_frozen(repo: Path, names: list[str] | None = None) -> dict[str, str]:
    """Raise if any frozen input (or only `names`) differs from the protocol's section 9. Returns the checked map."""
    table = frozen_inputs((repo / PROTOCOL).read_text(encoding="utf-8"))
    sel = {k: v for k, v in table.items() if names is None or k in names}
    if names is not None and set(names) - set(sel):
        raise KeyError(f"not in the protocol's frozen table: {sorted(set(names) - set(sel))}")
    bad = {k: (v, sha256_file(repo / k)) for k, v in sel.items() if sha256_file(repo / k) != v}
    if bad:
        raise PermissionError("frozen inputs changed since the protocol freeze: "
                              + "; ".join(f"{k} expected {e[:12]} got {g[:12]}" for k, (e, g) in bad.items()))
    return sel


def git_state(repo: Path) -> dict:
    """{"git_commit": sha or None (no commits yet), "git_dirty": bool or None (git unavailable)}."""
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=repo, capture_output=True, text=True,
                           timeout=10)
        commit = r.stdout.strip() if r.returncode == 0 else None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True,
                                    timeout=30).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty}


def protocol_sha(repo: Path) -> str:
    return sha256_file(repo / PROTOCOL)


def check_protocol(repo: Path, index: dict) -> str:
    """Raise if the protocol text differs from the sha frozen in the index."""
    want = index.get("protocol", {}).get("sha256")
    got = protocol_sha(repo)
    if want is None:
        raise ValueError("index has no frozen protocol sha (run scripts/results_index.py --freeze)")
    if got != want:
        raise PermissionError(f"evaluation protocol changed since the freeze ({want[:12]} -> {got[:12]}); log the "
                              "change in PROGRESS.md and re-run the affected results")
    return got
