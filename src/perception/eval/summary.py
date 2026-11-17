"""Spec 05.4 R4: values for docs/results_summary.md, read only through the results index (05.1).

Every number in the summary comes from Sources.get(path, *keys). It refuses a file that is not in
benchmarks/results/index.json, and a file whose bytes changed since the index was written, so the summary can never
cite a value the index does not vouch for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from perception.eval.results_index import sha256_file


class SourceError(RuntimeError):
    pass


class Sources:
    def __init__(self, repo: Path, index: dict):
        self.repo = repo
        self.rows = {r["path"]: r for r in index["files"]}
        self.used: dict[str, str] = {}
        self._cache: dict[str, Any] = {}

    def doc(self, path: str) -> Any:
        if path not in self.rows:
            raise SourceError(f"{path} is not in the results index; run scripts/results_index.py first")
        if path not in self._cache:
            p = self.repo / path
            if not p.exists():
                raise SourceError(f"{path} is indexed but missing on disk")
            if sha256_file(p) != self.rows[path]["file_sha256"]:
                raise SourceError(f"{path} changed since the index was written; refresh the index")
            self._cache[path] = json.loads(p.read_text(encoding="utf-8"))
            self.used[path] = self.rows[path]["file_sha256"]
        return self._cache[path]

    def get(self, path: str, *keys) -> Any:
        v = self.doc(path)
        for k in keys:
            try:
                v = v[k]
            except (KeyError, IndexError, TypeError) as e:
                raise SourceError(f"{path}: no value at {'/'.join(map(str, keys))}") from e
        return v

    def has(self, path: str) -> bool:
        return path in self.rows and (self.repo / path).exists()


def fmt(x: Any, nd: int = 3, unit: str = "", pct: bool = False) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if pct:
        return f"{100 * float(x):.1f}%"
    if isinstance(x, (int, float)):
        return f"{float(x):.{nd}f}{unit}"
    return str(x)
