"""Deterministic frame sampling for development and benchmarks (spec 01.4, R4)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from perception.config import REPO_ROOT
from perception.data.splits import read_split

SPLIT_DIR = REPO_ROOT / "configs" / "splits"


def sample_ids(split: str | Sequence[str], n: int, seed: int = 0) -> list[str]:
    """Return n ids drawn without replacement, in sorted order, from a split.

    `split` is either a list of ids or a split name such as "sunrgbd_test" (configs/splits/<name>.txt).
    The ids are sorted before sampling, so the result depends only on (set of ids, n, seed).
    If n ≥ the split size, all ids are returned.
    """
    ids = read_split(SPLIT_DIR / f"{split}.txt") if isinstance(split, str) else list(split)
    ids = sorted(set(ids))
    if n >= len(ids):
        return ids
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ids), size=n, replace=False)
    return sorted(ids[i] for i in pick)


def write_ids(path: str | Path, ids: Sequence[str], header: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {h}" for h in header.splitlines()] + list(ids)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_ids(path: str | Path) -> list[str]:
    return [s.strip() for s in Path(path).read_text(encoding="utf-8").splitlines()
            if s.strip() and not s.startswith("#")]
