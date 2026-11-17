"""SUN RGB-D splits (spec 01.3, R4–R5).

- train/test: the official split (SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat: alltrain / alltest).
- val: carved from the official train by scene group with a fixed seed.

Scene group: in SUN RGB-D every `sequenceName` folder holds exactly one image, so the folder is the finest
grouping the metadata offers. Grouping by it prevents the same image entering two splits; it cannot detect
two different images of the same room (a known limitation, documented in docs/data_layout.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

from perception.data.sunrgbd import _rel


def load_official(allsplit_path: str | Path) -> tuple[set[str], set[str]]:
    """Return (train_sequences, test_sequences) with the remote prefix stripped."""
    d = sio.loadmat(str(allsplit_path), squeeze_me=True, struct_as_record=False)
    norm = lambda a: {_rel(str(s)).rstrip("/") for s in np.atleast_1d(a)}  # noqa: E731
    return norm(d["alltrain"]), norm(d["alltest"])


def carve_val(train_ids: list[str], groups: dict[str, str], frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Split train ids into (train, val) by group. Deterministic for a given seed and input set."""
    uniq = sorted({groups[i] for i in train_ids})
    rng = np.random.default_rng(seed)
    n_val = int(round(frac * len(uniq)))
    val_groups = set(rng.permutation(uniq)[:n_val].tolist())
    tr = sorted(i for i in train_ids if groups[i] not in val_groups)
    va = sorted(i for i in train_ids if groups[i] in val_groups)
    return tr, va


def check_no_overlap(splits: dict[str, list[str]], groups: dict[str, str]) -> None:
    """Raise if any image id or any scene group appears in more than one split."""
    names = list(splits)
    for a_i, a in enumerate(names):
        for b in names[a_i + 1:]:
            ids = set(splits[a]) & set(splits[b])
            if ids:
                raise AssertionError(f"{len(ids)} image ids in both {a} and {b}")
            ga = {groups[i] for i in splits[a]}
            gb = {groups[i] for i in splits[b]}
            if ga & gb:
                raise AssertionError(f"{len(ga & gb)} scene groups in both {a} and {b}")


def write_split(path: str | Path, ids: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def read_split(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
