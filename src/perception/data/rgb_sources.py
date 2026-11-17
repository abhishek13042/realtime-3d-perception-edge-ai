"""RGB-only frame lists for detection runs (spec 02.2).

Detection needs no depth. Loading RGB alone keeps the runs fast, and the image decode time can be measured
separately from detector latency. Each item is `RGBItem`, and the caller reads `path` with cv2.imread.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from perception.config import REPO_ROOT, data_root, load_config
from perception.data.sampling import read_ids
from perception.data.splits import read_split


@dataclass(frozen=True)
class RGBItem:
    frame_id: str
    dataset: str
    scene_id: str
    sensor: str | None
    path: Path


def _resolve_list(name_or_path: str) -> list[str]:
    """A SUN split name (sunrgbd_test), or a list file path (configs/benchmark_frames/...)."""
    split = REPO_ROOT / "configs" / "splits" / f"{name_or_path}.txt"
    if split.exists():
        return read_split(split)
    p = Path(name_or_path)
    p = p if p.is_absolute() else REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"no split or list named {name_or_path!r}")
    return read_ids(p)


def sun_items(list_name: str) -> list[RGBItem]:
    from perception.data.sunrgbd_annotations import load_annotations

    root = data_root(load_config("dataset")) / "sunrgbd"
    anns = {a.image_id: a for a in load_annotations(root)}
    return [RGBItem(i, "sunrgbd", anns[i].sequence, anns[i].sensor, root / anns[i].rgb_rel)
            for i in _resolve_list(list_name)]


def tum_items(sequence: str, list_name: str | None = None) -> list[RGBItem]:
    from perception.data.tum import TUMSequence

    cfg = load_config("dataset")
    seq = TUMSequence.from_config(sequence, cfg)
    rig = next((f"fr{k}" for k in "123" if f"freiburg{k}" in seq.name), None)  # TUM rig id, not a sensor claim
    items = [RGBItem(f"{t:.6f}", "tum", seq.name, rig, seq.dir / rgb_rel) for t, rgb_rel, _ in seq.items]
    if list_name is not None:
        want = set(_resolve_list(list_name))
        items = [it for it in items if it.frame_id in want]
        if len(items) != len(want):
            raise ValueError(f"{len(want) - len(items)} stamps in {list_name} not found in {sequence}")
    return items
