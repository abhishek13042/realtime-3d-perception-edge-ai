"""Label-free tracking metrics on TUM (spec 04.2 R2, R5). No identity labels are used anywhere.

Definitions (fixed before any run, spec 04.2):
- Static tracks: confirmed tracks (>= min_track_len observations) whose majority class is not in
  `excluded_static_classes`. All world-frame metrics use static tracks only and `valid` centres only.
- Stability: per track, the 3D RMS distance of its valid world centres from their per-axis median (m). This is
  sqrt(3) * sigma for isotropic noise. Reported over tracks with >= `min_valid` valid centres.
- ID-switch proxy: a jump larger than d_jump between consecutive valid world centres of one track. Rate = jumps per
  100 valid track-frames (consecutive pairs).
- Fragmentation proxy: static confirmed tracks per inventory object (per class and overall, inventory classes only).
- Recovery: an interruption is either a gap inside a track (the same ID resumed: recovered) or a track end with at
  least buffer + 2 processed frames left in the sequence (the ID can no longer resume: a loss; Ultralytics keeps a
  lost track through buffer + 1 missed frames, measured in 04.1). Ends closer to the sequence end are censored and
  not counted. Recovery = recovered / (recovered + losses), over confirmed static tracks, in processed frames.
- Loss event (R5): the sequence index of a lost track's last observation.
- Coverage: share of tracked boxes with status `valid`; processed frames with no tracked box.
- Camera-motion trigger (R5): camera speed at loss events vs over all processed frames, from GT poses.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from perception.tracking.track_history import Observation

D_JUMP = 0.75
D_JUMP_SENSITIVITY = (0.5, 0.75, 1.0)
MIN_VALID = 5


def track_class(obs: list[Observation]) -> int:
    """Majority class (ties -> smallest id)."""
    c = Counter(o.cls for o in obs)
    return min(c, key=lambda k: (-c[k], k))


def static_tracks(tracks: dict[int, list[Observation]], min_len: int, excluded_ids: set[int]
                  ) -> dict[int, list[Observation]]:
    return {i: v for i, v in tracks.items() if len(v) >= min_len and track_class(v) not in excluded_ids}


def valid_world(obs: list[Observation]) -> tuple[np.ndarray, np.ndarray]:
    """(frame_index array, Nx3 world centres) of the valid observations with a world centre."""
    keep = [o for o in obs if o.status == "valid" and o.centre_world is not None]
    return (np.array([o.frame_index for o in keep], np.int64),
            np.array([o.centre_world for o in keep], np.float64).reshape(-1, 3))


def stability_rms(points: np.ndarray) -> float:
    p = np.asarray(points, np.float64).reshape(-1, 3)
    if len(p) == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.sum((p - np.median(p, axis=0)) ** 2, axis=1))))


@dataclass(frozen=True)
class Jump:
    track_id: int
    cls: int
    frame_a: int        # processed frame index of the centre before the jump
    frame_b: int        # ... and after
    seq_a: int
    seq_b: int
    distance_m: float


def jumps_in_track(track_id: int, obs: list[Observation], d_jump: float) -> tuple[list[Jump], int]:
    """Jumps > d_jump between consecutive valid world centres; also returns the number of consecutive pairs."""
    keep = [o for o in obs if o.status == "valid" and o.centre_world is not None]
    out = []
    for a, b in zip(keep, keep[1:]):
        d = float(np.linalg.norm(np.subtract(b.centre_world, a.centre_world)))
        if d > d_jump:
            out.append(Jump(track_id, a.cls, a.frame_index, b.frame_index, a.seq_index, b.seq_index, d))
    return out, max(len(keep) - 1, 0)


def interruptions(obs: list[Observation], n_processed: int, buffer: int) -> tuple[int, int, list[int]]:
    """(recovered gaps, losses, sequence index of the last observation of each loss) for one track."""
    fi = [o.frame_index for o in obs]
    recovered = sum(1 for a, b in zip(fi, fi[1:]) if b - a > 1)
    lost = (n_processed - 1 - fi[-1]) >= buffer + 2
    return recovered, int(lost), ([obs[-1].seq_index] if lost else [])


def fragmentation(static: dict[int, list[Observation]], inventory: dict[str, dict], names: list[str]) -> dict:
    per = Counter(names[track_class(v)] for v in static.values())
    rows = {c: {"tracks": int(per.get(c, 0)), "objects": int(v["n"]),
                "tracks_per_object": round(per.get(c, 0) / v["n"], 3)} for c, v in inventory.items()}
    t, n = sum(r["tracks"] for r in rows.values()), sum(r["objects"] for r in rows.values())
    lo = sum(per.get(c, 0) for c in inventory) / max(sum(v["range"][1] for v in inventory.values()), 1)
    hi = sum(per.get(c, 0) for c in inventory) / max(sum(v["range"][0] for v in inventory.values()), 1)
    return {"overall": round(t / n, 3) if n else math.nan, "overall_range": [round(lo, 3), round(hi, 3)],
            "tracks": t, "objects": n, "per_class": rows}


def evaluate(tracks: dict[int, list[Observation]], n_processed: int, frames_without_tracks: int, *,
             min_len: int, buffer: int, excluded_ids: set[int], inventory: dict[str, dict] | None,
             names: list[str], d_jumps: Iterable[float] = D_JUMP_SENSITIVITY, min_valid: int = MIN_VALID) -> dict:
    static = static_tracks(tracks, min_len, excluded_ids)
    stab = [stability_rms(valid_world(v)[1]) for v in static.values() if len(valid_world(v)[0]) >= min_valid]
    jumps: dict[str, dict] = {}
    for d in d_jumps:
        all_j, pairs = [], 0
        for i, v in static.items():
            j, p = jumps_in_track(i, v, d)
            all_j += j
            pairs += p
        jumps[f"{d:.2f}"] = {"n_jumps": len(all_j), "valid_pairs": pairs,
                             "per_100": round(100 * len(all_j) / pairs, 3) if pairs else math.nan}
    rec = loss = 0
    loss_frames: list[int] = []
    for v in static.values():
        r, l_, lf = interruptions(v, n_processed, buffer)
        rec, loss = rec + r, loss + l_
        loss_frames += lf
    all_obs = [o for v in tracks.values() for o in v]
    st = Counter(o.status for o in all_obs)
    res = {
        "n_tracks": len(tracks), "n_confirmed": sum(1 for v in tracks.values() if len(v) >= min_len),
        "n_static_confirmed": len(static),
        "stability_m": {"n_tracks": len(stab), "median": _r(np.median(stab)) if stab else None,
                        "p90": _r(np.percentile(stab, 90)) if stab else None},
        "stability_values_m": [_r(s) for s in stab],
        "id_switch_proxy": jumps,
        "recovery": {"recovered_gaps": rec, "losses": loss,
                     "rate": round(rec / (rec + loss), 3) if rec + loss else None},
        "loss_seq_indices": sorted(loss_frames),
        "coverage": {"tracked_boxes": len(all_obs), "valid_share": round(st["valid"] / len(all_obs), 4)
                     if all_obs else None, "status_counts": dict(st), "frames_without_tracks": frames_without_tracks,
                     "n_processed": n_processed},
        "mean_track_len": round(float(np.mean([len(v) for v in static.values()])), 2) if static else None,
    }
    if inventory is not None:
        res["fragmentation"] = fragmentation(static, inventory, names)
    return res


def largest_jumps(tracks: dict[int, list[Observation]], min_len: int, excluded_ids: set[int], d_jump: float,
                  n: int = 8) -> list[Jump]:
    static = static_tracks(tracks, min_len, excluded_ids)
    js = [j for i, v in static.items() for j in jumps_in_track(i, v, d_jump)[0]]
    return sorted(js, key=lambda j: (-j.distance_m, j.track_id, j.frame_a))[:n]


# ---- camera motion (R5) -------------------------------------------------------------------------------------------

def camera_speeds(T: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear (m/s) and angular (deg/s) camera speed per frame from Nx4x4 camera-to-world poses (central differences,
    one-sided at the ends)."""
    T, t = np.asarray(T, np.float64), np.asarray(t, np.float64)
    n = len(t)
    lin, ang = np.zeros(n), np.zeros(n)
    for i in range(n):
        a, b = max(i - 1, 0), min(i + 1, n - 1)
        dt = t[b] - t[a]
        if dt <= 0:
            continue
        lin[i] = np.linalg.norm(T[b, :3, 3] - T[a, :3, 3]) / dt
        R = T[a, :3, :3].T @ T[b, :3, :3]
        ang[i] = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))) / dt
    return lin, ang


def motion_trigger(speed_at_losses: np.ndarray, speed_all: np.ndarray, ratio: float = 1.5, min_events: int = 20,
                   p_max: float = 0.01) -> dict:
    """Fires if median(loss) > ratio * median(all), >= min_events losses, and one-sided Mann-Whitney p < p_max."""
    from scipy.stats import mannwhitneyu

    s_l, s_a = np.asarray(speed_at_losses, float), np.asarray(speed_all, float)
    n = len(s_l)
    med_l = float(np.median(s_l)) if n else math.nan
    med_a = float(np.median(s_a))
    p = float(mannwhitneyu(s_l, s_a, alternative="greater").pvalue) if n else math.nan
    r = med_l / med_a if med_a > 0 and n else math.nan
    fires = bool(n >= min_events and r > ratio and p < p_max)
    return {"n_losses": n, "median_at_losses": _r(med_l), "median_all": _r(med_a), "ratio": _r(r),
            "mannwhitney_p": p, "fires": fires,
            "rule": f"ratio > {ratio} and n >= {min_events} and p < {p_max}"}


def _r(x, nd=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


# ---- split guard (R4) ---------------------------------------------------------------------------------------------

TUNE_SEQUENCE = "rgbd_dataset_freiburg1_desk"
REPORT_SEQUENCE = "rgbd_dataset_freiburg3_long_office_household"


def split_guard(stage: str, sequence: str) -> None:
    """Tuning only ever touches fr1_desk; the report stage never tunes (it takes no grid) and reports fr3, or fr1 as a
    labelled tuning-set row."""
    if stage == "tune":
        if sequence != TUNE_SEQUENCE:
            raise PermissionError(f"tuning is allowed on {TUNE_SEQUENCE} only, not {sequence} (spec 04.2 R4)")
    elif stage == "report":
        if sequence not in (TUNE_SEQUENCE, REPORT_SEQUENCE):
            raise PermissionError(f"no reporting split defined for {sequence}")
    else:
        raise ValueError(f"unknown stage {stage!r}")
