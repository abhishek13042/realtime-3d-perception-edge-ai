"""Spec 05.2 helpers: image-level bootstrap, the one-shot guard and the G4 status check.

Linkage, matching and error statistics come from `perception.eval.localization` (the 03.3 code, unchanged).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STATUSES = ("valid", "low_coverage", "no_depth")


def bootstrap_ci(values, groups, stats=("median", "p90"), n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI, resampling whole groups (images) with replacement.

    Objects from one image are correlated (same depth map, same sensor), so images are the resampling unit. Only
    images that contribute at least one value are resampled. Returns {stat: {"point", "lo", "hi"}} in metres.
    """
    v = np.asarray(values, np.float64)
    g = np.asarray(groups)
    if v.size == 0:
        return {s: {"point": None, "lo": None, "hi": None} for s in stats}
    _, inv = np.unique(g, return_inverse=True)
    G = int(inv.max()) + 1
    fns = {"median": np.median, "p90": lambda x: np.percentile(x, 90), "mean": np.mean}
    rng = np.random.default_rng(seed)
    draws = {s: np.empty(n_boot) for s in stats}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, G, G), minlength=G)[inv]  # times each object's image was drawn
        x = np.repeat(v, w)
        for s in stats:
            draws[s][b] = fns[s](x)
    q = (100 * alpha / 2, 100 * (1 - alpha / 2))
    return {s: {"point": round(float(fns[s](v)), 4), "lo": round(float(np.percentile(draws[s], q[0])), 4),
                "hi": round(float(np.percentile(draws[s], q[1])), 4)} for s in stats}


def bootstrap_ratio(values, num_mask, den_mask, groups, n_boot: int = 1000, seed: int = 0,
                    alpha: float = 0.05) -> dict:
    """Image-level bootstrap CI of median(values[num_mask]) / median(values[den_mask]) (the G4 low/valid ratio)."""
    v = np.asarray(values, np.float64)
    nm, dm = np.asarray(num_mask, bool), np.asarray(den_mask, bool)
    if not nm.any() or not dm.any():
        return {"point": None, "lo": None, "hi": None}
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    G = int(inv.max()) + 1
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, G, G), minlength=G)[inv]
        a, b = np.repeat(v[nm], w[nm]), np.repeat(v[dm], w[dm])
        if a.size and b.size:
            draws.append(np.median(a) / np.median(b))
    q = (100 * alpha / 2, 100 * (1 - alpha / 2))
    return {"point": round(float(np.median(v[nm]) / np.median(v[dm])), 3),
            "lo": round(float(np.percentile(draws, q[0])), 3), "hi": round(float(np.percentile(draws, q[1])), 3),
            "n_valid_draws": len(draws)}


def check_status_complete(statuses) -> dict:
    """R4 / G4: every output box carries one of the three statuses. Raises on any gap."""
    st = list(statuses)
    bad = [s for s in st if s not in STATUSES]
    if bad:
        raise AssertionError(f"G4 failed: {len(bad)} of {len(st)} output boxes lack a depth_status "
                             f"(e.g. {bad[:3]!r})")
    n = max(1, len(st))
    return {"n_boxes": len(st), "with_status": len(st), "coverage": 1.0 if st else math.nan,
            **{s: round(sum(x == s for x in st) / n, 4) for s in STATUSES}}


def one_shot_guard(out: Path, rerun_reason: str | None) -> dict | None:
    """R6: the headline file is written once. Returns the re-run record to store, or None for a first run.

    Raises PermissionError when the file exists and no reason is given.
    """
    if not Path(out).exists():
        return None
    if not rerun_reason or not rerun_reason.strip():
        raise PermissionError(f"{out} already exists; the SUN test evaluation is one-shot. Re-running needs "
                              "--rerun-reason (logged in the file and in PROGRESS.md).")
    prev = json.loads(Path(out).read_text(encoding="utf-8"))
    return {"reason": rerun_reason.strip(), "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "previous_created_utc": prev.get("created_utc"), "previous_reruns": prev.get("reruns", [])}


def log_rerun(progress: Path, out_rel: str, record: dict) -> None:
    """Append the re-run to PROGRESS.md under 'Phase 05 re-runs' (section created if missing)."""
    text = Path(progress).read_text(encoding="utf-8")
    head = "## Phase 05 re-runs"
    line = f"- {record['utc']}: `{out_rel}` re-run. Reason: {record['reason']}"
    if head not in text:
        text = text.rstrip("\n") + f"\n\n{head}\n{line}\n"
    else:
        i = text.index(head) + len(head)
        text = text[:i] + f"\n{line}" + text[i:]
    Path(progress).write_text(text, encoding="utf-8")
