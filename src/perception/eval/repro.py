"""Spec 05.4 R3: reproducibility comparators. Tolerances are fixed here, before any re-run.

Each check returns a plain dict with pass/fail, so repro_check.json reads the same for every item.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

TOLERANCES = {
    "sun_test_200": {"kind": "abs", "tol": 1e-9, "unit": "m", "what": "per-object 3D centre error, 05.2"},
    "fr3_s1_metrics": {"kind": "exact", "tol": 0.0, "unit": None, "what": "04.2 fr3 stride-1 metrics"},
    "detection_map": {"kind": "abs", "tol": 0.001, "unit": "mAP", "what": "02.3 SUN test mAP@[.5:.95]"},
    "pipeline_median": {"kind": "rel", "tol": 0.15, "unit": "ms", "what": "05.3 pipeline median per list"},
}


def within(ref: float, new: float, kind: str, tol: float) -> bool:
    """abs: |new - ref| <= tol; rel: |new - ref| <= tol * |ref|; exact: new == ref. NaN equals NaN."""
    if isinstance(ref, float) and isinstance(new, float) and math.isnan(ref) and math.isnan(new):
        return True
    if kind == "exact":
        return ref == new
    d = abs(float(new) - float(ref))
    if kind == "abs":
        return d <= tol
    if kind == "rel":
        return d <= tol * abs(float(ref))
    raise ValueError(f"unknown tolerance kind {kind!r}")


def compare_scalar(name: str, ref: float, new: float) -> dict:
    t = TOLERANCES[name]
    d = float(new) - float(ref)
    return {"item": name, "what": t["what"], "tolerance": {k: t[k] for k in ("kind", "tol", "unit")},
            "reported": ref, "rerun": new, "diff": d,
            "rel_diff": d / ref if ref else None, "pass": within(ref, new, t["kind"], t["tol"])}


def compare_arrays(name: str, ref: np.ndarray, new: np.ndarray) -> dict:
    """Element-wise abs tolerance; NaN positions must match exactly (status changes show up as NaN mismatches)."""
    t = TOLERANCES[name]
    ref, new = np.asarray(ref, np.float64), np.asarray(new, np.float64)
    out = {"item": name, "what": t["what"], "tolerance": {k: t[k] for k in ("kind", "tol", "unit")},
           "n_reported": int(ref.size), "n_rerun": int(new.size)}
    if ref.shape != new.shape:
        return {**out, "pass": False, "reason": f"shape {ref.shape} vs {new.shape}"}
    nan_ref, nan_new = np.isnan(ref), np.isnan(new)
    fin = ~nan_ref & ~nan_new
    max_d = float(np.abs(ref[fin] - new[fin]).max(initial=0.0))
    nan_mismatch = int((nan_ref != nan_new).sum())
    ok = nan_mismatch == 0 and within(0.0, max_d, t["kind"], t["tol"])
    return {**out, "max_abs_diff": max_d, "nan_mismatch": nan_mismatch, "pass": bool(ok)}


def _canon(x: Any) -> Any:
    return json.loads(json.dumps(x, sort_keys=True))


def compare_exact(name: str, ref: Any, new: Any) -> dict:
    """Exact equality after a JSON round trip (the reported values are stored as JSON)."""
    t = TOLERANCES[name]
    a, b = _canon(ref), _canon(new)
    diffs: list[str] = []

    def walk(x, y, path):
        if len(diffs) >= 20:
            return
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y:
                    diffs.append(f"{path}/{k}: missing on one side")
                else:
                    walk(x[k], y[k], f"{path}/{k}")
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]")
        elif x != y:
            diffs.append(f"{path}: {str(x)[:60]} != {str(y)[:60]}")

    walk(a, b, "")
    return {"item": name, "what": t["what"], "tolerance": {k: t[k] for k in ("kind", "tol", "unit")},
            "differences": diffs, "pass": not diffs}
