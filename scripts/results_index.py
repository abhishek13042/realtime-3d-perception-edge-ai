"""Write benchmarks/results/index.json: every JSON result file with its provenance (spec 05.1 R3, R4).

    python scripts/results_index.py            # refresh the index, keep the frozen protocol sha
    python scripts/results_index.py --freeze   # (once) record the protocol sha as frozen

Refreshing never changes the frozen sha; it only reports whether the protocol still matches it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from perception.eval.results_index import (PROTOCOL, check_frozen, index_rows, protocol_sha)  # noqa: E402

OUT = REPO / "benchmarks" / "results" / "index.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", action="store_true", help="record the current protocol sha as the frozen one")
    a = ap.parse_args(argv)
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    proto = old.get("protocol", {})
    cur = protocol_sha(REPO)
    if a.freeze:
        if proto.get("sha256") and proto["sha256"] != cur:
            print("refusing: a different protocol sha is already frozen; log the change in PROGRESS.md and edit "
                  "the index by hand with the reason", file=sys.stderr)
            return 2
        proto = {"path": PROTOCOL.as_posix(), "sha256": cur,
                 "frozen_utc": proto.get("frozen_utc") or datetime.now(timezone.utc).isoformat(timespec="seconds")}
    frozen = check_frozen(REPO)  # raises if a frozen config / split changed
    rows = index_rows(REPO)
    idx = {
        "spec": "05.1 R3",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {**proto, "current_sha256": cur, "matches_frozen": proto.get("sha256") == cur},
        "frozen_inputs": frozen,
        "n_files": len(rows),
        "n_incomplete": sum(1 for r in rows if r["missing"]),
        "files": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(idx, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}: {len(rows)} files, {idx['n_incomplete']} with missing provenance fields; "
          f"protocol frozen={bool(proto.get('sha256'))} matches={idx['protocol']['matches_frozen']}")
    for r in rows:
        if r["missing"]:
            print(f"  incomplete: {r['path']}: missing {', '.join(r['missing'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
