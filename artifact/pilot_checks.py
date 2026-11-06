#!/usr/bin/env python3
"""Values for the protocol's kill criteria, from the pilot runs and the frozen frames. Writes experiments/pilot_checks.json."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = Path(os.environ.get("GROUND_TRUTH_DIR", str(ROOT.parents[1] / "runtime" / "ground_truth" / "icsc2027")))


def main() -> None:
    runs = [r for r in json.loads((ROOT / "experiments" / "registry.json").read_text(encoding="utf-8"))["runs"] if str(r.get("run_id", "")).startswith("pilot-") and r.get("status") == "ok"]
    by_method = {r["method"]: r for r in runs}
    values = {m: r["metric_values"] for m, r in by_method.items()}
    hit = {}
    for dataset in ("fever", "scifact"):
        valid = {r["id"] for r in map(json.loads, (GROUND_TRUTH / "frames" / f"{dataset}.rationales.jsonl").open(encoding="utf-8")) if r["valid"]}
        frames = [f for f in map(json.loads, (GROUND_TRUTH / "frames" / f"{dataset}.jsonl").open(encoding="utf-8")) if f["id"] in valid]
        hit[dataset] = sum(f["retrieval_hit"] for f in frames) / len(frames)
    notes_drop = values["no_context"]["notes_drop"]
    unparsed = max(v["parse_failure_rate"] for v in values.values())
    out = {"claims": {m: r.get("n") for m, r in by_method.items()}, "notes_drop_no_context": notes_drop, "kill_notes_floor": notes_drop < 0.05, "max_parse_failure_rate": unparsed,
           "kill_parse": unparsed > 0.05, "retrieval_hit_rate": hit, "kill_hit_rate": min(hit.values()) < 0.25,
           "notes_drop": {m: v["notes_drop"] for m, v in values.items()}, "peer_drop": {m: v["peer_drop"] for m, v in values.items()}, "accuracy_alone": {m: v["acc_alone"] for m, v in values.items()}}
    (ROOT / "experiments" / "pilot_checks.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
