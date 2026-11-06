#!/usr/bin/env python3
"""Pre-freeze feasibility run on FEVER claims disjoint from the study frames (frames/feasibility.jsonl), seed 1. Measures the
drop under false notes and false peers and the paired variance of contrasts C1, C2 and C3 per channel so the frozen sample size can be chosen before any study
claim is run. Its claims and answers are never part of the reported results. Writes experiments/feasibility/result.json."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

from run_cell import CACHE_ROOT, FLEET, GROUND_TRUTH, Client, arms_for, run_scripted

METHODS = ["no_context", "retrieved", "distractor_retrieved", "gold", "distractor_gold", "instruction"]
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    models = sys.argv[1:] or FLEET
    frames = [json.loads(l) for l in (GROUND_TRUTH / "frames" / "feasibility.jsonl").open(encoding="utf-8")]
    rationales = {r["id"]: r["rationales"] for r in map(json.loads, (GROUND_TRUTH / "frames" / "feasibility.rationales.jsonl").open(encoding="utf-8")) if r["valid"]}
    frames = [f for f in frames if f["id"] in rationales]
    table = {}
    rows = {}
    for model in models:
        client = Client(model, CACHE_ROOT / "feasibility" / f"{hashlib.sha256(model.encode()).hexdigest()[:12]}.jsonl")
        for method in METHODS:
            out = run_scripted(frames, method, model, 1, rationales, client)
            for r in out:
                rows[(model, method, r["arm"], r["item"])] = r["correct"]
            parse = sum(not r["parsed"] for r in out) / len(out)
            cell = {arm: sum(r["correct"] for r in out if r["arm"] == arm) / len(frames) for arm in arms_for(method)}
            table.setdefault(model, {})[method] = {**cell, "drop": cell["alone"] - cell["false"], "notes_drop": cell["alone"] - cell["false_text"], "gain": cell["true"] - cell["alone"], "parse_failure_rate": parse, "mean_latency_s": statistics.mean(r["latency_s"] or 0 for r in out)}
            print(model, method, {k: round(v, 3) for k, v in table[model][method].items()}, flush=True)

    def did(item: str, ctx: str, ms: list[str], arm: str) -> float:
        def mean(method: str, a: str) -> float:
            return sum(rows[(m, method, a, item)] for m in ms) / len(ms)
        return (mean("no_context", "alone") - mean("no_context", arm)) - (mean(ctx, "alone") - mean(ctx, arm))

    items = [f["id"] for f in frames]
    with_gold = [f["id"] for f in frames if f["oracle"] and f["distractor"]]
    contrasts = {}
    for channel, arm in {"notes": "false_text", "peers": "false"}.items():
        forms = {"C1": (items, lambda i, ms: did(i, "retrieved", ms, arm)), "C2": (items, lambda i, ms: did(i, "retrieved", ms, arm) - did(i, "distractor_retrieved", ms, arm)), "C3": (with_gold, lambda i, ms: did(i, "gold", ms, arm) - did(i, "distractor_gold", ms, arm))}
        for name, (keep, fn) in forms.items():
            for label, ms in {"pooled": models, **{m: [m] for m in models}}.items():
                vals = [fn(i, ms) for i in keep]
                sd = statistics.stdev(vals)
                contrasts[f"{channel}:{name}:{label}"] = {"estimate": statistics.mean(vals), "sd": sd, "n_for_half_width_5pts": math.ceil((1.96 * sd / 0.05) ** 2), "se_now": sd / math.sqrt(len(vals)), "claims": len(vals)}
                print(channel, name, label, {k: round(v, 3) for k, v in contrasts[f"{channel}:{name}:{label}"].items()}, flush=True)
    out = ROOT / "experiments" / "feasibility"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({"claims": len(frames), "seed": 1, "note": "pre-freeze feasibility on FEVER claims disjoint from the study frames; single seed; not part of the reported results; n_for_half_width_5pts ignores clustering and seed averaging", "retrieval_hit_rate": sum(f["retrieval_hit"] for f in frames) / len(frames), "by_model": table, "contrasts": contrasts}, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
