#!/usr/bin/env python3
"""How much the stimulus validation selects: no-context alone accuracy, seed 1, on every claim of each frame, split by whether the
claim passed validation. A rejected claim enters no arm of the study, so this file is its only measurement. Writes
experiments/selection.json. Run once per model, then with no argument for the final file."""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
from pathlib import Path

from run_cell import FLEET, GROUND_TRUTH, PARALLEL, Client, cell_cache, parse_label, request_seed, scripted_prompt


def main() -> None:
    models = sys.argv[1:] or FLEET
    out: dict = {"seed": 1, "note": "no-context alone accuracy on all frame claims, by validation outcome", "rows": []}
    for dataset in ("fever", "scifact"):
        frames = [json.loads(l) for l in (GROUND_TRUTH / "frames" / f"{dataset}.jsonl").open(encoding="utf-8")]
        rows = {r["id"]: r for r in map(json.loads, (GROUND_TRUTH / "frames" / f"{dataset}.rationales.jsonl").open(encoding="utf-8"))}
        for model in models:
            client = Client(model, cell_cache(dataset, 1, model), preflight=True)

            def one(frame: dict) -> tuple[bool, str, bool]:
                prompt = scripted_prompt(frame, "no_context", "alone", {frame["id"]: rows[frame["id"]]["rationales"]})
                text = client.ask("", prompt, request_seed(1, frame["id"], "no_context/alone", model), 8)["text"]
                return rows[frame["id"]]["valid"], frame["gold"], parse_label(text) == frame["gold"]

            with cf.ThreadPoolExecutor(PARALLEL) as pool:
                answers = list(pool.map(one, frames))
            for valid in (True, False):
                for gold in ("TRUE", "FALSE", "all"):
                    part = [c for v, g, c in answers if v == valid and gold in (g, "all")]
                    out["rows"].append({"dataset": dataset, "model": model, "validated": valid, "gold": gold, "claims": len(part), "alone_accuracy": sum(part) / len(part) if part else None})
    path = Path(__file__).resolve().parents[1] / "experiments" / "selection.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
