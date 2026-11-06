#!/usr/bin/env python3
"""Audit of the context blocks before any study cell: a judge outside the study fleet reads the claim and one block and says whether
the block supports it, refutes it or neither. A distractor judged anything but NEITHER is contaminated; C2_clean and C3_clean drop
such claims. Gold and retrieved blocks are judged the same way, as a check on the judge. Output frames/<dataset>.audit.jsonl."""
from __future__ import annotations

import concurrent.futures as cf
import json
import re
import sys
from collections import Counter

from make_rationales import JUDGE
from run_cell import CACHE_ROOT, GROUND_TRUTH, Client, write_sums

BLOCKS = ("retrieved", "distractor_retrieved", "oracle", "distractor")
PROMPT = 'Claim: "{claim}"\n\nPassages:\n{block}\n\nDo these passages support the claim, refute it, or neither? Reply with exactly one word: SUPPORTS, REFUTES or NEITHER.'


def judge_block(client: Client, frame: dict, field: str) -> str:
    if not frame[field]:
        return "EMPTY"
    block = "\n".join(f"[{i}] {p['text']}" for i, p in enumerate(frame[field], 1))
    found = re.findall(r"\b(SUPPORTS|REFUTES|NEITHER)\b", client.ask("", PROMPT.format(claim=frame["claim"], block=block), 1, 8)["text"].upper())
    return found[0] if found else "UNPARSED"


def main() -> None:
    frames_dir = GROUND_TRUTH / "frames"
    for dataset in sys.argv[1:] or ["fever", "scifact"]:
        frames = [json.loads(l) for l in (frames_dir / f"{dataset}.jsonl").open(encoding="utf-8")]
        client = Client(JUDGE, CACHE_ROOT / "rationales" / f"{dataset}.audit.jsonl", 0.0)
        with cf.ThreadPoolExecutor(2) as pool:
            rows = list(pool.map(lambda f: {"id": f["id"], "gold": f["gold"], **{b: judge_block(client, f, b) for b in BLOCKS}}, frames))
        (frames_dir / f"{dataset}.audit.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
        print(dataset, {b: dict(Counter(r[b] for r in rows)) for b in BLOCKS}, flush=True)
    write_sums(frames_dir)


if __name__ == "__main__":
    main()
