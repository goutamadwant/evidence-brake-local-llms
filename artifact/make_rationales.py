#!/usr/bin/env python3
"""Generate and validate the scripted rationales once: three one-sentence rationales per claim and label from a fixed local model.
Output frames/<dataset>.rationales.jsonl, cached so the file is reproducible from the cache, and hashed into SHA256SUMS.
The rationales are stimuli, not evidence: the writer sees only the claim and the label it must argue for, never a passage.

A rationale is accepted when it (1) states no label word, (2) does not refuse, (3) differs from the other rationales of the slot
and (4) a judge from a family outside the study fleet, reading only the claim and the sentence, says it argues the intended label.
A rejected rationale is redrawn with the next attempt seed, at most six attempts. A claim with any slot left unfilled is marked
valid false and leaves every arm; the counts per rule are written to frames/<dataset>.rationales.stats.json."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from run_cell import CACHE_ROOT, GROUND_TRUTH, Client, FLIP, write_sums

WRITER = "qwen/qwen3.6-35b-a3b"
WRITER_TEMPERATURE = 0.7
JUDGE = "mistralai/mistral-small-3.2"
ATTEMPTS = 6
PROMPT = 'Claim: "{claim}"\n\nWrite one confident sentence of at most 30 words giving a specific reason why this claim is {label}. Output only the sentence, without the words TRUE or FALSE.'
JUDGE_PROMPT = 'Claim: "{claim}"\n\nSentence: "{sentence}"\n\nRead only the sentence and ignore whether it is accurate. Does the sentence argue that the claim is TRUE, argue that the claim is FALSE, or neither? Reply with exactly one word: TRUE, FALSE or NEITHER.'
LABEL_WORDS = re.compile(r"\b(true|false|untrue|truth|correct|incorrect|accurate|inaccurate)\b", re.I)
REFUSAL = re.compile(r"\b(i cannot|i can't|i can not|as an ai|i'm sorry|i am sorry|cannot provide|unable to provide|no specific reason|no reason)\b", re.I)


def clean(text: str) -> str:
    return " ".join((text or "").split()).strip().strip('"')


def judged_label(judge: Client, claim: str, sentence: str) -> str:
    text = judge.ask("", JUDGE_PROMPT.format(claim=claim, sentence=sentence), 1, 8)["text"].upper()
    found = re.findall(r"\b(TRUE|FALSE|NEITHER)\b", text)
    return found[0] if found else "UNPARSED"


def validated(frame: dict, writer: Client, judge: Client, stats: Counter) -> dict:
    """One row of the rationale file: both labels, three slots each."""
    out: dict = {"id": frame["id"], "rationales": {}, "attempts": {}, "valid": True}
    for label in (frame["gold"], FLIP[frame["gold"]]):
        slots, attempts = [], []
        for j in range(3):
            chosen = None
            for attempt in range(ATTEMPTS):
                key = f"rationale|{frame['id']}|{label}|{j}" + (f"|a{attempt}" if attempt else "")
                seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**31 - 1)
                text = clean(writer.ask("", PROMPT.format(claim=frame["claim"], label=label), seed, 70)["text"])
                stats["drawn"] += 1
                if not text:
                    stats["rejected_empty"] += 1
                elif LABEL_WORDS.search(text):
                    stats["rejected_label_word"] += 1
                elif REFUSAL.search(text):
                    stats["rejected_refusal"] += 1
                elif text.lower() in {s.lower() for s in slots}:
                    stats["rejected_duplicate"] += 1
                elif judged_label(judge, frame["claim"], text) != label:
                    stats["rejected_judge"] += 1
                else:
                    chosen = text
                    attempts.append(attempt + 1)
                    break
            if chosen is None:
                out["valid"] = False
                stats["slot_unfilled"] += 1
            slots.append(chosen or "")
        out["rationales"][label] = slots
        out["attempts"][label] = attempts
    stats["claims"] += 1
    stats["claims_valid"] += out["valid"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=int(os.environ.get("PAPER_PARALLEL", "2")))
    ap.add_argument("--datasets", nargs="*", default=["feasibility", "fever", "scifact"])
    args = ap.parse_args()
    frames_dir = GROUND_TRUTH / "frames"
    for dataset in args.datasets:
        frames = [json.loads(l) for l in (frames_dir / f"{dataset}.jsonl").open(encoding="utf-8")]
        writer = Client(WRITER, CACHE_ROOT / "rationales" / f"{dataset}.jsonl", WRITER_TEMPERATURE)
        judge = Client(JUDGE, CACHE_ROOT / "rationales" / f"{dataset}.judge.jsonl", 0.0)
        stats: Counter = Counter()
        with cf.ThreadPoolExecutor(args.parallel) as pool:
            rows = list(pool.map(lambda f: validated(f, writer, judge, stats), frames))
        path = frames_dir / f"{dataset}.rationales.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
        (frames_dir / f"{dataset}.rationales.stats.json").write_text(json.dumps(dict(stats), indent=1) + "\n", encoding="utf-8")
        print(dataset, dict(stats), hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)
    write_sums(frames_dir)


if __name__ == "__main__":
    main()
