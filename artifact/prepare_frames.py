#!/usr/bin/env python3
"""Build the frozen claim frames: a fixed FEVER sample and all labelled SciFact claims, each with BM25-retrieved
sentences from an independent corpus, the official gold evidence, and whether retrieval contains the official evidence.

Inputs (not in Git): GROUND_TRUTH_DIR holding shared_task_dev.jsonl, fever_wiki.sqlite (FTS5 index of the FEVER Wikipedia
dump, see build_index below) and SciFact data/{claims_train,claims_dev,corpus}.jsonl.
Outputs: GROUND_TRUTH_DIR/frames/{fever,scifact}.jsonl, one JSON object per claim, and frames/SHA256SUMS.
Deterministic: fixed frame seed, no model calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path

FRAME_SEED = 2027
FEVER_PER_LABEL = 200
TOP_PAGES = 5
TOP_SENTENCES = 3
STOP = set("a an the of in on at to for and or is are was were be been by with as from that this it its his her their has have had".split())


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def fever_clean(text: str) -> str:
    text = text.replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]").replace("-COLON-", ":")
    return re.sub(r"\s+([.,;:)\]'])", r"\1", re.sub(r"([(\[])\s+", r"\1", text)).strip()


def fever_lines(raw: str) -> dict[int, str]:
    out = {}
    for row in raw.split("\n"):
        parts = row.split("\t")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].strip():
            out[int(parts[0])] = parts[1]
    return out


def rank_sentences(query: list[str], sentences: list[tuple[str, int, str]], k: int) -> list[tuple[str, int, str]]:
    """BM25 over the candidate sentences; ties break on (page, line) so the result is deterministic."""
    docs = [tokens(s[2]) for s in sentences]
    if not docs:
        return []
    avg = sum(map(len, docs)) / len(docs) or 1.0
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)
    scored = []
    for sent, doc in zip(sentences, docs):
        tf = Counter(doc)
        score = 0.0
        for t in set(query):
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(doc) / avg))
        scored.append((-score, sent[0], sent[1], sent))
    scored.sort(key=lambda x: x[:3])
    return [x[3] for x in scored[:k]]


def pick_distractors(gold: list[str], candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """For each gold sentence, the unused non-evidence sentence of the same source closest in length (word count), ties
    broken by position, so the distractor block matches the gold block in length and topic but is not annotated evidence."""
    out, used = [], set()
    for g in gold:
        ranked = sorted((abs(len(c[1].split()) - len(g.split())), i) for i, c in enumerate(candidates) if i not in used)
        if ranked:
            used.add(ranked[0][1])
            out.append(candidates[ranked[0][1]])
    return out


def fever_frame(gt: Path, per_label: int = FEVER_PER_LABEL, seed: int = FRAME_SEED, exclude: set[int] = frozenset()) -> list[dict]:
    rows = [json.loads(l) for l in (gt / "shared_task_dev.jsonl").open(encoding="utf-8")]
    rng = random.Random(seed)
    picked = []
    for label in ("SUPPORTS", "REFUTES"):
        pool = sorted((r for r in rows if r["label"] == label and r["id"] not in exclude), key=lambda r: r["id"])
        picked += rng.sample(pool, per_label)
    db = sqlite3.connect(gt / "fever_wiki.sqlite")

    pid_lookup = {}
    wanted = {str(e[2]) for r in picked for group in r["evidence"] for e in group if e[2]}
    # the FTS table has no index on pid: one scan collects the pages the gold evidence needs
    for pid, lines in db.execute("SELECT pid, lines FROM pages"):
        if pid in wanted:
            pid_lookup[pid] = fever_lines(lines)
    out = []
    for r in sorted(picked, key=lambda x: x["id"]):
        query = tokens(fever_clean(r["claim"]))
        match = " OR ".join(f'"{t}"' for t in dict.fromkeys(query)) or '"the"'
        pages = db.execute("SELECT pid, lines FROM pages WHERE pages MATCH ? ORDER BY bm25(pages, 0.0, 4.0, 1.0, 0.0), pid LIMIT ?", (match, TOP_PAGES)).fetchall()
        cands = [(pid, i, s) for pid, raw in pages for i, s in sorted(fever_lines(raw).items())]
        top = rank_sentences(query, cands, TOP_SENTENCES)
        got = {(p, i) for p, i, _ in top}
        sets = [[(str(e[2]), int(e[3])) for e in group if e[2] is not None] for group in r["evidence"]]
        sets = [s for s in sets if s]
        hit = any(set(s) <= got for s in sets)
        page_hit = any({p for p, _ in s} <= {p for p, _ in got} for s in sets)
        first = sets[0]
        gold = [(p, i, pid_lookup.get(p, {}).get(i, "")) for p, i in first]
        annotated = {x for group in sets for x in group}
        same_pages = [(p, s) for p in dict.fromkeys(q for q, _ in first) for i, s in sorted(pid_lookup.get(p, {}).items()) if (p, i) not in annotated and len(s.split()) >= 4]
        distractor = pick_distractors([s for _, _, s in gold if s], same_pages)
        rest = [(p, s) for p, i, s in cands if (p, i) not in got and (p, i) not in annotated and len(s.split()) >= 4]
        distractor_retrieved = pick_distractors([s for _, _, s in top], rest)
        out.append({"id": f"fever-{r['id']}", "claim": fever_clean(r["claim"]), "gold": "TRUE" if r["label"] == "SUPPORTS" else "FALSE",
                    "official_label": r["label"], "cluster": first[0][0],
                    "retrieved": [{"source": p.replace("_", " "), "text": fever_clean(s)} for p, _, s in top],
                    "oracle": [{"source": p.replace("_", " "), "text": fever_clean(s)} for p, _, s in gold if s],
                    "distractor": [{"source": p.replace("_", " "), "text": fever_clean(s)} for p, s in distractor],
                    "distractor_retrieved": [{"source": p.replace("_", " "), "text": fever_clean(s)} for p, s in distractor_retrieved],
                    "retrieval_hit": hit, "retrieval_page_hit": page_hit})
    return out


def scifact_frame(gt: Path) -> list[dict]:
    corpus = {str(d["doc_id"]): d for d in map(json.loads, (gt / "data" / "corpus.jsonl").open(encoding="utf-8"))}
    sentences = [(doc, i, s) for doc, d in sorted(corpus.items()) for i, s in enumerate(d["abstract"])]
    df = Counter()
    doc_tokens = {}
    for doc, d in corpus.items():
        doc_tokens[doc] = tokens(d["title"] + " " + " ".join(d["abstract"]))
        df.update(set(doc_tokens[doc]))
    avg = sum(map(len, doc_tokens.values())) / len(doc_tokens)
    out = []
    claims = [json.loads(l) for f in ("claims_train", "claims_dev") for l in (gt / "data" / f"{f}.jsonl").open(encoding="utf-8")]
    for c in sorted(claims, key=lambda x: x["id"]):
        labels = {e["label"] for ev in c["evidence"].values() for e in ev}
        if len(labels) != 1:
            continue
        label = labels.pop()
        query = tokens(c["claim"])
        scored = []
        for doc, toks in doc_tokens.items():
            tf = Counter(toks)
            score = sum(math.log(1 + (len(doc_tokens) - df[t] + 0.5) / (df[t] + 0.5)) * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(toks) / avg)) for t in set(query) if t in tf)
            scored.append((-score, doc))
        scored.sort()
        pages = [doc for _, doc in scored[:TOP_PAGES]]
        cands = [(doc, i, s) for doc in pages for i, s in enumerate(corpus[doc]["abstract"])]
        top = rank_sentences(query, cands, TOP_SENTENCES)
        got = {(p, i) for p, i, _ in top}
        sets = [[(doc, int(i)) for i in e["sentences"]] for doc, ev in sorted(c["evidence"].items()) for e in ev]
        hit = any(set(s) <= got for s in sets)
        page_hit = bool(set(c["evidence"]) & {p for p, _ in got})
        first = sets[0]
        annotated = {x for group in sets for x in group}
        same_doc = [(p, corpus[p]["abstract"][i].strip()) for p in dict.fromkeys(q for q, _ in first) for i in range(len(corpus[p]["abstract"])) if (p, i) not in annotated]
        distractor = pick_distractors([corpus[p]["abstract"][i].strip() for p, i in first], same_doc)
        rest = [(p, s.strip()) for p, i, s in cands if (p, i) not in got and (p, i) not in annotated and len(s.split()) >= 4]
        distractor_retrieved = pick_distractors([s.strip() for _, _, s in top], rest)
        out.append({"id": f"scifact-{c['id']}", "claim": c["claim"], "gold": "TRUE" if label == "SUPPORT" else "FALSE", "official_label": label, "cluster": first[0][0],
                    "retrieved": [{"source": corpus[p]["title"], "text": s.strip()} for p, _, s in top],
                    "oracle": [{"source": corpus[p]["title"], "text": corpus[p]["abstract"][i].strip()} for p, i in first],
                    "distractor": [{"source": corpus[p]["title"], "text": s} for p, s in distractor],
                    "distractor_retrieved": [{"source": corpus[p]["title"], "text": s} for p, s in distractor_retrieved],
                    "retrieval_hit": hit, "retrieval_page_hit": page_hit})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", default=os.environ.get("GROUND_TRUTH_DIR", str(Path(__file__).resolve().parents[3] / "runtime" / "ground_truth" / "icsc2027")))
    ap.add_argument("--feasibility", type=int, default=0, help="write frames/feasibility.jsonl: this many FEVER claims per label, disjoint from frames/fever.jsonl and not part of the results")
    ap.add_argument("--extend-fever", type=int, default=0, help="append this many FEVER claims per label to frames/fever.jsonl, disjoint from the existing frame and frames/feasibility.jsonl; the existing rows stay as they are")
    args = ap.parse_args()
    gt = Path(args.ground_truth)
    frames = gt / "frames"
    frames.mkdir(exist_ok=True)
    if args.feasibility:
        used = {int(json.loads(l)["id"].split("-")[1]) for l in (frames / "fever.jsonl").open(encoding="utf-8")}
        rows = fever_frame(gt, per_label=args.feasibility, seed=FRAME_SEED + 1, exclude=used)
        (frames / "feasibility.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
        print("feasibility", len(rows), "hit", sum(r["retrieval_hit"] for r in rows))
        return
    if args.extend_fever:
        path = frames / "fever.jsonl"
        used = {int(json.loads(l)["id"].split("-")[1]) for name in ("fever", "feasibility") for l in (frames / f"{name}.jsonl").open(encoding="utf-8")}
        rows = fever_frame(gt, per_label=args.extend_fever, seed=FRAME_SEED + 2, exclude=used)
        with path.open("a", encoding="utf-8") as out:
            out.write("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
        print("fever extended by", len(rows), "hit", sum(r["retrieval_hit"] for r in rows))
        return
    sums = []
    for name, build in (("scifact", scifact_frame), ("fever", fever_frame)):
        rows = build(gt)
        path = frames / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
        sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
        print(name, len(rows), "hit", sum(r["retrieval_hit"] for r in rows), "page_hit", sum(r["retrieval_page_hit"] for r in rows), "TRUE", sum(r["gold"] == "TRUE" for r in rows), "oracle", sum(bool(r["oracle"]) for r in rows), "distractor", sum(len(r["distractor"]) == len(r["oracle"]) and bool(r["oracle"]) for r in rows), "distractor_retrieved", sum(len(r["distractor_retrieved"]) == len(r["retrieved"]) and bool(r["retrieved"]) for r in rows), flush=True)
    (frames / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
