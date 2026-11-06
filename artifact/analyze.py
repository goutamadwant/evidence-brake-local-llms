#!/usr/bin/env python3
"""Turn the registry into the results manifest, generated macros, the table and the figure. Every reported number comes from
here. Unit of analysis: the claim. Per claim a condition's accuracy is the mean over seeds (and over models for the pooled
estimates). Confidence intervals and p-values come from a paired bootstrap that resamples source clusters (the first
official evidence page or abstract), 10,000 resamples, seed 20270201.

Primary contrasts, per dataset, pooled across models, Holm across the twelve, for each channel (notes: the false rationales
without speakers; peers: the same rationales attributed to three agents with the wrong label):
  C1 DiD(retrieved)                       against 0
  C2 DiD(retrieved) - DiD(distractor_retrieved), on claims whose blocks match
  C3 DiD(gold)      - DiD(distractor_gold), on claims whose blocks match
Two blocks match when they have the same sentence count and their word counts differ by at most 20% of the evidence block.
Secondary: CH drop(notes) - drop(peers) with no context, per-model rows, C2 and C3 on all claims with both blocks, C2h and C3h on
a common stratum, accuracy levels under the channel (L1, L2, L3), and C3 with the block placed last or with a reasoned answer on
the fixed subset of claims, each with its direct difference from C3 on the same claims.
where DiD(c) = [A(no_context, alone) - A(no_context, misinformed)] - [A(c, alone) - A(c, misinformed)].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = Path(os.environ.get("GROUND_TRUTH_DIR", str(ROOT.parents[1] / "runtime" / "ground_truth" / "icsc2027")))
RESAMPLES = int(os.environ.get("PAPER_RESAMPLES", "10000"))
SESOI = 0.05
FLOOR = 0.10
MATCH_TOLERANCE = 0.20
ALPHA = 0.05
PRIMARY_ROWS = 12
DATASETS = {"fever": "e_fever", "scifact": "e_scifact"}
MODELS = ["qwen/qwen3.6-35b-a3b", "qwen3-coder-30b-a3b-instruct", "gemma-4-31b-it-qat"]
MODEL_SHORT = {"qwen/qwen3.6-35b-a3b": "qwen3.6-35B", "qwen3-coder-30b-a3b-instruct": "qwen3-coder-30B", "gemma-4-31b-it-qat": "gemma-4-31B"}
SCRIPTED = ["no_context", "instruction", "retrieved", "distractor_retrieved", "gold", "distractor_gold"]
CHANNELS = {"notes": "false_text", "peers": "false"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(registry_path: Path, frames_dir: Path, kinds: set[str]) -> tuple[dict, dict, dict, dict]:
    """(dataset, method, model, seed, item, arm) -> correct, group rows, frames, parse counts. Only current ok runs of the given kinds count;
    the latest ok run wins for a cell (every run stays in the registry)."""
    runs = json.loads(registry_path.read_text(encoding="utf-8"))["runs"]
    latest: dict[tuple, dict] = {}
    for run in runs:
        if run.get("status") == "ok" and run.get("kind") in kinds:
            latest[(run["dataset"], run["method"], str(run.get("model")), str(run["seed"]))] = run
    scripted, group = {}, defaultdict(dict)
    parse: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for (dataset, method, _, seed), run in latest.items():
        path = next(ROOT / o for o in run["outputs"] if str(o).endswith("predictions.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            count = parse[(dataset, method, r["model"], r["arm"])]
            count[0] += 1
            count[1] += not r["parsed"]
            count[2] += bool(r["correct"])
            count[3] += r.get("answer") == "FALSE"
            if method.startswith("group_"):
                group[(dataset, method, seed, r["item"])].setdefault(r["arm"], {})[r["model"]] = r["answer"]
            else:
                scripted[(dataset, method, r["model"], seed, r["item"], r["arm"])] = bool(r["correct"])
    frames = {}
    for d in DATASETS:
        if (frames_dir / f"{d}.jsonl").exists():
            rows = {r["id"]: r for r in map(json.loads, (frames_dir / f"{d}.jsonl").open(encoding="utf-8"))}
            stimuli, audit = frames_dir / f"{d}.rationales.jsonl", frames_dir / f"{d}.audit.jsonl"
            if stimuli.exists():  # a claim whose stimuli failed validation leaves every contrast
                rows = {r["id"]: rows[r["id"]] for r in map(json.loads, stimuli.open(encoding="utf-8")) if r["valid"]}
            if audit.exists():
                for r in map(json.loads, audit.open(encoding="utf-8")):
                    if r["id"] in rows:
                        rows[r["id"]]["audit"] = r
            frames[d] = rows
    return scripted, group, frames, parse


class Data:
    """Outcomes keyed by dataset, method, model, arm, claim and seed. Every paired comparison joins on those keys, never on order."""

    def __init__(self, scripted: dict, frames: dict):
        self.frames = frames
        self.values: dict[tuple, dict[str, dict[str, bool]]] = defaultdict(lambda: defaultdict(dict))
        self.seeds: set[str] = set()
        for (dataset, method, model, seed, item, arm), correct in scripted.items():
            if item in frames[dataset]:
                self.values[(dataset, method, model, arm)][item][seed] = correct
                self.seeds.add(seed)
        self.verify()

    def verify(self) -> None:
        """Within one method, model and dataset every arm must hold the same claims and seeds; a gap would break the pairing silently."""
        cells: dict[tuple, list[dict]] = defaultdict(list)
        for (dataset, method, model, _), items in self.values.items():
            cells[(dataset, method, model)].append(items)
        for key, arms in cells.items():
            shapes = {frozenset((i, s) for i, d in items.items() for s in d) for items in arms}
            if len(shapes) != 1:
                raise ValueError(f"incomplete arms in {key}")

    def accuracy(self, dataset: str, method: str, arm: str, models: list[str]) -> dict[str, float]:
        """Per-claim mean over seeds and the given models; a claim must be present for every model."""
        out = {}
        for item in self.frames[dataset]:
            vals = [self.values.get((dataset, method, m, arm), {}).get(item) for m in models]
            if all(vals):
                out[item] = sum(sum(v.values()) / len(v) for v in vals) / len(vals)
        return out


def did_series(data: Data, dataset: str, context: str, models: list[str], false_arm: str = "false") -> dict[str, float]:
    a0, f0 = data.accuracy(dataset, "no_context", "alone", models), data.accuracy(dataset, "no_context", false_arm, models)
    ac, fc = data.accuracy(dataset, context, "alone", models), data.accuracy(dataset, context, false_arm, models)
    return {i: (a0[i] - f0[i]) - (ac[i] - fc[i]) for i in a0 if i in f0 and i in ac and i in fc}


def clean_block(frame: dict, distractor: str) -> bool:
    """The pre-study audit judged this distractor as neither supporting nor refuting the claim."""
    return frame.get("audit", {}).get(distractor) == "NEITHER"


def both_blocks(frame: dict, evidence: str, distractor: str) -> bool:
    return bool(frame[evidence] and frame[distractor])


def blocks_match(frame: dict, evidence: str, distractor: str) -> bool:
    """Same sentence count, and word counts within the declared tolerance of the evidence block."""
    if not both_blocks(frame, evidence, distractor) or len(frame[evidence]) != len(frame[distractor]):
        return False
    words = [sum(len(p["text"].split()) for p in frame[k]) for k in (evidence, distractor)]
    return abs(words[0] - words[1]) <= MATCH_TOLERANCE * words[0]


def contrast_series(data: Data, dataset: str, models: list[str], false_arm: str, matched_only: bool = True) -> dict[str, dict[str, float]]:
    """C1, C2, C3 for one channel; C2 and C3 keep claims whose blocks match, or with matched_only False all claims with both blocks."""
    d_ret, d_dr = did_series(data, dataset, "retrieved", models, false_arm), did_series(data, dataset, "distractor_retrieved", models, false_arm)
    d_gold, d_dg = did_series(data, dataset, "gold", models, false_arm), did_series(data, dataset, "distractor_gold", models, false_arm)
    frames = data.frames[dataset]
    keep = blocks_match if matched_only else both_blocks
    return {"C1": d_ret, "C2": {i: d_ret[i] - d_dr[i] for i in d_ret if i in d_dr and keep(frames[i], "retrieved", "distractor_retrieved")},
            "C3": {i: d_gold[i] - d_dg[i] for i in d_gold if i in d_dg and keep(frames[i], "oracle", "distractor")}}


def level_series(data: Data, dataset: str, models: list[str], false_arm: str) -> dict[str, dict[str, float]]:
    """Accuracy under the channel itself: a block can raise it without shrinking the drop, so the drop is never reported alone."""
    a = {m: data.accuracy(dataset, m, false_arm, models) for m in ("no_context", "retrieved", "distractor_retrieved", "gold", "distractor_gold")}
    frames = data.frames[dataset]
    return {"L1": {i: a["retrieved"][i] - a["no_context"][i] for i in a["retrieved"] if i in a["no_context"]},
            "L2": {i: a["retrieved"][i] - a["distractor_retrieved"][i] for i in a["retrieved"] if i in a["distractor_retrieved"] and blocks_match(frames[i], "retrieved", "distractor_retrieved")},
            "L3": {i: a["gold"][i] - a["distractor_gold"][i] for i in a["gold"] if i in a["distractor_gold"] and blocks_match(frames[i], "oracle", "distractor")}}


def drop_series(data: Data, dataset: str, method: str, models: list[str], false_arm: str) -> dict[str, float]:
    a, f = data.accuracy(dataset, method, "alone", models), data.accuracy(dataset, method, false_arm, models)
    return {i: a[i] - f[i] for i in a if i in f}


def merged_clusters(frames: dict) -> dict[str, str]:
    """Source-overlap clusters: first-source clusters joined whenever a block of either claim cites the same page or abstract."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def name(text: object) -> str:
        return " ".join(str(text).replace("_", " ").split())

    for frame in frames.values():
        for block in ("retrieved", "distractor_retrieved", "oracle", "distractor"):
            for sentence in frame.get(block) or []:
                parent[find(name(sentence["source"]))] = find(name(frame["cluster"]))
    return {item: find(name(frame["cluster"])) for item, frame in frames.items()}


def bootstrap(series: dict[str, float], frames: dict, seed: int, cluster_key: str = "cluster") -> dict:
    """Cluster bootstrap over source pages for the mean of a per-claim series."""
    clusters: dict[str, list[float]] = defaultdict(list)
    for item, value in series.items():
        clusters[frames[item][cluster_key]].append(value)
    keys = sorted(clusters)
    sums = [(sum(clusters[k]), len(clusters[k])) for k in keys]
    rng = random.Random(seed)
    n_keys = len(keys)
    stats = []
    for _ in range(RESAMPLES):
        total, count = 0.0, 0
        for _ in range(n_keys):
            s, c = sums[rng.randrange(n_keys)]
            total += s
            count += c
        stats.append(total / count)
    stats.sort()
    estimate = sum(series.values()) / len(series)
    low, high = stats[int(0.025 * RESAMPLES)], stats[int(0.975 * RESAMPLES) - 1]
    familywise_high = stats[min(RESAMPLES - 1, int((1 - ALPHA / PRIMARY_ROWS) * RESAMPLES))]
    tail = min(sum(s <= 0 for s in stats), sum(s >= 0 for s in stats))
    return {"estimate": estimate, "ci_low": low, "ci_high": high, "familywise_high": familywise_high, "p_value": min(1.0, 2 * (tail + 1) / (RESAMPLES + 1)), "n": len(series), "clusters": n_keys}


def holm(rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda r: r["p_value"])
    running = 0.0
    m = len(ordered)
    for rank, row in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * row["p_value"]))
        row["corrected_p"] = running


def decide(row: dict) -> str:
    """The pre-registered rule: positive effect, excluded relevant effect, or inconclusive; size is a separate description.
    An exclusion is a claim about all twelve rows at once, so it uses the one-sided bound at alpha/12, not the printed 95% interval.
    apply_floor() then relabels an exclusion where the channel's own no-context drop is too small for a brake to be seen."""
    if row["corrected_p"] < 0.05 and row["ci_low"] > 0:
        return "supported"  # positive takes precedence and carries no magnitude label
    if row["familywise_high"] < SESOI:
        return "supported_negative"
    return "inconclusive"


def combine_decisions(first_source: str, merged: str) -> str:
    """The reported decision: a conclusion stands only if both clusterings reach it (decision_rules.overlap)."""
    return first_source if first_source == merged else "inconclusive"


def apply_floor(primary: list[dict], desc: dict) -> None:
    arm = {"notes": "notes_drop", "peers": "drop"}
    for row in primary:
        if row["decision"] == "supported_negative" and row["channel"] in arm and desc[row["dataset"]]["pooled"]["no_context"].get(arm[row["channel"]], 0.0) < FLOOR:
            row["decision"] = "floor_limited"


def transitions(data: Data, dataset: str, context: str, models: list[str]) -> dict:
    """Harmful: correct alone in no_context and in context, wrong under the false channel. Beneficial: wrong alone in both, right under
    true peers. The stratum is per model and seed; each claim counts once, so rates and differences are the same claim-weighted
    estimand. Beneficial non-inferiority: comparator no_context, margin 5 points, H0 difference <= -5, decided by the 95% lower bound,
    not estimable under 50 claims; four uncorrected exploratory tests (retrieved and gold on two datasets)."""
    out: dict[str, dict] = {}
    for name, base_correct, arm, wanted in (("harmful", True, "false", False), ("harmful_notes", True, "false_text", False), ("beneficial", False, "true", True)):
        hits_c, hits_0, base = defaultdict(int), defaultdict(int), defaultdict(int)
        for model in models:
            base0 = data.values.get((dataset, "no_context", model, "alone"), {})
            basec = data.values.get((dataset, context, model, "alone"), {})
            armc = data.values.get((dataset, context, model, arm), {})
            arm0 = data.values.get((dataset, "no_context", model, arm), {})
            for item, draws in base0.items():
                for seed, correct in draws.items():
                    if seed in basec.get(item, {}) and correct == base_correct and basec[item][seed] == base_correct:
                        base[item] += 1
                        hits_c[item] += armc[item][seed] == wanted
                        hits_0[item] += arm0[item][seed] == wanted
        rate_c, rate_0 = {i: hits_c[i] / base[i] for i in base}, {i: hits_0[i] / base[i] for i in base}
        mean = lambda d: sum(d.values()) / max(1, len(d))  # noqa: E731
        out[name] = {"rate_context": mean(rate_c), "rate_no_context": mean(rate_0), "stratum_claims": len(base), "stratum_observations": sum(base.values())}
        if base:
            out[name]["difference"] = bootstrap({i: rate_c[i] - rate_0[i] for i in base}, data.frames[dataset], 20270300 + len(name))
    b = out["beneficial"]
    b["non_inferior"] = "not_estimable" if b["stratum_claims"] < 50 else "yes" if b["difference"]["ci_low"] > -SESOI else "no"
    return out


def conditional_contrast(data: Data, dataset: str, models: list[str], arm: str, context: str, distractor: str) -> dict[str, float]:
    """Per claim, over the model and seed draws correct alone under both the evidence and its distractor: the wrong rate under the
    channel with the distractor minus with the evidence. Both rates share one stratum, so they compare the same claims and draws;
    positive means less harm with the evidence, as in C2 and C3. Conditioning on alone answers makes this a sensitivity only."""
    harm_c, harm_d, base = defaultdict(int), defaultdict(int), defaultdict(int)
    frames = data.frames[dataset]
    for model in models:
        alone_c, alone_d = data.values.get((dataset, context, model, "alone"), {}), data.values.get((dataset, distractor, model, "alone"), {})
        under_c, under_d = data.values.get((dataset, context, model, arm), {}), data.values.get((dataset, distractor, model, arm), {})
        for item in alone_c:
            if not blocks_match(frames[item], "oracle" if context == "gold" else context, "distractor" if context == "gold" else distractor):
                continue
            for seed, a in alone_c[item].items():
                if a and alone_d[item][seed]:
                    base[item] += 1
                    harm_c[item] += not under_c[item][seed]
                    harm_d[item] += not under_d[item][seed]
    return {i: (harm_d[i] - harm_c[i]) / base[i] for i in base}


def block_contrast(data: Data, dataset: str, models: list[str], arm: str, evidence: str, distractor: str) -> dict[str, float]:
    """drop(distractor) - drop(evidence) for the control methods that have no no-context pair of their own; same sign as C3."""
    e, d = drop_series(data, dataset, evidence, models, arm), drop_series(data, dataset, distractor, models, arm)
    frames = data.frames[dataset]
    return {i: d[i] - e[i] for i in e if i in d and blocks_match(frames[i], "oracle", "distractor")}


def apply_dependence(primary: list[dict], secondary: list[dict]) -> None:
    """A C3 row depends on block position or format only when the control is positive and its direct difference from C3 is too. A C2 or
    C3 row depends on distractor contamination when it is not positive and the same contrast on clean distractors is. All descriptive."""
    rows = {(r["channel"], r["contrast"], r["dataset"]): r for r in secondary}
    for row in primary:
        clean = rows.get((row["channel"], row["contrast"] + "_clean", row["dataset"]))
        row["contamination_dependent"] = bool(clean and row["decision"] != "supported" and clean["ci_low"] > 0)
        for name, control, gap in (("position_dependent", "C3_last", "C3_last_gap"), ("format_dependent", "C3_reasoned", "C3_reasoned_gap")):
            a, b = rows.get((row["channel"], control, row["dataset"])), rows.get((row["channel"], gap, row["dataset"]))
            row[name] = bool(row["contrast"] == "C3" and a and b and a["ci_low"] > 0 and b["ci_low"] > 0)


def model_dependent(row: dict, per_model: list[dict]) -> bool:
    """Pooling hides a disagreement: one model's interval lies above zero and another's below it."""
    rows = [r for r in per_model if (r["channel"], r["contrast"], r["dataset"]) == (row["channel"], row["contrast"], row["dataset"])]
    return any(r["ci_low"] > 0 for r in rows) and any(r["ci_high"] < 0 for r in rows)


def apply_room(primary: list[dict], secondary: list[dict]) -> None:
    """A relevant effect is not excluded when a smaller room to fall could hide it: the conditional contrast is positive."""
    cond = {(r["channel"], r["contrast"], r["dataset"]): r for r in secondary if r["contrast"] in {"C2h", "C3h"}}
    for row in primary:
        match = cond.get((row["channel"], row["contrast"] + "h", row["dataset"]))
        if row["decision"] == "supported_negative" and match and match["ci_low"] > 0:
            row["decision"] = "room_limited"


def seed_disagreement(data: Data, dataset: str, model: str) -> float:
    """Share of claims whose no-context alone outcome differs between seeds: zero would mean the seeds are one stream."""
    draws = data.values.get((dataset, "no_context", model, "alone"), {})
    return sum(len(set(v.values())) > 1 for v in draws.values()) / max(1, len(draws))


def spontaneous(data: Data, dataset: str, method: str, models: list[str]) -> float:
    changed, total = 0, 0
    for model in models:
        a, b = data.values.get((dataset, method, model, "alone"), {}), data.values.get((dataset, method, model, "alone2"), {})
        for item, draws in a.items():
            for seed, x in draws.items():
                changed += x != b[item][seed]
                total += 1
    return changed / max(1, total)


def majority(answers: dict[str, str | None]) -> str | None:
    vals = list(answers.values())
    t, f = vals.count("TRUE"), vals.count("FALSE")
    return "TRUE" if t > f else "FALSE" if f > t else None


def group_rows(group: dict, frames: dict) -> list[dict]:
    per: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (dataset, method, _seed, item), arms in group.items():
        gold = frames[dataset][item]["gold"]
        for arm, answers in arms.items():
            per[(dataset, method, arm)][item].append(float(majority(answers) == gold))
    series = {k: {i: sum(v) / len(v) for i, v in d.items()} for k, d in per.items()}
    rows = []
    for dataset in DATASETS:
        for method in ("group_none", "group_retrieved"):
            a, b = series.get((dataset, method, "deliberated")), series.get((dataset, method, "independent"))
            if a and b:
                diff = {i: a[i] - b[i] for i in a if i in b}
                rows.append({"id": f"group-{method}-{dataset}", "dataset": dataset, "method": method, "vote_independent": sum(b.values()) / len(b), "vote_deliberated": sum(a.values()) / len(a), **bootstrap(diff, frames[dataset], 20270400 + len(method))})
        a, b = series.get((dataset, "group_retrieved", "independent")), series.get((dataset, "group_none", "independent"))
        if a and b:
            diff = {i: a[i] - b[i] for i in a if i in b}
            rows.append({"id": f"group-evidence-vote-{dataset}", "dataset": dataset, "method": "vote_retrieved_vs_vote_none", **bootstrap(diff, frames[dataset], 20270500)})
    return rows


def macro(name: str, value: float | int | str, digits: int = 1, pct: bool = False) -> str:
    if isinstance(value, str):
        text = value
    elif pct:
        text = f"{100 * value:.{digits}f}"
    elif isinstance(value, int):
        text = str(value)
    else:
        text = f"{value:.{digits}f}"
    return f"\\newcommand{{\\{name}}}{{{text}}}"


def analyse(registry: Path, frames_dir: Path, kinds: set[str], models: list[str], datasets: list[str]) -> dict:
    scripted, group, frames, parse = load(registry, frames_dir, kinds)
    data = Data(scripted, frames)
    primary, secondary, primary_series = [], [], {}
    for dataset in datasets:
        for item, merged in merged_clusters(frames[dataset]).items():
            frames[dataset][item]["merged_cluster"] = merged
        for offset, (channel, arm) in enumerate(CHANNELS.items()):
            for name, series in contrast_series(data, dataset, models, arm).items():
                if series:
                    primary_series[f"{channel}-{name}-{dataset}"] = series
                    primary.append({"id": f"{channel}-{name}-{dataset}", "channel": channel, "contrast": name, "dataset": dataset, "models": "pooled", **bootstrap(series, frames[dataset], 20270100 + int(name[1]) + 10 * offset)})
            for model in models:
                short = MODEL_SHORT.get(model, model)
                for name, series in contrast_series(data, dataset, [model], arm).items():
                    if series:
                        secondary.append({"id": f"{channel}-{name}-{dataset}-{short}", "channel": channel, "contrast": name, "dataset": dataset, "models": short, **bootstrap(series, frames[dataset], 20270200 + int(name[1]) + 10 * offset)})
            for name, (context, distractor) in {"C2h": ("retrieved", "distractor_retrieved"), "C3h": ("gold", "distractor_gold")}.items():
                series = conditional_contrast(data, dataset, models, arm, context, distractor)
                if series:
                    secondary.append({"id": f"{channel}-{name}-{dataset}", "channel": channel, "contrast": name, "dataset": dataset, "models": "pooled", **bootstrap(series, frames[dataset], 20270900 + int(name[1]) + 10 * offset)})
            for name, (evidence, distractor) in {"C3_last": ("gold_last", "distractor_gold_last"), "C3_reasoned": ("reasoned_gold", "reasoned_distractor_gold")}.items():
                series = block_contrast(data, dataset, models, arm, evidence, distractor)
                if series:
                    secondary.append({"id": f"{channel}-{name}-{dataset}", "channel": channel, "contrast": name, "dataset": dataset, "models": "pooled", **bootstrap(series, frames[dataset], 20271000 + len(name) + 10 * offset)})
                    c3 = contrast_series(data, dataset, models, arm)["C3"]
                    gap = {i: series[i] - c3[i] for i in series if i in c3}
                    if gap:
                        secondary.append({"id": f"{channel}-{name}-gap-{dataset}", "channel": channel, "contrast": f"{name}_gap", "dataset": dataset, "models": "pooled", **bootstrap(gap, frames[dataset], 20271100 + len(name) + 10 * offset)})
            matched = contrast_series(data, dataset, models, arm)
            for name, field in (("C2", "distractor_retrieved"), ("C3", "distractor")):
                series = {i: v for i, v in matched[name].items() if clean_block(frames[dataset][i], field)}
                if series:
                    secondary.append({"id": f"{channel}-{name}-clean-{dataset}", "channel": channel, "contrast": f"{name}_clean", "dataset": dataset, "models": "pooled", **bootstrap(series, frames[dataset], 20271400 + int(name[1]) + 10 * offset)})
            skeptic = drop_series(data, dataset, "skeptic", models, arm)
            if skeptic:
                none = drop_series(data, dataset, "no_context", models, arm)
                level_s, level_0 = data.accuracy(dataset, "skeptic", arm, models), data.accuracy(dataset, "no_context", arm, models)
                secondary.append({"id": f"{channel}-skeptic-{dataset}", "channel": channel, "contrast": "skeptic", "dataset": dataset, "models": "pooled", **bootstrap({i: none[i] - skeptic[i] for i in skeptic if i in none}, frames[dataset], 20271500 + 10 * offset)})
                secondary.append({"id": f"{channel}-skeptic-level-{dataset}", "channel": channel, "contrast": "skeptic_level", "dataset": dataset, "models": "pooled", **bootstrap({i: level_s[i] - level_0[i] for i in level_s if i in level_0}, frames[dataset], 20271510 + 10 * offset)})
            for name, series in contrast_series(data, dataset, models, arm, matched_only=False).items():
                if name != "C1" and series:
                    secondary.append({"id": f"{channel}-{name}-all-{dataset}", "channel": channel, "contrast": f"{name}_all", "dataset": dataset, "models": "pooled", **bootstrap(series, frames[dataset], 20271200 + int(name[1]) + 10 * offset)})
            for name, series in matched.items():
                for label in ("TRUE", "FALSE"):
                    sub = {i: v for i, v in series.items() if frames[dataset][i]["gold"] == label}
                    if sub:
                        secondary.append({"id": f"{channel}-{name}-gold-{label}-{dataset}", "channel": channel, "contrast": f"{name}_{label}", "dataset": dataset, "models": "pooled", **bootstrap(sub, frames[dataset], 20271400 + int(name[1]) + 10 * offset + (label == "TRUE"))})
            for name, series in level_series(data, dataset, models, arm).items():
                if series:
                    secondary.append({"id": f"{channel}-{name}-{dataset}", "channel": channel, "contrast": name, "dataset": dataset, "models": "pooled", **bootstrap(series, frames[dataset], 20271300 + int(name[1]) + 10 * offset)})
            d_ret = did_series(data, dataset, "retrieved", models, arm)
            d_inst = did_series(data, dataset, "instruction", models, arm)
            if d_inst:
                secondary.append({"id": f"{channel}-instruction-{dataset}", "channel": channel, "contrast": "instruction", "dataset": dataset, "models": "pooled", **bootstrap(d_inst, frames[dataset], 20270600 + offset)})
            if d_ret and d_inst:
                cmp_series = {i: d_ret[i] - d_inst[i] for i in d_ret if i in d_inst}
                secondary.append({"id": f"{channel}-retrieved-vs-instruction-{dataset}", "channel": channel, "contrast": "retrieved_vs_instruction", "dataset": dataset, "models": "pooled", **bootstrap(cmp_series, frames[dataset], 20270610 + offset)})
            hits = {i for i in d_ret if frames[dataset][i]["retrieval_hit"]}
            for label, keep in (("hit", True), ("miss", False)):
                sub = {i: v for i, v in d_ret.items() if (i in hits) == keep}
                if len(sub) > 10:
                    secondary.append({"id": f"{channel}-association-{label}-{dataset}", "channel": channel, "contrast": f"C1_{label}", "dataset": dataset, "models": "pooled", **bootstrap(sub, frames[dataset], 20270700 + offset)})
            for label, ms in {"pooled": models, **{m: [m] for m in models}}.items():
                short = MODEL_SHORT.get(label, label)
                one_word, reasoned = drop_series(data, dataset, "no_context", ms, arm), drop_series(data, dataset, "reasoned", ms, arm)
                if reasoned:
                    secondary.append({"id": f"{channel}-reasoned-{dataset}-{short}", "channel": channel, "contrast": "reasoned_drop", "dataset": dataset, "models": short, **bootstrap(reasoned, frames[dataset], 20270800 + offset)})
                    gap = {i: reasoned[i] - one_word[i] for i in reasoned if i in one_word}
                    secondary.append({"id": f"{channel}-format-effect-{dataset}-{short}", "channel": channel, "contrast": "format_effect", "dataset": dataset, "models": short, **bootstrap(gap, frames[dataset], 20270810 + offset)})
        notes, peers = drop_series(data, dataset, "no_context", models, "false_text"), drop_series(data, dataset, "no_context", models, "false")
        gap = {i: notes[i] - peers[i] for i in notes if i in peers}
        if gap:
            secondary.append({"id": f"channel-CH-{dataset}", "channel": "both", "contrast": "CH", "dataset": dataset, "models": "pooled", **bootstrap(gap, frames[dataset], 20270190)})
        d_notes, d_peers = did_series(data, dataset, "retrieved", models, "false_text"), did_series(data, dataset, "retrieved", models, "false")
        if d_notes and d_peers:
            social = {i: d_peers[i] - d_notes[i] for i in d_peers if i in d_notes}
            secondary.append({"id": f"attribution-{dataset}", "channel": "both", "contrast": "attributed_minus_unattributed", "dataset": dataset, "models": "pooled", **bootstrap(social, frames[dataset], 20270602)})
    holm(primary)
    for row in primary:
        row["decision"] = decide(row)
    overlap = [{"id": row["id"], "dataset": row["dataset"], **bootstrap(primary_series[row["id"]], frames[row["dataset"]], 20272000 + k, "merged_cluster")} for k, row in enumerate(primary)]
    holm(overlap)
    for row, other in zip(primary, overlap):
        other["decision"] = decide(other)
        row["first_source_decision"], row["overlap_decision"], row["overlap_ci_low"], row["overlap_ci_high"] = row["decision"], other["decision"], other["ci_low"], other["ci_high"]
        row["overlap_dependent"] = row["decision"] in ("supported", "supported_negative") and other["decision"] != row["decision"]
        row["decision"] = combine_decisions(row["decision"], other["decision"])
    holm(secondary)
    for row in primary:
        row["model_dependent"] = model_dependent(row, [r for r in secondary if r["models"] != "pooled"])
    desc = {}
    for dataset in datasets:
        desc[dataset] = {"hit_rate": sum(f["retrieval_hit"] for f in frames[dataset].values()) / len(frames[dataset]), "claims": len(frames[dataset]), "models": {}, "pooled": {}, "clusters": len({f["cluster"] for f in frames[dataset].values()}), "merged_clusters": len({f["merged_cluster"] for f in frames[dataset].values()})}
        for method in SCRIPTED:
            cell = {}
            for arm in ("alone", "false", "true", "false_text"):
                acc = data.accuracy(dataset, method, arm, models)
                if acc:
                    cell[arm] = sum(acc.values()) / len(acc)
            if "alone" in cell and "false" in cell:
                cell["drop"] = cell["alone"] - cell["false"]
            if "alone" in cell and "false_text" in cell:
                cell["notes_drop"] = cell["alone"] - cell["false_text"]
            if "alone" in cell and "true" in cell:
                cell["gain"] = cell["true"] - cell["alone"]
            desc[dataset]["pooled"][method] = cell
        for model in models:
            desc[dataset]["models"][MODEL_SHORT.get(model, model)] = {m: {arm: (sum(a.values()) / len(a)) for arm in ("alone", "false", "true", "false_text") if (a := data.accuracy(dataset, m, arm, [model]))} for m in SCRIPTED}
        desc[dataset]["seed_disagreement"] = {MODEL_SHORT.get(m, m): seed_disagreement(data, dataset, m) for m in models}
        desc[dataset]["spontaneous"] = {m: spontaneous(data, dataset, m, models) for m in ("no_context", "retrieved")}
        desc[dataset]["transitions"] = {c: transitions(data, dataset, c, models) for c in ("retrieved", "gold", "distractor_retrieved")}
        desc[dataset]["excluded_from_C3"] = sum(not (f["oracle"] and f["distractor"]) for f in frames[dataset].values())
        desc[dataset]["matched_retrieved"] = sum(blocks_match(f, "retrieved", "distractor_retrieved") for f in frames[dataset].values())
        desc[dataset]["matched_gold"] = sum(blocks_match(f, "oracle", "distractor") for f in frames[dataset].values())
        desc[dataset]["clean_distractor"] = {b: sum(clean_block(f, b) for f in frames[dataset].values()) for b in ("distractor_retrieved", "distractor")}
        desc[dataset]["parse"] = {f"{method}|{MODEL_SHORT.get(model, model)}|{arm}": {"answers": c[0], "unparsed": c[1], "correct": c[2], "false_answers": c[3], "accuracy_parsed_only": c[2] / max(1, c[0] - c[1])} for (d, method, model, arm), c in sorted(parse.items()) if d == dataset}
        # Majority class among the validated claims, printed beside alone accuracy; and the share of FALSE answers alone per block,
        # which shows how far a block shifts the answer label rather than its correctness.
        desc[dataset]["majority_true"] = sum(f["official_label"] in ("SUPPORTS", "SUPPORT") for f in frames[dataset].values()) / len(frames[dataset])
        shares = {}
        for method in SCRIPTED:
            cells = {MODEL_SHORT.get(m, m): parse.get((dataset, method, m, "alone")) for m in models}
            cells = {k: c for k, c in cells.items() if c and c[0]}
            if cells:
                shares[method] = {"pooled": sum(c[3] for c in cells.values()) / sum(c[0] for c in cells.values()), **{k: c[3] / c[0] for k, c in cells.items()}}
        desc[dataset]["false_answer_share_alone"] = shares
        rates = {k: v["unparsed"] / v["answers"] for k, v in desc[dataset]["parse"].items() if v["answers"]}
        primary_rates = [r for k, r in rates.items() if k.split("|")[0] in SCRIPTED]
        desc[dataset]["unparsed"] = {"max_primary": max(primary_rates), "over_5pct": sorted(k for k, r in rates.items() if r > 0.05), "max_any": max(rates.values()), "max_any_cell": max(rates, key=rates.get)}
        sizes = Counter(f["merged_cluster"] for f in frames[dataset].values())
        desc[dataset]["largest_merged_cluster"] = max(sizes.values())
    apply_floor(primary, desc)
    apply_room(primary, secondary)
    apply_dependence(primary, secondary)
    return {"primary": primary, "secondary": secondary, "overlap": overlap, "group": group_rows(group, frames), "descriptive": desc, "calls": len(scripted)}


def design_macros(frames_dir: Path, datasets: list[str]) -> list[str]:
    """Numbers of the design: stimulus yield, block audit, selection check, feasibility runs and calibration."""
    letters = {"fever": "Fever", "scifact": "Scifact"}
    out = []
    for dataset in datasets:
        L = letters[dataset]
        stats_path = frames_dir / f"{dataset}.rationales.stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            for key, name in (("claims", "Frame"), ("claims_valid", "Valid"), ("drawn", "Drawn"), ("rejected_label_word", "RejLabel"), ("rejected_judge", "RejJudge"), ("rejected_duplicate", "RejDup"), ("rejected_refusal", "RejRefusal"), ("slot_unfilled", "Unfilled")):
                out.append(macro(f"Stim{name}{L}", stats[key]))
            frames = {r["id"]: r for r in map(json.loads, (frames_dir / f"{dataset}.jsonl").open(encoding="utf-8"))}
            valid = [json.loads(l)["id"] for l in (frames_dir / f"{dataset}.rationales.jsonl").open(encoding="utf-8") if json.loads(l)["valid"]]
            out += [macro(f"StimValid{label.title()}{L}", sum(frames[i]["gold"] == label for i in valid)) for label in ("TRUE", "FALSE")]
            audit = {r["id"]: r for r in map(json.loads, (frames_dir / f"{dataset}.audit.jsonl").open(encoding="utf-8"))}
            rows = [audit[i] for i in valid if i in audit]
            agree = {"TRUE": "SUPPORTS", "FALSE": "REFUTES"}
            verdicts = ("SUPPORTS", "REFUTES", "NEITHER")
            gold = [r for r in rows if r.get("oracle") in verdicts]
            out += [macro(f"AuditContam{key}{L}", sum(r.get(field) not in (None, "NEITHER", "EMPTY") for r in rows)) for field, key in (("distractor_retrieved", "Ret"), ("distractor", "Gold"))]
            out += [macro(f"AuditGoldAgree{L}", sum(r["oracle"] == agree[r["gold"]] for r in gold)), macro(f"AuditGoldNeither{L}", sum(r["oracle"] == "NEITHER" for r in gold)), macro(f"AuditGoldAll{L}", len(gold))]
    selection = ROOT / "experiments" / "selection.json"
    if selection.exists():
        rows = [r for r in json.loads(selection.read_text(encoding="utf-8"))["rows"] if r["gold"] == "all"]
        for dataset in datasets:
            for valid, key in ((True, "Kept"), (False, "Rejected")):
                acc = [r["alone_accuracy"] for r in rows if r["dataset"] == dataset and r["validated"] == valid]
                out += [macro(f"Sel{key}Low{letters[dataset]}", min(acc), pct=True), macro(f"Sel{key}High{letters[dataset]}", max(acc), pct=True)]
    for name, key in (("result.first.json", "First"), ("result.json", "Repeat")):
        path = ROOT / "experiments" / "feasibility" / name
        if path.exists():
            feas = json.loads(path.read_text(encoding="utf-8"))
            out.append(macro(f"Feas{key}Claims", feas["claims"]))
            for field, chan in (("notes_drop", "Notes"), ("drop", "Peers")):
                drops = [m["no_context"][field] for m in feas["by_model"].values()]
                out += [macro(f"Feas{key}{chan}Low", min(drops), pct=True), macro(f"Feas{key}{chan}High", max(drops), pct=True)]
            for contrast, label in (("C2", "RetDist"), ("C3", "GoldDist")):
                out.append(macro(f"Feas{key}Notes{label}", f"{100 * feas['contrasts'][f'notes:{contrast}:pooled']['estimate']:+.1f}"))
                out.append(macro(f"Feas{key}Notes{label}Se", feas["contrasts"][f"notes:{contrast}:pooled"]["se_now"], pct=True))
    validation = ROOT / "analysis" / "analysis_validation.json"
    if validation.exists():
        settings = json.loads(validation.read_text(encoding="utf-8"))["family"]["settings"]

        def span(name: str, values: list[float]) -> list[str]:
            return [macro(f"Calib{name}Low", min(values), pct=True), macro(f"Calib{name}High", max(values), pct=True)]

        null, five, minus = ([s for s in settings if s["scenario"] == name] for name in ("global_null", "all_5", "all_minus_2"))
        out += span("FalsePos", [s["any_positive"]["rate"] for s in null])
        out += span("FalseExcl", [s["any_exclusion"]["rate"] for s in five])
        out += span("Power", [r["positive"] for s in five for r in s["rows"].values()])
        out += span("ExclNull", [r["exclusion"] for s in null for r in s["rows"].values()])
        out += span("ExclMinus", [r["exclusion"] for s in minus for r in s["rows"].values()])
        out.append(macro("CalibCoverage", min(s["min_coverage"] for s in settings), pct=True))
        overlap = json.loads(validation.read_text(encoding="utf-8")).get("overlap")
        if overlap:
            for key, name in (("first_source", "First"), ("merged", "Merged")):
                rows = [s[key] for s in overlap["settings"]]
                null_rows = [s[key] for s in overlap["settings"] if s["scenario"] == "global_null"]
                five_rows = [s[key] for s in overlap["settings"] if s["scenario"] == "all_5"]
                out += [macro(f"Overlap{name}FalsePosHigh", max(r["any_positive"]["rate"] for r in null_rows), pct=True), macro(f"Overlap{name}FalseExclHigh", max(r["any_exclusion"]["rate"] for r in five_rows), pct=True), macro(f"Overlap{name}Coverage", min(r["min_coverage"] for r in rows), pct=True)]
            reported = {s["scenario"]: [] for s in overlap["settings"]}
            for s in overlap["settings"]:
                reported[s["scenario"]].append(s["reported_rule"])
            out += span("OverlapRuleFalsePos", [s["any_positive"]["rate"] for s in reported["global_null"]])
            out += span("OverlapRuleFalseExcl", [s["any_exclusion"]["rate"] for s in reported["all_5"]])
            out += span("OverlapPower", [r["positive"] for s in reported["all_5"] for r in s["rows"].values()])
            out += span("OverlapExclNull", [r["exclusion"] for s in reported["global_null"] for r in s["rows"].values()])
    return out


VERDICT = {"supported": "positive", "supported_negative": "5-point benefit excluded", "inconclusive": "inconclusive", "floor_limited": "floor-limited", "room_limited": "room-limited"}


def comparison_rows(result: dict) -> list[dict]:
    comparisons = []
    for row in result["primary"]:
        comparisons.append({"id": row["id"], "method": {"C1": "retrieved", "C2": "retrieved", "C3": "gold"}[row["contrast"]], "baseline": {"C1": "no_context", "C2": "distractor_retrieved", "C3": "distractor_gold"}[row["contrast"]], "dataset": row["dataset"], "metric_id": {"notes": "did_notes", "peers": "did_peers"}[row["channel"]], "evaluator_id": DATASETS[row["dataset"]], "n": row["n"], "test": "paired cluster bootstrap over source clusters, per-claim series, 10,000 resamples, seed family 20270100 and up, two-sided p from the sign of the resampled means", "statistic": row["estimate"], "p_value": row["p_value"], "corrected_p": row["corrected_p"], "correction": "Holm", "effect_size": row["estimate"], "effect_measure": "difference in accuracy-drop difference", "ci_low": row["ci_low"], "ci_high": row["ci_high"], "decision": row["decision"], "first_source_decision": row["first_source_decision"], "merged_cluster_decision": row["overlap_decision"], "merged_ci_low": row["overlap_ci_low"], "merged_ci_high": row["overlap_ci_high"]})
    return comparisons


def primary_table(result: dict) -> list[str]:
    name = {"fever": "FEVER", "scifact": "SciFact"}
    nice = {"C1": "Retrieved vs. no context", "C2": "Retrieved vs. distractor", "C3": "Gold vs. distractor"}
    chan = {"notes": "Notes", "peers": "Peers"}
    lines = ["\\begin{tabular}{lllrrrrlrrrr}", "\\toprule", "Data & Chan. & Contrast & Est. & 95\\% CI & Merged CI & Holm $p$ & Decision & Level & Qwen3.6 & Coder & Gemma \\\\", "\\midrule"]
    per_model = {r["id"]: r["estimate"] for r in result["secondary"] if r["models"] != "pooled"}
    # Level: accuracy under the false channel, first context minus second (L1, L2, L3), printed beside every drop contrast.
    level = {(r["channel"], r["dataset"], r["contrast"]): r["estimate"] for r in result["secondary"] if r["models"] == "pooled" and r["contrast"] in ("L1", "L2", "L3")}
    for row in result["primary"]:
        label = VERDICT[row["decision"]]
        lev = f"{100 * level[(row['channel'], row['dataset'], 'L' + row['contrast'][1])]:+.1f}" if (row['channel'], row['dataset'], 'L' + row['contrast'][1]) in level else "--"
        models = " & ".join(f"{100 * per_model[row['id'] + '-' + short]:+.1f}" if row["id"] + "-" + short in per_model else "--" for short in ("qwen3.6-35B", "qwen3-coder-30B", "gemma-4-31B"))
        lines.append(f"{name[row['dataset']]} & {chan[row['channel']]} & {nice[row['contrast']]} & {100 * row['estimate']:+.1f} & [{100 * row['ci_low']:+.1f}, {100 * row['ci_high']:+.1f}] & [{100 * row['overlap_ci_low']:+.1f}, {100 * row['overlap_ci_high']:+.1f}] & {row['corrected_p']:.3f} & {label}{'$^{m}$' if row['model_dependent'] else ''}{'$^{p}$' if row['position_dependent'] else ''}{'$^{f}$' if row['format_dependent'] else ''}{'$^{c}$' if row['contamination_dependent'] else ''}{'$^{o}$' if row['overlap_dependent'] else ''} & {lev} & {models} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return lines


def write_outputs(result: dict, registry: Path, datasets: list[str], frames_dir: Path | None = None) -> None:
    analysis = ROOT / "analysis"
    (analysis / "tables").mkdir(parents=True, exist_ok=True)
    (analysis / "figures").mkdir(parents=True, exist_ok=True)
    name = {"fever": "FEVER", "scifact": "SciFact"}
    lines = primary_table(result)
    (analysis / "tables" / "primary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 8, "axes.linewidth": 0.6})
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.15), sharey=True)
    order = ["no_context", "instruction", "distractor_retrieved", "retrieved", "distractor_gold", "gold"]
    labels = ["No context", "Instruction", "Distractor (ret.)", "Retrieved", "Distractor (gold)", "Gold"]
    for ax, dataset in zip(axes, datasets):
        cells = result["descriptive"][dataset]["pooled"]
        y = list(range(len(order)))[::-1]
        drop = [100 * cells[m].get("drop", math.nan) for m in order]
        notes = [100 * cells[m].get("notes_drop", math.nan) for m in order]
        gain = [100 * cells[m].get("gain", math.nan) for m in order]
        ax.barh([v + 0.27 for v in y], notes, height=0.26, color="#7a1f16", label="Drop, false notes")
        ax.barh(y, drop, height=0.26, color="#d9755f", label="Drop, false peers")
        ax.barh([v - 0.27 for v in y], gain, height=0.26, color="#2a6fb5", label="Gain, true peers")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_title(name[dataset], fontsize=8)
        ax.set_xlabel("Accuracy points")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, width=0.5)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=6.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.4, rect=(0, 0.08, 1, 1))
    fig.savefig(analysis / "figures" / "drop_gain.pdf")
    plt.close(fig)
    macros = []
    letters = {"fever": "Fever", "scifact": "Scifact"}
    runs = json.loads(registry.read_text(encoding="utf-8"))["runs"]
    run_span = max(r["finished_at"] for r in runs), min(r["started_at"] for r in runs)
    from datetime import datetime
    wall = (datetime.fromisoformat(run_span[0]) - datetime.fromisoformat(run_span[1])).total_seconds() / 3600
    macros += [macro("ComputeWallHours", wall), macro("ComputeCellHours", sum(a.get("seconds", 0) for r in runs for a in r.get("attempts", [])) / 3600), macro("RegisteredRuns", len(runs))]
    for row in result["primary"]:
        base = {"notes": "Notes", "peers": "Peers"}[row["channel"]] + {"C1": "Ret", "C2": "RetDist", "C3": "GoldDist"}[row["contrast"]] + letters[row["dataset"]]
        macros += [macro(base + "Est", row["estimate"], pct=True), macro(base + "Low", row["ci_low"], pct=True), macro(base + "High", row["ci_high"], pct=True), macro(base + "P", row["corrected_p"], 3), macro(base + "Decision", VERDICT[row["decision"]])]
    numbers = {}
    for dataset in datasets:
        d = result["descriptive"][dataset]
        L = letters[dataset]
        macros += [macro(f"Claims{L}", d["claims"]), macro(f"HitRate{L}", d["hit_rate"], pct=True), macro(f"Clusters{L}", d["clusters"]), macro(f"MergedClusters{L}", d["merged_clusters"])]
        for method, key in (("no_context", "None"), ("retrieved", "Ret"), ("instruction", "Inst"), ("gold", "Gold"), ("distractor_retrieved", "DistRet"), ("distractor_gold", "DistGold")):
            for arm, akey in (("alone", "Alone"), ("false", "False"), ("true", "True"), ("false_text", "Notes"), ("drop", "Drop"), ("notes_drop", "NotesDrop"), ("gain", "Gain")):
                value = d["pooled"][method].get(arm)
                if value is not None:
                    macros.append(macro(f"Acc{L}{key}{akey}", value, pct=True))
        macros += [macro(f"SeedDiff{L}{k}", d["seed_disagreement"][m], pct=True) for m, k in (("qwen3.6-35B", "QwenA"), ("qwen3-coder-30B", "QwenB"), ("gemma-4-31B", "Gemma"))]
        macros += [macro(f"Spont{L}None", d["spontaneous"]["no_context"], pct=True), macro(f"Spont{L}Ret", d["spontaneous"]["retrieved"], pct=True), macro(f"ExcludedThree{L}", d["excluded_from_C3"]), macro(f"MatchedRet{L}", d["matched_retrieved"]), macro(f"MatchedGold{L}", d["matched_gold"]), macro(f"CleanRet{L}", d["clean_distractor"]["distractor_retrieved"]), macro(f"CleanGold{L}", d["clean_distractor"]["distractor"])]
        macros += [macro(f"Majority{L}", d["majority_true"], pct=True), macro(f"LargestCluster{L}", d["largest_merged_cluster"]),
                   macro(f"UnparsedMaxPrimary{L}", d["unparsed"]["max_primary"], pct=True), macro(f"UnparsedMax{L}", d["unparsed"]["max_any"], pct=True),
                   macro(f"UnparsedOverFive{L}", len(d["unparsed"]["over_5pct"]))]
        worst = d["parse"][d["unparsed"]["max_any_cell"]]
        macros += [macro(f"UnparsedWorstAcc{L}", worst["correct"] / worst["answers"], pct=True), macro(f"UnparsedWorstParsedAcc{L}", worst["accuracy_parsed_only"], pct=True)]
        for method, key in (("no_context", "None"), ("retrieved", "Ret"), ("distractor_retrieved", "DistRet"), ("gold", "Gold"), ("distractor_gold", "DistGold")):
            share = d["false_answer_share_alone"].get(method)
            if share:
                macros += [macro(f"FalseShare{L}{key}", share["pooled"], pct=True), macro(f"FalseShare{L}{key}Gemma", share["gemma-4-31B"], pct=True)]
        for ctx, ckey in (("retrieved", "Ret"), ("gold", "Gold"), ("distractor_retrieved", "DistRet")):
            for tname, tkey in (("harmful", "Harm"), ("harmful_notes", "HarmNotes"), ("beneficial", "Ben")):
                t = d["transitions"][ctx][tname]
                macros += [macro(f"{tkey}{L}{ckey}", t["rate_context"], pct=True), macro(f"{tkey}{L}{ckey}None", t["rate_no_context"], pct=True), macro(f"{tkey}{L}{ckey}N", t["stratum_claims"])]
                if tname == "beneficial":
                    macros += [macro(f"{tkey}{L}{ckey}DiffLow", t["difference"]["ci_low"], pct=True), macro(f"{tkey}{L}{ckey}DiffHigh", t["difference"]["ci_high"], pct=True)]
    keys = {"C1_TRUE": "RetTrue", "C1_FALSE": "RetFalse", "C2_TRUE": "RetDistTrue", "C2_FALSE": "RetDistFalse", "C3_TRUE": "GoldDistTrue", "C3_FALSE": "GoldDistFalse", "instruction": "Inst", "retrieved_vs_instruction": "RetInst", "attributed_minus_unattributed": "Attr", "C1_hit": "Hit", "C1_miss": "Miss", "reasoned_drop": "Reasoned", "format_effect": "Format", "C2h": "RetDistCond", "C3h": "GoldDistCond", "C3_last": "GoldDistLast", "C3_reasoned": "GoldDistReasoned", "C2_clean": "RetDistClean", "C3_clean": "GoldDistClean", "skeptic": "Skeptic", "skeptic_level": "SkepticLevel", "C3_last_gap": "LastGap", "C3_reasoned_gap": "ReasonedGap", "C2_all": "RetDistAll", "C3_all": "GoldDistAll", "L1": "LevelRet", "L2": "LevelRetDist", "L3": "LevelGoldDist", "CH": "ChannelGap"}
    for row in result["secondary"]:
        if row["contrast"] in keys and (row["models"] == "pooled" or row["contrast"] in {"reasoned_drop", "format_effect"}):
            model = "" if row["models"] == "pooled" else {"qwen3.6-35B": "QwenA", "qwen3-coder-30B": "QwenB", "gemma-4-31B": "Gemma"}[row["models"]]
            key = {"notes": "Notes", "peers": "Peers", "both": ""}[row["channel"]] + keys[row["contrast"]] + model + letters[row["dataset"]]
            macros += [macro(key + "Est", row["estimate"], pct=True), macro(key + "Low", row["ci_low"], pct=True), macro(key + "High", row["ci_high"], pct=True), macro(key + "N", row["n"])]
            if row.get("corrected_p") is not None:
                macros.append(macro(key + "Holm", row["corrected_p"], digits=2))
    for row in result["group"]:
        L = letters[row["dataset"]]
        key = {"group_none": "GroupNone", "group_retrieved": "GroupRet", "vote_retrieved_vs_vote_none": "GroupEvidence"}[row["method"]] + L
        macros += [macro(key + "Est", row["estimate"], pct=True), macro(key + "Low", row["ci_low"], pct=True), macro(key + "High", row["ci_high"], pct=True)]
        if "vote_independent" in row:
            macros += [macro(key + "Ind", row["vote_independent"], pct=True), macro(key + "Del", row["vote_deliberated"], pct=True)]
    macros.append(macro("OverlapDependentRows", sum(row["overlap_dependent"] for row in result["primary"])))
    macros += design_macros(frames_dir or GROUND_TRUTH / "frames", datasets)
    for line in macros:
        head, _, tail = line.partition("}{")
        text = tail[:-1]
        try:
            numbers[head.split("\\")[-1]] = float(text)
        except ValueError:
            numbers[head.split("\\")[-1]] = text
    (analysis / "generated_numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    comparisons = comparison_rows(result)
    manifest = {"generated_at": os.environ.get("PAPER_GENERATED_AT", ""), "source_registry_sha256": sha256(registry), "analysis_script_sha256": sha256(Path(__file__)), "bootstrap_resamples": RESAMPLES,
                "achieved_evidence": {"datasets": len(datasets), "baselines": 8, "instances": sum(result["descriptive"][d]["claims"] for d in datasets), "seeds": 3, "models_or_configs": len(MODELS), "stat_tests": True, "effect_sizes": True, "independent_ground_truth": True, "references": len(re.findall(r"^@\w+\{", (ROOT / "manuscript" / "refs.bib").read_text(encoding="utf-8"), re.M)) if (ROOT / "manuscript" / "refs.bib").exists() else 0, "figures": 1, "tables": 1},
                "comparisons": comparisons, "secondary": result["secondary"], "overlap": result["overlap"], "group": result["group"], "descriptive": result["descriptive"],
                "tables": [{"id": "tbl-primary", "path": "analysis/tables/primary.tex", "sha256": sha256(analysis / "tables" / "primary.tex")}], "figures": [{"id": "fig-drop-gain", "path": "analysis/figures/drop_gain.pdf", "sha256": sha256(analysis / "figures" / "drop_gain.pdf")}], "numbers": numbers}
    (analysis / "results_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(ROOT / "experiments" / "registry.json"))
    ap.add_argument("--kinds", default="full,baseline,sensitivity")
    ap.add_argument("--frames", default=str(GROUND_TRUTH / "frames"))
    ap.add_argument("--print-only", action="store_true", help="print the contrasts without writing analysis/")
    args = ap.parse_args()
    datasets = list(DATASETS)
    result = analyse(Path(args.registry), Path(args.frames), set(args.kinds.split(",")), MODELS, datasets)
    for row in result["primary"]:
        print(row["id"], f"{100 * row['estimate']:+.1f} [{100 * row['ci_low']:+.1f}, {100 * row['ci_high']:+.1f}] n={row['n']} clusters={row['clusters']} p={row['p_value']:.4f} holm={row['corrected_p']:.4f} {row['decision']}")
    if not args.print_only:
        write_outputs(result, Path(args.registry), datasets, Path(args.frames))


if __name__ == "__main__":
    main()
