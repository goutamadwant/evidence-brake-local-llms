#!/usr/bin/env python3
"""Calibration of the decision procedure before the study. Uses no study outcome. Writes analysis/analysis_validation.json.

Single rows: on the real cluster structure of the matched claim sets, simulate Gaussian per-claim contrasts with within-cluster correlation
and known effects, and measure how often the cluster bootstrap rejects zero and how often the exclusion bound fires.

The family: simulate binary answers of three models per claim, context and arm, with latent claim and cluster effects shared across channels, so
the twelve rows are dependent and their contrasts bounded and skewed, and pass each simulated study through the production bootstrap, Holm and
decide. Acceptance, fixed before this ran: any-false-positive under the global null and any-false-exclusion under a true 5-point benefit in all rows
at most 0.05 + 2 Monte Carlo standard errors, and per-row coverage at least 0.90. A failure would drop the simultaneous claim."""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import sys
from pathlib import Path
from statistics import NormalDist

os.environ.setdefault("PAPER_RESAMPLES", "2000")  # the family loop passes each row through the production bootstrap 500 times per scenario

import numpy as np

from analyze import ALPHA, DATASETS, GROUND_TRUTH, PRIMARY_ROWS, RESAMPLES as FAMILY_RESAMPLES, ROOT, SESOI, blocks_match, bootstrap, combine_decisions, decide, holm, merged_clusters

SIMS, RESAMPLES = 1000, 2000
SD = 0.45  # the largest notes contrast sd of the feasibility run on the validated stimuli


def cluster_index(frames: list[dict]) -> np.ndarray:
    names = sorted({f["cluster"] for f in frames})
    pos = {n: i for i, n in enumerate(names)}
    return np.array([pos[f["cluster"]] for f in frames])


def simulate(idx: np.ndarray, effect: float, icc: float, rng: np.random.Generator) -> dict:
    k, n = idx.max() + 1, len(idx)
    sizes = np.bincount(idx, minlength=k).astype(float)
    reject, exclude, cover = 0, 0, 0
    for _ in range(SIMS):
        values = effect + SD * (np.sqrt(icc) * rng.standard_normal(k)[idx] + np.sqrt(1 - icc) * rng.standard_normal(n))
        sums = np.bincount(idx, weights=values, minlength=k)
        draw = rng.integers(0, k, size=(RESAMPLES, k))
        stats = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)
        low, high = np.quantile(stats, [0.025, 0.975])
        reject += low > 0 or high < 0
        cover += low <= effect <= high
        exclude += np.quantile(stats, 1 - ALPHA / PRIMARY_ROWS) < SESOI
    return {"reject_zero": reject / SIMS, "ci_coverage": cover / SIMS, "familywise_exclusion_fires": exclude / SIMS}


FAMILY_SIMS = 500
CONTEXTS = ("no_context", "retrieved", "distractor_retrieved", "gold", "distractor_gold")
ARMS = {"notes": 1, "peers": 2}
NORMAL = NormalDist()
# accuracy alone per context and the no-context drop per channel, from the feasibility run on validated stimuli
ALONE = {"no_context": 0.66, "retrieved": 0.77, "distractor_retrieved": 0.57, "gold": 0.87, "distractor_gold": 0.60}
DROP = {"notes": 0.49, "peers": 0.19}
SCENARIOS = {
    "global_null": {"C1": 0.0, "C2": 0.0, "C3": 0.0},
    "all_5": {"C1": SESOI, "C2": SESOI, "C3": SESOI},
    "all_minus_2": {"C1": -0.02, "C2": -0.02, "C3": -0.02},
    "feasibility_like": {"C1": 0.15, "C2": 0.0, "C3": 0.0},
    "c1_15_rest_5": {"C1": 0.15, "C2": SESOI, "C3": SESOI},
}
CLAIM_SD, CONTEXT_SD = 2.0, 0.5  # latent scales: simulated contrast sds of 42 to 52 points, at or above the 23 to 45 of the feasibility run


def family_targets(effects: dict[str, float], shift: float) -> np.ndarray:
    """Marginal accuracy per context and arm whose contrasts equal the effects exactly: C1 fixes the retrieved drop; C2 and C3 the distractor drops."""
    target = np.zeros((len(CONTEXTS), 3))
    for ci, context in enumerate(CONTEXTS):
        target[ci, 0] = min(0.97, ALONE[context] + shift)
    for channel, arm in ARMS.items():
        none = DROP[channel]
        retrieved = none - effects["C1"]
        drops = {"no_context": none, "retrieved": retrieved, "distractor_retrieved": none - (effects["C1"] - effects["C2"]), "gold": none - effects["C1"], "distractor_gold": none - (effects["C1"] - effects["C3"])}
        for ci, context in enumerate(CONTEXTS):
            target[ci, arm] = target[ci, 0] - drops[context]
    if target.min() <= 0.01:
        raise ValueError("a target accuracy left the unit interval")
    return target


def one_family(job: tuple) -> list[dict]:
    """One simulated study: twelve rows through the production bootstrap, Holm and decide."""
    scenario, icc, shift, sim, sets, *mode = job
    overlap = bool(mode)  # latent effects shared across source-overlap clusters; both clusterings then read the same simulated answers
    rng = np.random.default_rng([20270300, sim, abs(hash_text(scenario)) % 100000, int(icc * 10), int(shift * 100)])
    effects = SCENARIOS[scenario]
    target = family_targets(effects, shift)
    rows, merged_rows = [], []
    key = "merged_cluster" if overlap else "cluster"
    for dataset, claims in sets.items():
        names = sorted({c[key] for c in claims})
        position = {n: i for i, n in enumerate(names)}
        idx = np.array([position[c[key]] for c in claims])
        n = len(claims)
        latent = np.sqrt(icc) * rng.standard_normal(len(names))[idx] + np.sqrt(1 - icc) * rng.standard_normal(n)
        scale = math.sqrt(1 + CLAIM_SD**2 + CONTEXT_SD**2)
        mean = np.vectorize(NORMAL.inv_cdf)(target) * scale
        sway = rng.standard_normal((n, len(ARMS) + 1))
        context_noise = rng.standard_normal((n, len(CONTEXTS), 3))
        # three models answer on a shared claim and context signal, with their own binary noise
        answers = np.empty((n, len(CONTEXTS), 3, 3))
        for arm in range(3):
            claim_part = CLAIM_SD * (latent if arm == 0 else (latent + sway[:, arm]) / np.sqrt(2))
            for ci in range(len(CONTEXTS)):
                signal = mean[ci, arm] + claim_part + CONTEXT_SD * context_noise[:, ci, arm]
                answers[:, ci, arm] = signal[:, None] + rng.standard_normal((n, 3)) > 0
        accuracy = answers.mean(axis=3)
        lookup = {c: i for i, c in enumerate(CONTEXTS)}
        frames = {c["id"]: c for c in claims}
        family = []
        for channel, arm in ARMS.items():
            drop = accuracy[:, :, 0] - accuracy[:, :, arm]
            did = {c: drop[:, lookup["no_context"]] - drop[:, lookup[c]] for c in CONTEXTS[1:]}
            series = {"C1": did["retrieved"], "C2": did["retrieved"] - did["distractor_retrieved"], "C3": did["gold"] - did["distractor_gold"]}
            masks = {"C1": [True] * n, "C2": [c["match_retrieved"] for c in claims], "C3": [c["match_gold"] for c in claims]}
            for name, values in series.items():
                kept = {c["id"]: float(v) for c, v, keep in zip(claims, values, masks[name]) if keep}
                family.append({"row": f"{channel}-{name}-{dataset}", "true": effects[name], "sd": float(np.std(list(kept.values()))), **bootstrap(kept, frames, int(rng.integers(2**31)))})
                if overlap:
                    merged_rows.append({"row": f"{channel}-{name}-{dataset}", "true": effects[name], "sd": family[-1]["sd"], **bootstrap(kept, frames, int(rng.integers(2**31)), "merged_cluster")})
        rows += family
    for group in (rows, merged_rows) if overlap else (rows,):
        holm(group)
        for row in group:
            row["decision"] = decide(row)
    return (rows, merged_rows) if overlap else rows


def hash_text(text: str) -> int:
    return sum(ord(ch) * (i + 1) for i, ch in enumerate(text))


def claim_sets() -> dict:
    sets = {}
    for dataset in DATASETS:
        frames = [json.loads(l) for l in (GROUND_TRUTH / "frames" / f"{dataset}.jsonl").open(encoding="utf-8")]
        valid = {r["id"] for r in map(json.loads, (GROUND_TRUTH / "frames" / f"{dataset}.rationales.jsonl").open(encoding="utf-8")) if r["valid"]}
        kept = [f for f in frames if f["id"] in valid]
        merged = merged_clusters({f["id"]: f for f in kept})
        sets[dataset] = [{"id": f["id"], "cluster": f["cluster"], "merged_cluster": merged[f["id"]], "match_retrieved": blocks_match(f, "retrieved", "distractor_retrieved"), "match_gold": blocks_match(f, "oracle", "distractor")} for f in kept]
    return sets


def rates(sims: list[list[dict]]) -> dict:
    out = {}
    for name, test in {"any_positive": lambda r: r["decision"] == "supported", "any_exclusion": lambda r: r["decision"] == "supported_negative"}.items():
        value = sum(any(test(r) for r in family) for family in sims) / len(sims)
        out[name] = {"rate": value, "mc_se": math.sqrt(max(value * (1 - value), 0.01) / len(sims))}
    out["min_coverage"] = min(sum(family[k]["ci_low"] <= family[k]["true"] <= family[k]["ci_high"] for family in sims) / len(sims) for k in range(len(sims[0])))
    return out


def overlap_validation() -> dict:
    """Latent effects shared across source-overlap clusters. The same answers pass through the first-source and the merged clustering. Acceptance, fixed
    before this ran: the merged clustering meets the familywise and coverage rule of the family validation."""
    sets = claim_sets()
    out: dict = {"sims": FAMILY_SIMS, "resamples": FAMILY_RESAMPLES, "settings": []}
    with multiprocessing.get_context("fork").Pool(max(1, (os.cpu_count() or 2) - 3)) as pool:
        for scenario, icc in [(s, icc) for s in ("global_null", "all_5") for icc in (0.3, 0.7)]:
            sims = pool.map(one_family, [(scenario, icc, 0.0, sim, sets, "overlap") for sim in range(FAMILY_SIMS)], chunksize=10)
            first, merged = [s[0] for s in sims], [s[1] for s in sims]
            changed = sum(a["decision"] != b["decision"] for f, m in zip(first, merged) for a, b in zip(f, m)) / (FAMILY_SIMS * len(first[0]))
            # the reported rule: a conclusion stands only if both clusterings reach it
            both = [[{**a, "decision": combine_decisions(a["decision"], b["decision"])} for a, b in zip(f, m)] for f, m in zip(first, merged)]
            per_row = {r["row"]: {"positive": sum(family[k]["decision"] == "supported" for family in both) / FAMILY_SIMS, "exclusion": sum(family[k]["decision"] == "supported_negative" for family in both) / FAMILY_SIMS} for k, r in enumerate(both[0])}
            setting = {"scenario": scenario, "icc": icc, "first_source": rates(first), "merged": rates(merged), "reported_rule": {**rates(both), "rows": per_row}, "row_decision_changes": changed}
            out["settings"].append(setting)
            print({k: v for k, v in setting.items() if k != "reported_rule"}, {k: v for k, v in setting["reported_rule"].items() if k != "rows"}, flush=True)
    null = [s["merged"] for s in out["settings"] if s["scenario"] == "global_null"]
    five = [s["merged"] for s in out["settings"] if s["scenario"] == "all_5"]
    out["acceptance"] = {"familywise_false_positive": all(s["any_positive"]["rate"] <= ALPHA + 2 * s["any_positive"]["mc_se"] for s in null),
                         "familywise_false_exclusion": all(s["any_exclusion"]["rate"] <= ALPHA + 2 * s["any_exclusion"]["mc_se"] for s in five),
                         "coverage": all(s["merged"]["min_coverage"] >= 0.90 for s in out["settings"])}
    print(out["acceptance"], flush=True)
    return out


def family_validation() -> dict:
    sets = claim_sets()
    settings = [(s, icc, 0.0) for s in SCENARIOS for icc in (0.3, 0.7)] + [("global_null", 0.3, 0.08), ("all_5", 0.3, 0.08)]
    out: dict = {"sims": FAMILY_SIMS, "resamples": FAMILY_RESAMPLES, "settings": []}
    with multiprocessing.get_context("fork").Pool(max(1, (os.cpu_count() or 2) - 3)) as pool:
        for scenario, icc, shift in settings:
            sims = pool.map(one_family, [(scenario, icc, shift, sim, sets) for sim in range(FAMILY_SIMS)], chunksize=10)
            rates = {}
            for name, test in {"any_positive": lambda r: r["decision"] == "supported", "any_exclusion": lambda r: r["decision"] == "supported_negative"}.items():
                value = sum(any(test(r) for r in family) for family in sims) / FAMILY_SIMS
                rates[name] = {"rate": value, "mc_se": math.sqrt(max(value * (1 - value), 0.01) / FAMILY_SIMS)}
            names = [r["row"] for r in sims[0]]
            per_row = {}
            for k, name in enumerate(names):
                rows = [family[k] for family in sims]
                per_row[name] = {"positive": sum(r["decision"] == "supported" for r in rows) / FAMILY_SIMS, "exclusion": sum(r["decision"] == "supported_negative" for r in rows) / FAMILY_SIMS,
                                 "coverage": sum(r["ci_low"] <= r["true"] <= r["ci_high"] for r in rows) / FAMILY_SIMS, "mean_sd": sum(r["sd"] for r in rows) / FAMILY_SIMS, "mean_half_width": sum((r["ci_high"] - r["ci_low"]) / 2 for r in rows) / FAMILY_SIMS}
            setting = {"scenario": scenario, "icc": icc, "alone_shift": shift, **rates, "min_coverage": min(v["coverage"] for v in per_row.values()), "rows": per_row}
            out["settings"].append(setting)
            print({k: setting[k] for k in ("scenario", "icc", "alone_shift", "any_positive", "any_exclusion", "min_coverage")}, flush=True)
    null = [s for s in out["settings"] if s["scenario"] == "global_null"]
    five = [s for s in out["settings"] if s["scenario"] == "all_5"]
    out["acceptance"] = {"familywise_false_positive": all(s["any_positive"]["rate"] <= ALPHA + 2 * s["any_positive"]["mc_se"] for s in null),
                         "familywise_false_exclusion": all(s["any_exclusion"]["rate"] <= ALPHA + 2 * s["any_exclusion"]["mc_se"] for s in five),
                         "coverage": all(s["min_coverage"] >= 0.90 for s in out["settings"])}
    print(out["acceptance"], flush=True)
    return out


def main() -> None:
    path = ROOT / "analysis" / "analysis_validation.json"
    if "--overlap-only" in sys.argv:
        out = json.loads(path.read_text(encoding="utf-8"))
        out["overlap"] = overlap_validation()
        path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        return
    rng = np.random.default_rng(20270201)
    out: dict = {"sims": SIMS, "resamples": RESAMPLES, "sd": SD, "rows": []}
    for dataset in DATASETS:
        rows = [json.loads(l) for l in (GROUND_TRUTH / "frames" / f"{dataset}.jsonl").open(encoding="utf-8")]
        valid = {r["id"] for r in map(json.loads, (GROUND_TRUTH / "frames" / f"{dataset}.rationales.jsonl").open(encoding="utf-8")) if r["valid"]}
        rows = [r for r in rows if r["id"] in valid]
        for contrast, (evidence, distractor) in {"C2": ("retrieved", "distractor_retrieved"), "C3": ("oracle", "distractor")}.items():
            idx = cluster_index([r for r in rows if blocks_match(r, evidence, distractor)])
            for icc in (0.0, 0.3, 0.7):
                for effect in (0.0, SESOI, -0.02):
                    row = {"dataset": dataset, "contrast": contrast, "claims": len(idx), "clusters": int(idx.max() + 1), "icc": icc, "true_effect": effect, **simulate(idx, effect, icc, rng)}
                    out["rows"].append(row)
                    print({k: (float(v) if isinstance(v, np.floating) else v) for k, v in row.items()}, flush=True)
    out["family"] = family_validation()
    out["overlap"] = overlap_validation()
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
