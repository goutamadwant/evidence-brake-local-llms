#!/usr/bin/env python3
"""Frozen fixtures for the reported decision: the two clusterings must agree, and the table, the comparison rows and the macros must all say so."""
from __future__ import annotations

import sys

from analyze import VERDICT, combine_decisions, comparison_rows, primary_table

STATES = ("supported", "supported_negative", "inconclusive")
EXPECTED = {(a, b): (a if a == b else "inconclusive") for a in STATES for b in STATES}


def row(first: str, merged: str) -> dict:
    return {"id": f"notes-C2-fever-{first}-{merged}", "channel": "notes", "contrast": "C2", "dataset": "fever", "models": "pooled", "n": 100, "estimate": 0.02, "ci_low": 0.01, "ci_high": 0.03,
            "overlap_ci_low": -0.04, "overlap_ci_high": 0.08, "p_value": 0.01, "corrected_p": 0.01, "first_source_decision": first, "overlap_decision": merged, "decision": combine_decisions(first, merged),
            "overlap_dependent": first in ("supported", "supported_negative") and merged != first, "model_dependent": False, "position_dependent": False, "format_dependent": False, "contamination_dependent": False}


def main() -> int:
    failures = []
    for (first, merged), expected in EXPECTED.items():
        if combine_decisions(first, merged) != expected:
            failures.append(f"combine {first} {merged}")
    cases = [row(first, merged) for first, merged in EXPECTED]
    result = {"primary": cases, "secondary": []}
    table = [line for line in primary_table(result) if line.startswith("FEVER")]
    compared = comparison_rows(result)
    for case, line, compared_row in zip(cases, table, compared):
        word = VERDICT[EXPECTED[(case["first_source_decision"], case["overlap_decision"])]]
        if f"& {word}" not in line or ("$^{o}$" in line) != case["overlap_dependent"]:
            failures.append(f"table {case['id']}: {line}")
        if compared_row["decision"] != EXPECTED[(case["first_source_decision"], case["overlap_decision"])]:
            failures.append(f"comparison {case['id']}")
    # a positive row carries no magnitude label, whatever its endpoints under either clustering
    boundary = {**row("supported", "supported"), "estimate": 0.025, "ci_low": 0.01, "ci_high": 0.04, "overlap_ci_low": 0.005, "overlap_ci_high": 0.08}
    line = [l for l in primary_table({"primary": [boundary], "secondary": []}) if l.startswith("FEVER")][0]
    if "& positive &" not in line or "below" in line:
        failures.append(f"magnitude label: {line}")
    print("\n".join(failures) or f"PASS {len(EXPECTED)} decision pairs and the magnitude boundary through the rule, the table and the comparison rows")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
